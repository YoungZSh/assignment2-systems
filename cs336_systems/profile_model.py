import argparse
import math
import torch
from einops import einsum, rearrange
import einx
from jaxtyping import Float, Bool, Int
from torch import Tensor
import cs336_basics.model
from cs336_basics.model import BasicsTransformerLM, SwiGLU, CausalMultiHeadSelfAttention, TransformerBlock, RMSNorm, Linear, Embedding, RotaryEmbedding
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import softmax

# --- Monkey Patching Start ---

# 1. Patch scaled_dot_product_attention (Granular Attention Internal)
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    with torch.cuda.nvtx.range("scaled_dot_product_attention"):
        d_k = K.shape[-1]
        
        with torch.cuda.nvtx.range("computing attention scores"):
            attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

            if mask is not None:
                attention_scores = torch.where(mask, attention_scores, float("-inf"))

        with torch.cuda.nvtx.range("computing softmax"):
            attention_weights = softmax(attention_scores, dim=-1)

        with torch.cuda.nvtx.range("final matmul"):
            output = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
            
        return output

# 2. Patch SwiGLU (MLP)
def annotated_swiglu_forward(self, x):
    with torch.cuda.nvtx.range("MLP (SwiGLU)"):
        return self.w2(cs336_basics.model.silu(self.w1(x)) * self.w3(x))

# 3. Patch CausalMultiHeadSelfAttention (Whole Attention Block)
def annotated_attention_forward(self, x: Float[Tensor, " ... seq d_k"], token_positions: Int[Tensor, " ... seq"] | None = None) -> Float[Tensor, " ... seq d_v"]:
    with torch.cuda.nvtx.range("Attention Block"):
        *b, sequence_length, d_model = x.size()
        assert d_model == self.d_model

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q, K, V = (
            rearrange(X, "... seq (heads d) -> ... heads seq d", heads=self.num_heads)
            for X in (Q, K, V)
        )

        if token_positions is None:
            token_positions = einx.rearrange("seq -> b... seq", torch.arange(sequence_length, device=x.device), b=[1] * len(b))

        token_positions = rearrange(token_positions, "... seq -> ... 1 seq")

        Q = self.positional_encoder(Q, token_positions)
        K = self.positional_encoder(K, token_positions)

        seq = torch.arange(sequence_length, device=x.device)
        qi = einx.rearrange('query -> b... 1 query 1', seq, b=[1] * len(b))
        kj = einx.rearrange('key   -> b... 1 1   key', seq, b=[1] * len(b))
        causal_mask = qi >= kj

        attn_output = cs336_basics.model.scaled_dot_product_attention(K=K, Q=Q, V=V, mask=causal_mask)

        attn_output = rearrange(attn_output, "batch heads seq d_v -> batch seq (heads d_v)").contiguous()
        output = self.output_proj(attn_output)
        return output

# 4. Patch TransformerBlock (Layer Wrapper)
def annotated_block_forward(self, x: torch.Tensor):
    with torch.cuda.nvtx.range("TransformerLayer"):
        x_attn = self.attn(self.ln1(x))
        attn_sublayer_output = x + x_attn

        x_ffn = self.ffn(self.ln2(attn_sublayer_output))
        ffn_sublayer_output = attn_sublayer_output + x_ffn
        return ffn_sublayer_output

# 5. Patch RMSNorm
def annotated_rmsnorm_forward(self, x):
    with torch.cuda.nvtx.range("RMSNorm"):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * rms
        return (self.weight * x).to(in_dtype)

# 6. Patch Linear (Crucial for Projections & LM Head)
def annotated_linear_forward(self, x: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
    with torch.cuda.nvtx.range(f"Linear ({self.extra_repr()})"):
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

# 7. Patch Embedding
def annotated_embedding_forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]:
    with torch.cuda.nvtx.range("Embedding"):
        return self.weight[token_ids, :]

# 8. Patch RotaryEmbedding (RoPE)
def annotated_rope_forward(self, x: Float[Tensor, " ... seq d"], pos_ids: Int[Tensor, " ... seq"]) -> Float[Tensor, " ... seq d"]:
    with torch.cuda.nvtx.range("RotaryEmbedding (RoPE)"):
        x1, x2 = rearrange(x, '... (half_d xy) -> xy ... half_d', xy=2)
        # Using the einx logic from original model.py
        # self._freq_cis_cache is available on self
        cos, sin = einx.get_at('cos_sin [pos] half_dim, ... -> cos_sin ... half_dim', self._freq_cis_cache, pos_ids)
        x1_rot = cos * x1 - sin * x2
        x2_rot = sin * x1 + cos * x2
        result = einx.rearrange('... x_half, ... x_half -> ... (x_half (1 + 1))', x1_rot, x2_rot).contiguous()
        return result

print("Applying ALL monkey patches (Attention, MLP, Norm, Linear, Embedding, RoPE)...")
cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
cs336_basics.model.SwiGLU.forward = annotated_swiglu_forward
cs336_basics.model.CausalMultiHeadSelfAttention.forward = annotated_attention_forward
cs336_basics.model.TransformerBlock.forward = annotated_block_forward
cs336_basics.model.RMSNorm.forward = annotated_rmsnorm_forward
cs336_basics.model.Linear.forward = annotated_linear_forward
cs336_basics.model.Embedding.forward = annotated_embedding_forward
cs336_basics.model.RotaryEmbedding.forward = annotated_rope_forward

# --- Monkey Patching End ---

def main():
    parser = argparse.ArgumentParser(description="Profile Transformer model with NVTX ranges")
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--d_ff", type=int, required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--num_heads", type=int, required=True)
    parser.add_argument("--context_length", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--warmup_iterations", type=int, default=5)
    parser.add_argument("--benchmark_iterations", type=int, default=10)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])
    
    args = parser.parse_args()

    # Set dtype
    if args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}, dtype: {dtype}")

    # Initialize model
    print(f"Initializing model (d_model={args.d_model}, layers={args.num_layers})...")
    model = BasicsTransformerLM(
        vocab_size=50257,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=10000.0,
    ).to(device=device, dtype=dtype)

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=1e-3)

    # Create dummy data
    input_ids = torch.randint(0, 50257, (args.batch_size, args.context_length), device=device)
    
    # Warmup
    print(f"Starting warmup ({args.warmup_iterations} steps)...")
    model.train()
    for _ in range(args.warmup_iterations):
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = logits.mean() # simple dummy loss
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

    # Benchmark
    print(f"Starting benchmark ({args.benchmark_iterations} steps)...")
    for i in range(args.benchmark_iterations):
        with torch.cuda.nvtx.range("Iteration"):
            optimizer.zero_grad()
            
            with torch.cuda.nvtx.range("Forward"):
                logits = model(input_ids)
                loss = logits.mean()
            
            with torch.cuda.nvtx.range("Backward"):
                loss.backward()
            
            with torch.cuda.nvtx.range("Optimizer"):
                optimizer.step()
        
        torch.cuda.synchronize()
    
    print("Profiling completed.")

if __name__ == "__main__":
    main()
