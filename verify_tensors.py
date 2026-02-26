#!/usr/bin/env python3
"""Verify PowerCoder-3B tensor values for comparison with WASM debug output."""
import json
import struct
from pathlib import Path
from safetensors import safe_open

MODEL_DIR = Path("/Users/au/models/powercoder-3b")

# Load index
with open(MODEL_DIR / "model.safetensors.index.json") as f:
    index = json.load(f)

print("=== Model Info from tensor shapes ===")
shard1 = str(MODEL_DIR / "model-00001-of-00003.safetensors")
with safe_open(shard1, framework="numpy") as f:
    embed = f.get_tensor("model.embed_tokens.weight")
    print(f"embed shape: {embed.shape}")  # [vocab, emb]
    print(f"num_vocab={embed.shape[0]}, num_emb={embed.shape[1]}")

    q = f.get_tensor("model.layers.0.self_attn.q_proj.weight")
    k = f.get_tensor("model.layers.0.self_attn.k_proj.weight")
    g = f.get_tensor("model.layers.0.self_attn.g_proj.weight")
    cfc = f.get_tensor("model.layers.0.mlp.c_fc.weight")

    print(f"q_proj shape: {q.shape}")
    print(f"k_proj shape: {k.shape}")
    print(f"g_proj shape: {g.shape}")
    print(f"c_fc shape: {cfc.shape}")

    num_kv_head = g.shape[0]
    head_dim = k.shape[0] // num_kv_head
    num_head = q.shape[0] // head_dim
    print(f"num_kv_head={num_kv_head}, head_dim={head_dim}, num_head={num_head}")
    print(f"intermediate_size={cfc.shape[0]}")

print("\n=== Embed token=0 first 8 values ===")
with safe_open(shard1, framework="numpy") as f:
    embed = f.get_tensor("model.embed_tokens.weight")
    # Embed is stored as F16 in safetensors
    print(f"dtype: {embed.dtype}")
    vals = embed[0, :8]
    print(f"embed[0][0:8] = {[f'{v:.6f}' for v in vals.astype(float)]}")
    vals1 = embed[1, :8]
    print(f"embed[1][0:8] = {[f'{v:.6f}' for v in vals1.astype(float)]}")

print("\n=== Layer 0 input_layernorm.weight first 8 values ===")
with safe_open(shard1, framework="numpy") as f:
    norm = f.get_tensor("model.layers.0.input_layernorm.weight")
    print(f"dtype: {norm.dtype}, shape: {norm.shape}")
    print(f"norm[0:8] = {[f'{v:.6f}' for v in norm[:8].astype(float)]}")

print("\n=== Head norm (model.norm.weight) first 8 values ===")
# Find which shard has model.norm.weight
shard_file = index["weight_map"]["model.norm.weight"]
shard_path = str(MODEL_DIR / shard_file)
with safe_open(shard_path, framework="numpy") as f:
    norm = f.get_tensor("model.norm.weight")
    print(f"dtype: {norm.dtype}, shape: {norm.shape}")
    print(f"norm[0:8] = {[f'{v:.6f}' for v in norm[:8].astype(float)]}")

print("\n=== Layer 0 q_proj.weight first 8 values (flattened) ===")
with safe_open(shard1, framework="numpy") as f:
    qw = f.get_tensor("model.layers.0.self_attn.q_proj.weight")
    print(f"dtype: {qw.dtype}, shape: {qw.shape}")
    flat = qw.flatten()
    print(f"q_proj.flat[0:8] = {[f'{v:.6f}' for v in flat[:8].astype(float)]}")

print("\n=== lm_head.weight first 8 values (flattened) ===")
shard_file = index["weight_map"]["lm_head.weight"]
shard_path = str(MODEL_DIR / shard_file)
with safe_open(shard_path, framework="numpy") as f:
    lm = f.get_tensor("lm_head.weight")
    print(f"dtype: {lm.dtype}, shape: {lm.shape}")
    flat = lm.flatten()
    print(f"lm_head.flat[0:8] = {[f'{v:.6f}' for v in flat[:8].astype(float)]}")
