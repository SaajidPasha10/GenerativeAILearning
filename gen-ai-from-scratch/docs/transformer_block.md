# Day 3 Summary – Building the Transformer Block

## Components Completed

- ✅ Multi-Head Attention
- ✅ Output Projection (Wₒ)
- ✅ Residual Connection
- ✅ Layer Normalization
- ✅ Feed Forward Network (FFN)

---

## Multi-Head Attention

- Split embedding into multiple heads.
- Each head learns different relationships (grammar, semantics, syntax, etc.).
- Concatenate all head outputs.
- Apply Output Projection (Wₒ) to mix information from all heads.

---

## Residual Connection

Formula:

Output = Input + Block(Input)

Purpose:

- Preserve original information.
- Allow gradients to flow easily.
- Make very deep Transformers trainable.

---

## Layer Normalization

Formula:

(X - Mean) / √(Variance + ε)

Purpose:

- Normalize each token independently.
- Mean ≈ 0
- Standard Deviation ≈ 1
- ε prevents division by zero.

---

## Feed Forward Network

Architecture:

Embedding
→ Linear (Expand)
→ GELU
→ Linear (Project Back)

Example:

768 → 3072 → 768

Purpose:

- Process each token independently.
- Learn richer, non-linear feature representations.

---

## Key Insight

A Transformer alternates between:

1. **Attention** → Tokens communicate with each other.
2. **Feed Forward Network** → Each token processes the gathered information independently.

This repeating pattern is the foundation of modern Transformer models.