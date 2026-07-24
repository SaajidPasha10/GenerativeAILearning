# Day 2 - Transformer Architecture (Complete Notes)

# Transformer Overview

A Transformer converts raw text into contextual representations by repeatedly refining token embeddings through attention and neural networks.

The overall pipeline is:

```text
Input Text
    │
    ▼
Tokenizer
    │
    ▼
Token IDs
    │
    ▼
Token Embeddings
    │
    ▼
Positional Encoding
    │
    ▼
Transformer Blocks (Repeated N Times)
    │
    ▼
Final Token Representations
    │
    ▼
Linear Layer
    │
    ▼
Softmax
    │
    ▼
Next Token Prediction
```

---

# Step 1 - Token Embedding

Purpose:

> Convert tokens into dense numerical vectors representing semantic meaning.

Example:

```text
Dog

↓

[0.72, 0.13, 0.91, ...]
```

Memory:

```text
Embedding = What is this word?
```

---

# Step 2 - Positional Encoding

Purpose:

Transformers process tokens in parallel.

Position information is added so the model understands sequence order.

Formula:

```text
Input Representation

=

Token Embedding

+

Positional Encoding
```

Example:

```text
Dog + Position 1

Bites + Position 2

Man + Position 3
```

Memory:

```text
Position = Where is this word?
```

---

# Transformer Block

Modern GPT repeats the following block many times.

```text
                         Current Representation (x)
                                  │
                                  ▼
                          Layer Normalization
                                  │
                                  ▼
                        Multi-Head Attention
                                  │
                                  ▼
                     Residual Addition (x + ...)
                                  │
                                  ▼
                          Layer Normalization
                                  │
                                  ▼
                      Feed Forward Network
                     Linear → GELU → Linear
                                  │
                                  ▼
                     Residual Addition (x + ...)
                                  │
                                  ▼
                         Updated Representation
```

---

# Multi-Head Attention

Purpose:

Allow every token to gather information from every other token.

Question answered:

```text
Who should I listen to?
```

Each attention head learns different relationships.

Example:

Head 1

```text
Pronouns
```

Head 2

```text
Grammar
```

Head 3

```text
Semantic Meaning
```

Head 4

```text
Long-distance Relationships
```

Outputs from all heads are concatenated.

---

# Inside One Attention Head

```text
Current Representation (x)
            │
            ▼
      Linear Layers
            │
     ┌──────┼──────┐
     ▼      ▼      ▼
     Q      K      V
     │      │      │
     └──────┬──────┘
            ▼
        Q × Kᵀ
            │
            ▼
        ÷ √dk
            │
            ▼
        Softmax
            │
            ▼
Attention Weights
            │
            ▼
Weights × Values
            │
            ▼
Context Representation
```

---

# Query, Key, Value

## Query

Question:

```text
What information am I looking for?
```

---

## Key

Question:

```text
What type of information do I represent?
```

---

## Value

Question:

```text
What information can I provide?
```

Memory Trick:

```text
Query asks

Key matches

Value delivers
```

---

# Attention Formula

```text
Attention(Q,K,V)

=

softmax(QKᵀ / √dk)V
```

Meaning:

```text
Compare Queries with Keys

↓

Generate Importance Scores

↓

Normalize

↓

Collect Values

↓

Contextual Representation
```

---

# Residual Connection

Purpose:

Do not replace knowledge.

Refine it.

Formula:

```text
Output

=

Input

+

Layer(Input)
```

Example:

```text
Current Knowledge

+

New Knowledge

↓

Better Knowledge
```

Memory:

```text
Residual = Keep + Improve
```

---

# Layer Normalization

Purpose:

Keep every token representation numerically stable.

Without LayerNorm:

```text
[300, 800, 120]

vs

[0.002, 0.005, 0.001]
```

With LayerNorm:

```text
Stable Scale

↓

Better Training
```

Memory:

```text
Stay Balanced
```

---

# Feed Forward Network (FFN)

Purpose:

Process information gathered by attention.

Attention:

```text
Talk to everyone.
```

FFN:

```text
Think privately.
```

Architecture:

```text
Linear

↓

GELU

↓

Linear
```

Memory:

```text
Attention = Communication

FFN = Reasoning
```

---

# Responsibilities

| Component            | Responsibility              |
| -------------------- | --------------------------- |
| Token Embedding      | What is this token?         |
| Positional Encoding  | Where is this token?        |
| Multi-Head Attention | Who should I learn from?    |
| Residual Connection  | Preserve previous knowledge |
| Layer Normalization  | Stabilize learning          |
| Feed Forward Network | Refine understanding        |

---

# Complete GPT Architecture

```text
                         INPUT TEXT
                              │
                              ▼
                         Tokenizer
                              │
                              ▼
                         Token IDs
                              │
                              ▼
                     Token Embeddings
                              │
                              ▼
                    Positional Encoding
                              │
                              ▼
                  Embedding + Position
                              │
                              ▼

══════════════════════════════════════════════

            Transformer Block × N

      ┌─────────────────────────────┐

          LayerNorm
               │
               ▼
      Multi-Head Attention
               │
               ▼
      Residual Addition
               │
               ▼
          LayerNorm
               │
               ▼
      Feed Forward Network
               │
               ▼
      Residual Addition

      └─────────────────────────────┘

══════════════════════════════════════════════

                              │
                              ▼
                  Final Token Representation
                              │
                              ▼
                        Linear Layer
                              │
                              ▼
                           Softmax
                              │
                              ▼
                     Next Token Prediction
```

---

# Mental Model

Instead of memorizing the architecture, remember the story:

```text
Who am I?
      │
      ▼
Embedding

Where am I?
      │
      ▼
Position

Who should I talk to?
      │
      ▼
Attention

Don't forget what I knew.
      │
      ▼
Residual

Stay balanced.
      │
      ▼
LayerNorm

Think privately.
      │
      ▼
Feed Forward

Repeat this process many times.

↓

Predict the next token.
```

---

# Interview Explanation (30 Seconds)

> A Transformer first converts tokens into embeddings and injects positional information. Each Transformer block applies Layer Normalization, Multi-Head Self-Attention to gather contextual information, Residual Connections to preserve earlier knowledge, another Layer Normalization, and a Feed Forward Network to refine each token independently. Multiple blocks are stacked to produce rich contextual representations, which are finally passed through a linear layer and softmax to predict the next token.

---

# One-Line Memory Cheat Sheet

```text
Embedding  → Who am I?

Position   → Where am I?

Attention  → Who should I listen to?

Residual   → Keep what I already know.

LayerNorm  → Stay balanced.

FFN        → Think privately.

Repeat N times.

Linear + Softmax → Predict next token.
```
