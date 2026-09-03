# TinyGPT — Complete Training & Inference Pipeline

## 1. The Complete TinyGPT Pipeline

Suppose the input is:

```text
"ROMEO:"
```

With a character tokenizer, it becomes something like:

```text
"ROMEO:"
    ↓
[18, 15, 13, 5, 15, 9]
```

Shape:

```text
[B, S]
[1, 6]
```

---

## 2. Step 1 — Token Embeddings

The token IDs are passed through the embedding layer:

```text
[1, 6]
   ↓
Embedding
   ↓
[1, 6, 128]
```

Each character gets a **128-dimensional vector**.

So:

* `B = 1` → batch size
* `S = 6` → sequence length
* `D = 128` → embedding dimension

---

## 3. Step 2 — Positional Embeddings

The model also needs to know where each token occurs.

We add token embeddings and positional embeddings:

```text
Token embeddings       [1, 6, 128]
Position embeddings    [1, 6, 128]
                           +
                           ↓
                         [1, 6, 128]
```

The resulting representation contains:

> **What the token is + where the token is located**

---

## 4. Step 3 — Transformer Blocks

The representation passes through the Transformer blocks.

Your TinyGPT has 4 blocks:

```text
[1, 6, 128]
     ↓
Transformer Block 1
     ↓
[1, 6, 128]
     ↓
Transformer Block 2
     ↓
[1, 6, 128]
     ↓
Transformer Block 3
     ↓
[1, 6, 128]
     ↓
Transformer Block 4
     ↓
[1, 6, 128]
```

The outer shape remains the same.

Inside each block, however, attention transforms the representation using:

```text
Q = Query
K = Key
V = Value
```

and performs causal self-attention.

---

# 5. Step 4 — LM Head

After the Transformer blocks:

```text
[1, 6, 128]
     ↓
LayerNorm
     ↓
[1, 6, 128]
     ↓
Linear(128 → vocab_size)
     ↓
[1, 6, vocab_size]
```

Suppose the character vocabulary contains 66 characters:

```text
[1, 6, 66]
```

This means that at every position, the model produces a score for **every possible character**.

Conceptually:

```text
Position 1 → 66 possible next characters
Position 2 → 66 possible next characters
Position 3 → 66 possible next characters
Position 4 → 66 possible next characters
Position 5 → 66 possible next characters
Position 6 → 66 possible next characters
```

These scores are called **logits**.

---

# 6. What Is the Training Target?

This is one of the most important concepts in GPT training.

Suppose we have:

```text
Input:
R O M E O :
```

The target is shifted by one position:

```text
Target:
O M E O : \n
```

So:

```text
x = [R, O, M, E, O, :]
y = [O, M, E, O, :, \n]
```

The model produces predictions like:

```text
Position 1 → prediction for O
Position 2 → prediction for M
Position 3 → prediction for E
Position 4 → prediction for O
Position 5 → prediction for :
Position 6 → prediction for \n
```

Then:

```text
Model predictions
       ↓
CrossEntropyLoss
       ↓
Loss
       ↓
Backpropagation
       ↓
Weight updates
```

This is how GPT learns.

---

# 7. Why Can GPT Predict the Next Token at Every Position?

The answer is **causal masking**.

Consider:

```text
R O M E O :
```

### Position 1

It can see:

```text
R
```

and predicts:

```text
O
```

### Position 2

It can see:

```text
R O
```

and predicts:

```text
M
```

### Position 3

It can see:

```text
R O M
```

and predicts:

```text
E
```

And so on.

Therefore, one training sequence creates multiple learning examples:

```text
R              → O
R O            → M
R O M          → E
R O M E        → O
R O M E O      → :
R O M E O :    → \n
```

The causal mask prevents a position from looking into the future while still allowing the model to make predictions at every position.

---

# 8. Training vs Generation

Training and generation use the same model, but the process is slightly different.

## Training

During training, we already have the complete sequence:

```text
R O M E O :
```

The model can process all positions simultaneously.

It produces:

```text
Position 1 → prediction
Position 2 → prediction
Position 3 → prediction
Position 4 → prediction
Position 5 → prediction
Position 6 → prediction
```

The predictions are compared against the shifted targets.

---

## Generation

During generation, we only have the prompt:

```text
R O M E O :
```

We want to predict the next token:

```text
?
```

Therefore, we take only the logits from the final position:

```python
logits[:, -1, :]
```

This gives:

```text
[B, vocab_size]
```

For example:

```text
[1, 66]
```

Now we have the probability distribution for the **next token**.

---

# 9. Sampling During Generation

The generation process is:

```text
Logits
   ↓
Temperature
   ↓
Top-K
   ↓
Top-P
   ↓
Softmax / probabilities
   ↓
Sample
   ↓
Next token
```

For example, the model might predict:

```text
" "
```

as the next character.

We append it:

```text
R O M E O :
```

Then we run the process again.

The model predicts another token.

This continues autoregressively:

```text
Prompt
  ↓
Predict token
  ↓
Append token
  ↓
Predict token
  ↓
Append token
  ↓
Predict token
  ↓
...
```

---

# 10. Why Context Cropping Is Necessary

Your TinyGPT has:

```text
max_seq_len = 128
```

Therefore, it can only process a maximum of 128 tokens at a time.

But we may want to generate:

```text
300 tokens
```

or:

```text
1000 tokens
```

We don't throw away the generated sequence.

Instead, we use a **sliding context window**.

For example:

```text
Full generated sequence
──────────────────────────────────────────

Token 1 ... Token 172 | Token 173 ... Token 300
                      └──────────────────────┘
                            Last 128 tokens
                                  ↓
                                 GPT
                                  ↓
                              Next token
```

In code:

```python
context = token_ids[
    :, -self.pos_embeddings.num_embeddings:
]

logits = self(context)
```

The full generated sequence is preserved, but GPT only receives the most recent 128 tokens.

---

# 11. Complete TinyGPT Flow

The entire system can be viewed as:

```text
                    TRAINING
                       │
                       ▼
                Token IDs [B,S]
                       │
                       ▼
          Token + Position Embeddings
                       │
                       ▼
              Transformer Blocks
                       │
                       ▼
                   LayerNorm
                       │
                       ▼
               LM Head [B,S,V]
                       │
                       ▼
             Cross Entropy Loss
                       │
                       ▼
                Backpropagation
                       │
                       ▼
                 Update Weights
                       │
                       ▼
                TRAINED MODEL
                       │
                       │
                       ▼
────────────────────────────────────────
                       │
                    INFERENCE
                       │
                       ▼
                     Prompt
                       │
                       ▼
               Last 128 tokens
                       │
                       ▼
                     GPT
                       │
                       ▼
                 logits [B,V]
                       │
                       ▼
                  Temperature
                       │
                       ▼
                     Top-K
                       │
                       ▼
                     Top-P
                       │
                       ▼
                   Sampling
                       │
                       ▼
                  Next token
                       │
                       ▼
                 Append token
                       │
                       └──────────► Repeat
```

---

# 12. Key Shape Transformations

For TinyGPT:

```text
Token IDs
[B, S]

      ↓ Embedding

[B, S, D]

      ↓ + Positional Embedding

[B, S, D]

      ↓ Transformer Blocks

[B, S, D]

      ↓ LayerNorm

[B, S, D]

      ↓ LM Head

[B, S, V]

      ↓ During generation

[B, V]
```

Where:

```text
B = Batch Size
S = Sequence Length
D = Embedding Dimension
V = Vocabulary Size
```

For your current TinyGPT:

```text
B = depends on batch
S ≤ 128
D = 128
V = character vocabulary size
```

---

# 13. Five Things to Remember

### 1. Training

GPT learns by predicting the **next token at every position**.

### 2. Causal Mask

The causal mask prevents a position from seeing future tokens.

### 3. LM Head

The LM head converts:

```text
[B, S, D]
```

into:

```text
[B, S, V]
```

So every position gets a score for every vocabulary token.

### 4. Cross-Entropy Loss

The loss compares the model's prediction at each position against the **shifted target token**.

### 5. Generation

Generation uses:

```text
Last position
    ↓
Logits
    ↓
Sampling
    ↓
Next token
    ↓
Append
    ↓
Repeat
```

---

# 14. The Core Mental Model

The simplest way to remember GPT is:

> **GPT reads the previous tokens and learns to predict what comes next.**

During training:

```text
Previous tokens → Next token
```

During generation:

```text
Previous tokens → Predict next token
                       ↓
                  Append it
                       ↓
              Use it as context
                       ↓
              Predict next token
                       ↓
                     Repeat
```

That loop is the heart of an autoregressive GPT.

---

## Revision Summary

```text
Input tokens
    ↓
Token embeddings
    ↓
Position embeddings
    ↓
Causal self-attention
    ↓
Feed-forward networks
    ↓
Transformer blocks
    ↓
LayerNorm
    ↓
LM Head
    ↓
Logits
    ↓
Next-token probabilities
    ↓
Sampling
    ↓
Next token
```

**Training:** predict all next tokens in parallel.

**Inference:** predict one next token at a time.

**Causal masking:** prevents future-token leakage.

**Context cropping:** lets a model with a 128-token context generate longer sequences.

**Sampling:** decides which token to actually select from the model's predictions.
