# Residual Connections (Skip Connections) - Quick Revision Notes

## 1. Core Idea

Residual connections preserve the original representation by adding it to the output of a layer instead of replacing it.

Formula:

```text
Output = Input + Layer(Input)
```

The layer learns a **refinement**, not a complete replacement.

---

# 2. Why Do We Need Residual Connections?

Without residual connections:

```text
Layer 1
↓

Layer 2
↓

Layer 3
↓

Original information may be lost.
```

With residual connections:

```text
Original Representation

+

New Information

↓

Better Representation
```

---

# 3. Human Analogy

Learning DSA:

```text
Arrays

+

Linked Lists

+

Trees

+

Graphs
```

You don't forget Arrays when learning Trees.

Knowledge accumulates.

---

# 4. Island Analogy 🏝️

Initially:

```text
Indian

Software Engineer
```

After talking to everyone:

```text
AI

Python

Cloud
```

Final understanding:

```text
Indian

Software Engineer

AI

Python

Cloud
```

You enrich your knowledge rather than replacing it.

---

# 5. Benefits

### Preserve Information

Earlier useful representations are retained.

### Easier Training

Residual paths improve gradient flow and reduce the vanishing gradient problem.

### Enables Deep Models

Makes training Transformers with dozens or hundreds of layers practical.

---

# 6. Transformer Block

```text
Input

↓

Multi-Head Attention

↓

+ Residual

↓

Feed Forward

↓

+ Residual
```

---

# 7. Interview Explanation

> Residual connections allow each Transformer layer to learn incremental refinements while preserving earlier information. They also improve gradient flow during backpropagation, enabling very deep Transformer architectures to train effectively.

---

# 8. Memory Trick

```text
Residual = Keep + Improve
```

Or:

```text
Don't Replace.
Refine.
```

---

# 9. Common Mistake

❌ Each Transformer layer completely replaces the previous representation.

✅ Each layer refines the previous representation by adding new learned information.
