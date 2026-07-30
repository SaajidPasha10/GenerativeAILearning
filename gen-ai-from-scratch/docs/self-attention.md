# Self-Attention - Quick Revision Notes

## 1. Core Idea

Self-attention allows each token in a sentence to look at other tokens and decide which ones are important for understanding its own meaning.

Instead of treating every word equally, the Transformer learns relationships between words based on context.

Example:

```
The animal didn't cross the street because it was tired.
```

The token **"it"** pays more attention to **"animal"** because the context suggests the animal was tired.

---

# 2. Why Do We Need Attention?

## Problem Before Transformers

Older models like RNNs processed text sequentially:

```
Word 1 → Word 2 → Word 3 → Word 4
```

Problems:

* Slow processing
* Difficult to capture long-range relationships
* Earlier information can be lost

Example:

```
The book that I bought yesterday from the store was expensive.
```

Connecting "book" and "expensive" becomes harder as distance increases.

---

## Transformer Approach

Transformers allow every token to interact with every other token.

```
        Token A
       ↙   ↓   ↘
Token B ← Token C → Token D
       ↖   ↓   ↗
        Token E
```

Each token decides:

> "Which other tokens help me understand my meaning?"

---

# 3. Self-Attention Intuition

Human example:

Sentence:

```
John gave Mary a book because she loves reading.
```

When understanding **"she"**, our brain looks at:

```
Mary        → High importance
John        → Low importance
Book        → Low importance
Reading     → Supporting context
```

The brain assigns different importance levels.

Self-attention does the same mathematically.

---

# 4. Query, Key, Value (QKV)

Every token creates three vectors:

```
Token
 |
 +---- Query
 |
 +---- Key
 |
 +---- Value
```

---

## Query (Q)

Question:

> What information am I looking for?

Example:

The token "she" asks:

```
Who does this pronoun refer to?
```

---

## Key (K)

Description:

> What information do I represent?

Example:

Mary's key may represent:

```
Person
Female
Name
Entity
```

---

## Value (V)

Actual information provided.

Example:

Mary's value:

```
Mary is a person.
Mary is female.
Mary received the book.
```

---

# 5. Island Analogy 🏝️

Imagine people dropped on an island.

Initially:

```
Person A
Person B
Person C
```

Nobody knows each other.

Everyone talks to understand others.

---

## Query

A person asks:

> "Who has information useful to me?"

---

## Key

Each person says:

> "Here is what type of information I know."

Example:

```
I know:
- Country
- Profession
- Skills
```

---

## Value

The actual information shared:

```
You are from India.
You are a software engineer.
You know Python.
```

---

## Self-Attention

Everyone asks everyone else.

```
Person A → Everyone
Person B → Everyone
Person C → Everyone
```

This is why it is called:

**Self-Attention**

---

# 6. Important Insight

Attention happens at the token level.

It is not:

```
Sentence → Attention
```

It is:

```
Token → Other Tokens
```

Every token creates its own attention pattern.

---

# 7. Attention Flow

Simplified process:

```
Input Tokens

      ↓

Token Embeddings

      ↓

Create Q, K, V

      ↓

Calculate relevance between Q and K

      ↓

Generate attention weights

      ↓

Combine Values

      ↓

Contextual Token Representation
```

---

# 8. Common Mistakes

## Mistake 1

❌ Attention means the model only focuses on one word.

✅ Attention distributes importance scores across multiple relevant words.

---

## Mistake 2

❌ Query, Key, Value are manually created.

✅ They are learned transformations during training.

---

## Mistake 3

❌ Embedding of a word is always fixed.

✅ Transformers create contextual representations.

Example:

```
Bank account

vs

River bank
```

The word "bank" gets different meanings based on context.

---

# 9. Interview Explanation

> Self-attention allows each token in a sequence to dynamically gather information from other tokens. Each token generates Query, Key, and Value vectors. Query-Key similarity determines how much attention one token should give to another, and the weighted Values create a contextual representation.

---

# 10. Memory Trick

```
Query  → What do I need?
Key    → What do I represent?
Value  → What information do I provide?
```

Or:

```
Query asks
Key matches
Value delivers
```

---

# 11. Current Understanding

You should now understand:

✅ Why attention is needed
✅ Why Transformers replaced RNNs
✅ How tokens use context
✅ Why QKV exists
✅ How attention resembles human reasoning

Next:

## Self-Attention Mathematics

We will derive:

```
Attention(Q,K,V)
=
softmax(QKᵀ / √dk)V
```

Understanding why each component exists instead of memorizing it.
