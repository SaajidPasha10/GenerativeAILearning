# Self-Attention Mathematics - Quick Revision Notes

## 1. Core Idea

Self-attention allows each token to find which other tokens are important for understanding its meaning.

The process:

```
Token
 |
 |-- Ask: What information do I need?
 |
 |-- Find relevant tokens
 |
 |-- Collect useful information
 |
 ↓
Updated contextual representation
```

---

# 2. From Human Reasoning to Mathematics

Human:

> "I am the word 'it'. Who am I referring to?"

Transformer:

```
Query("it")

compares with

Keys of other tokens

↓

Attention scores

↓

Collect information from Values
```

---

# 3. Query, Key, Value (QKV)

Every token creates three vectors.

## Query (Q)

Question:

> What information am I looking for?

Example:

```
it → Who does this refer to?
```

---

## Key (K)

Description:

> What information do I represent?

Example:

```
Mary's Key:
- Person
- Female
- Name
```

The Key helps determine whether this token is relevant.

---

## Value (V)

Actual information provided.

Example:

```
Mary's Value:
- Mary is a person
- Mary received the book
```

The Value is what gets passed if attention selects that token.

---

# 4. Attention Flow

Complete process:

```
Input Tokens

      ↓

Token Embeddings

      ↓

Create Q, K, V

      ↓

Compare Query with Keys

      ↓

Calculate Attention Scores

      ↓

Softmax Normalization

      ↓

Weighted Values

      ↓

Contextual Token Representation
```

---

# 5. Attention Score

The model asks:

> "How relevant is this token to me?"

Example:

Sentence:

```
John gave Mary a book.
```

For the token "book":

Possible attention:

```
gave → High relevance
Mary → Medium relevance
John → Medium relevance
```

The score represents importance.

---

# 6. Attention Formula

The famous equation:

```
Attention(Q,K,V)
=
softmax(QKᵀ / √dk)V
```

---

## QKᵀ

Meaning:

> Compare my Query with all Keys.

Human analogy:

> "Who has useful information for me?"

---

## √dk

Purpose:

Keeps attention scores stable.

Without scaling:

```
Very large scores
        ↓
Softmax becomes extreme
        ↓
One token dominates
```

---

## Softmax

Converts scores into probabilities.

Example:

Before:

```
John  = 3.2
gave  = 5.7
Mary  = 4.1
```

After:

```
John  = 20%
gave  = 55%
Mary  = 25%
```

---

## × V

Meaning:

> Collect information from the important tokens.

The output becomes a new contextual representation.

---

# 7. Island Analogy 🏝️

People = Tokens

A person wants to understand their role.

## Query

The question asked:

> "Who can help me?"

---

## Key

Information badge:

> "What kind of information do I know?"

---

## Value

Actual information:

> "Here is what I know."

---

Self-attention:

```
Person A asks everyone

Person B asks everyone

Person C asks everyone
```

Every token attends to every other token.

---

# 8. Important Insights

## Attention is Dynamic

The same word can have different attention patterns depending on context.

Example:

```
The animal didn't cross the street because it was tired.
```

"it" attends more to:

```
animal
tired
```

---

```
The animal didn't cross the street because it was wide.
```

"it" attends more to:

```
street
wide
```

---

## Attention Happens Per Token

Not:

```
Sentence → Attention
```

Instead:

```
Token → Other Tokens
```

Each token creates its own attention pattern.

---

# 9. Common Mistakes

❌ Query means the user question.

✅ Query is created by every token internally.

---

❌ Key contains the actual information.

✅ Key represents what type of information the token has.

---

❌ Value decides attention.

✅ Query-Key interaction decides attention. Value provides the information.

---

# 10. Interview Explanation

> Self-attention works by creating Query, Key, and Value representations for each token. The Query-Key similarity determines attention weights, which are normalized using softmax. These weights are applied to Values to produce contextual token representations.

---

# 11. Memory Trick

```
Query  → What do I need?
Key    → What do you represent?
Value  → What can you provide?
```

Or:

```
Query asks
Key matches
Value delivers
```

---

# Current Understanding

After this section:

✅ Understand why QKV exists
✅ Understand attention scores
✅ Understand softmax role
✅ Understand attention formula intuition
✅ Understand how tokens build context

Next:

## Multi-Head Attention

Why one attention mechanism is not enough.
