"""

Prompt
  ↓
Tokenizer
  ↓
Token IDs
  ↓
Embedding
  ↓
Transformer Blocks
  ↓
Logits
  ↓
┌─────────────────────────────┐
│     Decoding Strategy       │
│                             │
│ Greedy                      │
│ Temperature                 │
│ Top-K                       │
│ Top-P                       │
└─────────────────────────────┘
  ↓
Next Token
  ↓
Append
  ↓
Repeat
  ↓
Generated Text

"""