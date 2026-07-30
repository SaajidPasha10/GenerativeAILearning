"""
Residual connections provide an easy path for both information and gradients to flow unchanged when that's the best thing to do.
Imagine you're editing a document.
Without residual:
Rewrite the entire document every time.
With residual:
Keep the original document and only write the changes.
That's much easier.
Similarly after X attention layers the original info can be lost easily
so we keep original info and add the learned info
"""
class Residual:
    @staticmethod
    def forward(input_embeddings, context_aware_embeddings):
        return input_embeddings + context_aware_embeddings