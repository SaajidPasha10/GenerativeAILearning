"""
              Input
             (3,4)

        ┌────┼────┐
        ▼    ▼    ▼

      WQ    WK    WV
    (4,4) (4,4) (4,4)

        ▼    ▼    ▼

      Q     K     V

    (3,4)(3,4)(3,4)
We will create 3 Vectors (Q,K,V) for each of the token
Q -> What am i looking for? K -> Here is what i know? V -> actual info
Each of the Weight vectors has shape (4,4)
Q = X * WQ
K = X * WK
V = X * WV
"""
import numpy as np
class ScaledAttention:
    def __init__(self,embedding_dim):
        # Initializing random weights of (n*n)
        # Embedding dim is size of the vector of a token here 4
        # Embedding of first token [-0.23415337 -0.23413696  1.57921282  0.76743473]
        # wq,wk,wv will be random vectors of size 4*4
        self.wq = np.random.randn(embedding_dim,embedding_dim)
        self.wk = np.random.randn(embedding_dim,embedding_dim)
        self.wv = np.random.randn(embedding_dim,embedding_dim)

    def project(self,embeddings):
        # Here embeddings of shape (3,4) will be dot product
        # with (4,4) weight vectors (3,4) * (4,4) --> (3,4)
        # m*n . n*k --> m * k
        q = np.dot(embeddings,self.wq)
        k = np.dot(embeddings,self.wk)
        v = np.dot(embeddings,self.wv)
        return q,k,v
    """
    Q [Token1 Token2 Token3]
        ×
    Kᵀ [Token1 Token2 Token3]
        ↓
Attention Matrix
          KT1    KT2    KT3

QT1      1.2   0.3   0.8

QT2      0.5   2.1   0.6

QT3      0.4   0.7   1.8
    """
    @staticmethod
    def attention_scores(q,k):
        # Attention score is basically how each word is
        # relevant to the current word. Q * K(transpose)
        # Embeddings shape of tokens (3,4)
        # ( WQ,WK,WV shapes ex : (4,4)
        # ( Q,K,V shapes ex : (3,4))
        # Q . K --> (3,4) . (3,4) Here column of Q and dont match
        # Hence transpose to (3,4) (4,3) --> (3,3)
        return np.dot(q,k.T)

    @staticmethod
    def scaled_attention_scores(attention_scores,embedding_dim):
            # When the embedding dim becomes large meaning there are so many numbers
            # in the vector [1,2,...4000] then the dot product becomes large
            # If softmax is applied then the probabilities become too biased
            # 99.99999% which makes model overconfident hence scale the scores
            # by dividing with sqrt(dim of k)
            scaled_attentions = attention_scores/np.sqrt(embedding_dim)
            return scaled_attentions

    # Implementing Softmax from scratch to understand
    # the raw attention scores conversion to probabilities
    # # softmax(x_i) = exp(x_i) / sum(exp(x_j))

    # Never use direct values to compute exponents
    # It will cause overflow of the result
    # Instead we can use for x in a vector --> x - max(vector)
    # Gives same result
    @staticmethod
    def softmax(attention_scores_vector):
        # Suppose a matrix [[1,2,3],[4,5,6],[7,8,9]]
        # np.max for row1 is 3 Axis = -1 means find max across last dimension which Row
        exp_scores = np.exp(attention_scores_vector - np.max(attention_scores_vector, axis=-1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    @staticmethod
    def compute_context(attention_weights,v):
        return attention_weights@v

    def forward(self,embeddings):
        # Here we took Embeddings of tokens[I, LOVE, AI]
        # Embedding dimensions created randomly are of (3,4) shape

        # Step 1 : Get Q,K,V values
        q,k,v = self.project(embeddings)
        #Step 2 : Get Attention scores for each Q wrt to each K
        # Q.K(Transpose)
        attention_scores = self.attention_scores(q,k)
        #Step 3: Scale the attention. Q.K(Transpose) / sqrt(dk)
        # dk is the dimension of the K vector
        scaled_attention_scores = self.scaled_attention_scores(attention_scores,embedding_dim=embeddings.shape[1])
        # Step 3.1 : Apply casual mask to hide next words of the current word
        masked_scores = self.casual_mask(scaled_attention_scores)
        # Step 4 : Apply Softmax on the scaled score
        # Softmax ( Q.KT / sqrt(dk))
        attention_weights = self.softmax(masked_scores)
        # Step 5 : Get Contextual Embeddings
        # Attention = Softmax ( Q.KT / sqrt(dk)). V
        context_aware_embeddings = self.compute_context(attention_weights,v)
        return context_aware_embeddings

    """
    Applied before softmax so that softmax predicts 0 for masked words
    meaning it cannot give probability values to words it has not seen
    For models like GPT they dont see the future words
    The cat sat on mat. 
    | Input          | Target |
    | -------------- | ------ |
    | The            | Cat    |
    | The Cat        | Sat    |
    | The Cat Sat    | On     |
    | The Cat Sat On | Mat    |

                The  Cat  Sat  On  Mat

    The          ✓    ✗    ✗    ✗    ✗
    
    Cat          ✓    ✓    ✗    ✗    ✗
    
    Sat          ✓    ✓    ✓    ✗    ✗
    
    On           ✓    ✓    ✓    ✓    ✗
    
    Mat          ✓    ✓    ✓    ✓    ✓
The cannot see future words.
Cat cannot see Sat, On, Mat.
Sat cannot see On or Mat.
On cannot see Mat.
Mat is the last word, so it can see everything before it.

This triangular structure is called the causal mask.
Predictions should be based on past words only
    """
    @staticmethod
    def casual_mask(raw_scores):
        """
           attention_scores shape:
           (sequence_length, sequence_length)
           """
        """
        [[1 1 1 1],[1 1 1 1],[1 1 1 1],[1 1 1 1]]
        np.triu(masked,k=1) -> [[0 1 1 1],[0 0 1 1],[0 0 0 1],[0 0 0 0]]
        -infinity where values are 1
        [[0 -∞ -∞ -∞ ],[0 0 -∞ -∞ ],[0 0 0 -∞],[0 0 0 0]]
        Add attention scores to it
        [[0 -∞ -∞ -∞ ],[0 0 -∞ -∞ ],[0 0 0 -∞],[0 0 0 0]] + [[1 2 3 4 ],....]
        [[1 -∞ -∞ -∞]...]] -∞ + number = -∞
        """
        seq_len  = raw_scores.shape[0]
        masked = np.ones((seq_len,seq_len))
        masked = np.triu(masked,k=1)
        masked = np.where(masked == 1, -np.inf,0)
        return raw_scores + masked

