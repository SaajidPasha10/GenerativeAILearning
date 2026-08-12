
import numpy as np
from ..decoding.greedy_decoder import GreedyDecoder
from ..decoding.temperature_sampling import TemperatureSampler
from ..decoding.top_k_sampling import TopKSampler
from ..decoding.top_p_sampling import TopPSampling



class Generator:

    def __init__(self,model):
        self.model = model

    # Predict the next token from logits and append the logit to original token ids
    def generate(self,token_ids,max_new_tokens,strategy=None,temperature=None,k=None,p=None):

        generated_tokens = list(token_ids)

        for _ in range(max_new_tokens):
            # Convert token list to np array
            input_tokens = np.array(generated_tokens)
            # Forward pass
            logits = self.model.forward(input_tokens)
            # Get the last logit as it has more context to predict next token
            next_token_logits = logits[-1]
            # Get the logit with max val
            next_token = GreedyDecoder.decode(next_token_logits)
            match strategy:
                case "temperature":
                    next_token = TemperatureSampler.sample(next_token_logits,temperature)

                case "top_k":
                    next_token = TopKSampler.sample(next_token_logits, k, temperature)
                case "top_p":
                    next_token = TopPSampling.sample(next_token_logits, p, temperature)

                case _ :
                    next_token = next_token
            generated_tokens.append(next_token)

        return generated_tokens