"""
problem we are solving:
for the seq : i love ai
Tokenized : [1, 2, 3]
[i] -> [love]
[i love] -> [ai]
====================
Input       Target
[1]    →     [2]
[1,2]  →     [3]
===================
Dataset : Divides the token ids to batches of input and target tokens
Data loader : Shuffles the input and target tokens into batches
"""
from torch.utils.data import DataLoader,Dataset
import torch

class DatasetDataLoader(Dataset):

    def __init__(self,token_ids,sequence_len):
        """
        :param token_ids: [1,2,3,4,5,6]
        :param sequence_len: 3 : {input[1,2,3], target[2,3,4]...}
        """
        self.token_ids = token_ids
        self.sequence_len = sequence_len

    def __len__(self):
        # Length of dataset would be token_ids - seq_len(Batch)
        return len(self.token_ids) - self.sequence_len

    def __getitem__(self, index):
        # Ex : input = self.token_ids[0 : 0 + 3] -> [1,2,3]
        # target - self.token_ids[1 : 0 + 3 + 1] -> [2,3,4]
        input_tokens = self.token_ids[index : index + self.sequence_len]
        target_tokens = self.token_ids[index + 1 : index+self.sequence_len + 1]
        return torch.tensor(input_tokens), torch.tensor(target_tokens)

if __name__ == "__main__":
    token_ids = [1, 2, 3, 4, 5, 6]
    sequence_len = 3
    dataset = DatasetDataLoader(token_ids, sequence_len)

    for i in range(len(dataset)):
        x, y = dataset[i]
        print(f"Input: {x}, Target : {y}")

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for inputs, targets in dataloader:
        print(f"Batch : Input: {inputs}, Target : {targets}")

"""
Output : 
Input: tensor([1, 2, 3]), Target : tensor([2, 3, 4])
Input: tensor([2, 3, 4]), Target : tensor([3, 4, 5])
Input: tensor([3, 4, 5]), Target : tensor([4, 5, 6])
Batch : Input: tensor([[3, 4, 5],
        [2, 3, 4]]), Target : tensor([[4, 5, 6],
        [3, 4, 5]])
Batch : Input: tensor([[1, 2, 3]]), Target : tensor([[2, 3, 4]])
"""