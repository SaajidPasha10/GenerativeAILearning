import torch
from torch.utils.data import Dataset

class GPTDataset(Dataset):
    def __init__(self,token_ids,block_size):
        super().__init__()
        self.token_ids = token_ids
        self.block_size = block_size

    def __len__(self):
        return len(self.token_ids) - self.block_size

    def __getitem__(self, index):
        x = self.token_ids[index : index+self.block_size]
        y = self.token_ids[index + 1 : index+self.block_size + 1]
        x = torch.tensor(x,dtype=torch.long)
        y = torch.tensor(y,dtype=torch.long)

        return x,y

token_ids = [
    1, 2, 3, 4,
    1, 2, 5, 3,
    4, 1
]
dataset=GPTDataset(token_ids=torch.tensor(token_ids,dtype=torch.long),block_size=4)
print(f"Token Ids : {token_ids}")
for i in range(3):
    x,y = dataset[i]
    print(f"Input {x}, Target :{y}")