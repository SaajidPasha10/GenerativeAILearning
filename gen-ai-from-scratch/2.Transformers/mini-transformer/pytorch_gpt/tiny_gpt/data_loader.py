from torch.utils.data import DataLoader
from dataset import GPTDataset,dataset

dataloader = DataLoader(dataset=dataset,batch_size=2,shuffle=True)

for x,y in dataloader:
    print(f"Input {x} \n Target {y}")