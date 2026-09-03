from torch.utils.data import DataLoader
from pytorch_gpt.tiny_gpt.training.dataset import dataset

dataloader = DataLoader(dataset=dataset,batch_size=2,shuffle=True)

for x,y in dataloader:
    print(f"Input {x} \n Target {y}")