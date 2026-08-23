import torch

def torch_basics():
    x = torch.Tensor([[1,2,3],[4,5,6]])
    print("Tensor Basics")
    print(f"Shape: {x.shape}")
    print(f"Num of Elements {x.numel()}")
    print(f"Dim {x.ndim}")
    print(f"Data type {x.dtype}")

def tensor_operations():
    x = torch.Tensor([[1,1],[2,2]])
    print("\n Tensor operations ")
    print(f" x+2 {x+2}")
    print(f" x*2 {x * 2}")
    print(f" x@x {x @ x}")
    print(f" x Transpose {x.T}")

torch_basics()
tensor_operations()
