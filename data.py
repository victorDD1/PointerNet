import torch
from torch.utils.data import Dataset

class DistanceDataset(Dataset):
    """
    Each sample consists of Li C dimensional points.
    Target are sorted points by distance to origin
    """
    def __init__(self,
                    size:int,
                    C:int,
                    Li:int,
                    Lo:int) -> None:
        super().__init__()
        self.size=size
        self.Li=Li
        self.Lo=Lo
        self.C=C

        self.input, self.target = self.generate_solutions()

    def generate_solutions(self):
        input = torch.randn((self.size, self.Li, self.C), dtype=torch.float)
        d = torch.norm(input, 2, dim=-1)
        target = torch.argsort(d)[:, :self.Lo].long()
        return input, target

    def __getitem__(self, index):
        return self.input[index, :, :], self.target[index, :]

    def __len__(self):
        return self.size