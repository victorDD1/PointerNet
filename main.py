import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam

from model import PointerNet
from data import DistanceDataset

if __name__ == "__main__":

    # Toy problem: sorting points by distance.
    SIZE = 512
    C = 3
    Li = 8
    Lo = 8
    B = 64
    E = 8
    H = 16
    N = 1
    bidir=False

    EPOCHS = 1000
    lr = 8e-4

    dataset = DistanceDataset(SIZE, C, Li, Lo)
    dataloader = DataLoader(dataset, B, shuffle=True)

    model = PointerNet(C, Lo, E, H, N, bidir=bidir)
    criterion = torch.nn.CrossEntropyLoss()
    optim = Adam(model.parameters(), lr)

    print("Number of parameters", sum(p.numel() for p in model.parameters() if p.requires_grad))

    iterator = tqdm(range(EPOCHS), unit='Epoch')
    for epoch in iterator:
        for i, t in dataloader:
            optim.zero_grad()
            logits = model(i)
            loss = criterion(logits, t)
            loss.backward()
            optim.step()
        
        acc = ((model.pointers == t).sum() / (t.numel())).item()
        d = {"loss":loss.item(), "acc":acc}
        iterator.set_postfix(d)
