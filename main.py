# --------------------------------------------------------------------------------------------------
# ------------------------------------- PokeFusion Entry Point -------------------------------------
# --------------------------------------------------------------------------------------------------
from Datasets.SharpenerDataset import SharpenerDataset

from Networks.Classifier import Classifier
from Networks.Contrastor import Contrastor
from Networks.Controller import Controller
from Networks.Decoder    import Decoder
from Networks.Encoder    import Encoder
from Networks.Network    import Network
from Networks.Painter    import Painter
from Networks.Sampler    import Sampler
from Networks.Segmentor  import Segmentor
from Networks.Sharpener  import Sharpener

from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

import torch
import os
import uuid
import random
import pathlib


# --------------------------------------------------------------------------------------------------
# ----------------------------------------- Save an Image ------------------------------------------
# --------------------------------------------------------------------------------------------------
def save(
    eid: int, bid: int, source: torch.Tensor, target: torch.Tensor, result: torch.Tensor) -> None:

    random = torch.randint(0, source.size(0), (1,)).item()
    simage = transforms.ToPILImage()(source[random].cpu())
    timage = transforms.ToPILImage()(target[random].cpu())
    rimage = transforms.ToPILImage()(result[random].cpu())

    merged = Image.new('RGB', (1536, 384))
    merged.paste(simage, (0,    0))
    merged.paste(timage, (512,  0))
    merged.paste(rimage, (1024, 0))

    merged.save(f'Output/{eid:05}-{bid:05}.png')


# --------------------------------------------------------------------------------------------------
# ----------------------------------------- Configuration ------------------------------------------
# --------------------------------------------------------------------------------------------------
dataset = DataLoader(SharpenerDataset(), batch_size=8, shuffle=True)
sharpener = Sharpener('Sharpener').load()
optimizer = torch.optim.Adam(sharpener.parameters(), lr=1e-5)


# --------------------------------------------------------------------------------------------------
# ------------------------------------------- Train Loop -------------------------------------------
# --------------------------------------------------------------------------------------------------
for eid in range(epochs := 1):

    for bid, (source, target) in enumerate(dataset):

        optimizer.zero_grad()

        result = sharpener(source)
        loss = sharpener.loss(result, target)

        loss.backward()
        optimizer.step()

        print(f"[{eid:05}:{bid:05}] Loss: {loss.item() / source.size(0)}")
        save(eid, bid, source, target, result)

sharpener.save()