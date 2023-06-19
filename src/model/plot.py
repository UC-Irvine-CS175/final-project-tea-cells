import matplotlib.pyplot as plt
import pyprojroot
root = pyprojroot.find_root(pyprojroot.has_dir(".git"))
import sys
sys.path.append(str(root))
from sklearn.metrics import accuracy_score
from src.dataset.bps_dataset import BPSMouseDataset
from src.dataset.bps_datamodule import BPSDataModule
from torchmetrics import Accuracy
from src.dataset.augmentation import(
    NormalizeBPS,
    ResizeBPS,
    VFlipBPS,
    HFlipBPS,
    RotateBPS,
    RandomCropBPS,
    ToTensor
)
import wandb
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

import boto3
from botocore import UNSIGNED
from botocore.config import Config

from torch.hub import load_state_dict_from_url
from dataclasses import dataclass
from datetime import datetime

import io
from io import BytesIO
from PIL import Image
from torchvision.models import resnet50
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision
from torchvision.transforms import Resize, ToTensor, Compose, Normalize
from torch import nn
import pandas as pd
from torch.optim import Adam

#collected loss during train
y = [0.5549733337546497, 0.5022127166121475, 0.4582791335237813, 0.4503001829685966, 0.38492737717409625, 0.3382541890625379, 0.31228199419478825, 0.3044505559283036, 0.28412207564337993, 0.27802123816539964, 0.27314876362437657, 0.26195637621479273, 0.2505703130132874, 0.2469872549153629, 0.24616336921577694, 0.24904042466801526, 0.23778092773952686, 0.22714510740206964, 0.2243401580373297, 0.2144197296730447]
x = [i for i in range(20)]
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("resnet50")
plt.plot(x,y)
plt.show()

