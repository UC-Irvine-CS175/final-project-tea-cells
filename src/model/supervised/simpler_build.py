import os
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


def main():
    bucket_name = "nasa-bps-training-data"
    s3_path = "Microscopy/train"
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    s3_meta_fname = "meta.csv"


    data_dir = root / 'data'
    # testing get file functions from s3
    local_train_dir = data_dir / 'processed'

    # testing PyTorch Lightning DataModule class ####
    train_csv_file = 'meta_dose_hi_hr_4_post_exposure_train.csv'
    train_dir = data_dir / 'processed'
    validation_csv_file = 'meta_dose_hi_hr_4_post_exposure_test.csv'
    validation_dir = data_dir / 'processed'

    bps_dm = BPSDataModule(train_csv_file=train_csv_file,
                            train_dir=train_dir,
                            val_csv_file=validation_csv_file,
                            val_dir=validation_dir,
                            resize_dims=(64, 64),
                            meta_csv_file = s3_meta_fname,
                            meta_root_dir=s3_path,
                            s3_client= s3_client,
                            bucket_name=bucket_name,
                            s3_path=s3_path,
                            )
    ##### UNCOMMENT THE LINE BELOW TO DOWNLOAD DATA FROM S3!!! #####
    # bps_dm.prepare_data()
    ##### WHEN YOU ARE DONE REMEMBER TO COMMENT THE LINE ABOVE TO AVOID
    ##### DOWNLOADING THE DATA AGAIN!!! #####
    bps_dm.setup(stage='train')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet50(pretrained=True)
    model = model.to(device)

    # Modify the last layer for multi-label classification
    num_features = model.fc.in_features
    num_classes = 2  # assuming the first column in the df is the image file name
    model.fc = nn.Linear(num_features, num_classes).to(device)

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  # suitable for multi-label classification
    optimizer = Adam(model.parameters())

    # Move model to GPU if available
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    torch.set_printoptions(precision=2)
    # Train the model
    loss_record = []

    for epoch in range(20):  # loop over the dataset multiple times
        total_loss = 0
        batch_count = 0
        for i, data in enumerate(bps_dm.train_dataloader()):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            if i % 5 == 4:    # print every 5 mini-batches
                print('Epoch: {}, Batch: {}, Loss: {:.3f}'.format(epoch + 1, i + 1, loss.item()))
                print("Probabilities", probabilities[0])
                print('Predictions:', predictions[0])  # print predictions for the first item in the batch
                print('True labels:', labels[0])  # print true labels for the first item in the batch
                print()
            # print statistics

            if i % 10 == 9:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))
            batch_count += 1
            total_loss += loss.item()
        if batch_count == 0:
            loss_record.append(0)
        else:
            loss_record.append(total_loss/batch_count)
    torch.save(model.state_dict(), 'model_weights1.pth')
    print(loss_record)
    with open("loss_record.txt", "w") as f:
        f.write(", ".join(loss_record))


    print("Done!")
if __name__ == "__main__":
    main()