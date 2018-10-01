from __future__ import print_function
from MyModels import AutoEncoder
from PIL import Image
from scipy.ndimage.interpolation import rotate
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import argparse
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pytorch_ssim
import random
import sys
import time
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
from tqdm   import tqdm


def reconstruction_loss(args, x, y):
    #Define Reconstruction Loss

    #L1 difference
    L1_loss = torch.nn.L1Loss(reduction='elementwise_mean')

    #SSIM index
    ssim_loss = pytorch_ssim.SSIM(window_size=args.window_size)  

    #Final loss is convex combination of the above
    loss = (1-args.alpha) * L1_loss(x,y) + \
            args.alpha * torch.clamp( (1-ssim_loss(x,y))/2,0,1)

    return loss


class TrainingDataset(Dataset):
    """Training set with only normal images """

    def __init__(self, filename, transform=None):
        """
        Args:
            filename: input h5 filename
        """
        h5_file = h5py.File(filename, 'r')
        self.dataset = h5_file['x']
        self.transform = transform

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        image = self.dataset[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image


def reconstructionTest(args, model, dataloader, epoch, path):
    """
    Produces an image with reconstruction results 

    Args:
        args:           arguments wrapper
        model:          pytorch model
        dataloader:     pytorch dataloader

    Output:
        none    
    """
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            targets = model(data)
            break

    #Concatenate ground truth
    output = torch.cat((targets, data), dim=0)

    #Save image to file
    filename = path + "/Reconstruction_Epoch_{:03d}.png".format(epoch)
    torchvision.utils.save_image(
        output.cpu(), filename, nrow=output.shape[0] // 2, padding=5)


def save_model(args, model, epoch):
    """
    saves a checkpoint so that model weight can later be used for inference
    Args:
        model:  pytorch model
    """
    path = './model_' + args.name
    #Create path if it doesn't exists
    if not os.path.exists(path):
        os.mkdir(path)
    torch.save(model.state_dict(),
               path + '/checkpoint_epoch_{}.pt'.format(epoch))


def main():
    parser = argparse.ArgumentParser(
        description='AutoEccoder for compressed representation')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        metavar='N',
        help='input batch size for training (default: 64)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.85,
        metavar='a',
        help='weight of SSIM in loss function (Default=0.85)')
    parser.add_argument(
        '--window-size',
        type=int,
        default=11,
        help='Window size for SSIM loss (Default=11 pixels)')
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='number of training epochs (Default=20)')
    parser.add_argument(
        '--lr', type=float, default=1e-4, help='learning rate (Default=1E-4)')
    parser.add_argument(
        '--eval_images',
        type=int,
        default=5,
        help='number of images for reconstruction test (Default=5)')
    parser.add_argument(
        '--save',
        type=int,
        default=10,
        metavar='N',
        help='save model every this number of epochs (Default=5)')
    parser.add_argument(
        '--recon-epochs',
        type=int,
        default=5,
        metavar='N',
        help='test reconstruction quality every N epochs (Default=5)')
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.3,
        metavar='N',
        help=
        'apply dropout to the feature vector to ensure spread of information')
    parser.add_argument(
        '--name',
        type=str,
        default='test',
        help=
        'name of run (default=test')
    parser.add_argument(
        '--print-progress',
        action='store_true',
        default='False',
        help=
        'do not print Mini-Batch-loss')

    args = parser.parse_args()

    #Print arguments to file
    for arg in vars(args):
        sys.stdout.write('{} = {} \n'.format(arg, getattr(args, arg)))
        sys.stdout.flush()
    sys.stdout.write('Random torch seed:{}\n'.format(torch.initial_seed()))
    sys.stdout.flush()

    #Dataset root directory
    train_rootdir = './pcamv1/camelyonpatch_level_2_split_train_normal_subsample.h5'

    #Define Training data transformations
    training_transformations = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Pad(96, padding_mode='reflect'),
        transforms.RandomRotation(30),
        transforms.CenterCrop(96),
        transforms.ToTensor()
    ])

    #Build DataLoaders
    train_loader = DataLoader(
        TrainingDataset(train_rootdir, transform=training_transformations),
        batch_size=args.batch_size,
        shuffle=True)

    eval_loader = DataLoader(
        TrainingDataset(train_rootdir, transform=transforms.ToTensor()),
        batch_size=args.eval_images,
        shuffle=True)

    #Load Model
    model = AutoEncoder(args.dropout)

    #Optimiser
    optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    logging_dir = './logs_' + args.name
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    #Create Summary class for TensorboardX
    writer = SummaryWriter(
        logging_dir, comment='AutoEncoder for compressed representation')

    sys.stdout.write('Start training\n')
    sys.stdout.flush()
    n_iter = 0

    #Train
    for epoch in range(1, args.epochs + 1):
        sys.stdout.write('Starting epoch {}/{} \n '.format(epoch, args.epochs))
        sys.stdout.flush()

        for batch_idx, data in enumerate(train_loader):
            model.train()
            #Get output (forward pass)
            target = model(data)

            #Set optimiser gradient to zero
            optimizer.zero_grad()

            #Get reconstruction loss:
            loss = reconstruction_loss(args, data, target)

            #Optimise
            loss.backward()
            optimizer.step()

            #Log into writer
            writer.add_scalar('Mini-Batch-loss', loss.item(), n_iter)

            #Print progress
            if args.print_progress:
           
                sys.stdout.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\r'
                .format(epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
                sys.stdout.flush()

            n_iter += 1


        #Reconstruction test
        if epoch % args.recon_epochs == 0:
            reconstructionTest(args, model, eval_loader, epoch, logging_dir)

        #Save model
        if epoch % args.save == 0:
            save_model(args, model, epoch)


if __name__ == '__main__':
    main()
