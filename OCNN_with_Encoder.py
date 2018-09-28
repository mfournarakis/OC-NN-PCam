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
from tqdm import tqdm


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


class ValidationDataset(Dataset):
    """ Validation set with both normal and abnormal images"""
    
    def __init__(self, images_filename, labels_filename,transform=None):
        """
        Args:
            image_filename: h5  image filename
            labels_filename: h5 labels filename
        """
        h5_images=h5py.File(images_filename,'r')
        h5_labels=h5py.File(labels_filename, 'r')
        self.dataset=h5_images['x']
        self.labels=h5_labels['y']
        self.transform=transform
    
    def __len__(self):
        return self.dataset.shape[0]
    
    def __getitem__(self, idx):
        image=self.dataset[idx]
        
        if self.transform is not None:
            image=self.transform(image)
        label=int(self.labels[idx][:])
        
        return (image, label)


def im2vec(x,model):
	#Get encoder representation of input image using encoder (model) 
	model.eval()
	return model.encoder(x)


def ocnn_objective(X, nu, w1, w2,r):
	#learning object - loss function
	w = w1
	V = w2

	term1 = 0.5 * (w.norm(2))**2
	term2 = 0.5 * (V.norm(2))**2
	term3 = 1 / nu * F.relu( (r - nnScore(X,w,V)).mean())
	term4 = -r 

	return term1 + term2 + term3 + term4 


def forward_propagation(x,w1,w2):
	"""
	Forward propagation using OC-NN
	"""
	hidden = torch.matmul(x, w1)

	yhat=torch.matmu(hidden,w2)

	return yhat


def nnScore(X, w, V):
	#Score function 
	return torch.matmul(F.sigmoid(torch.matmul(X,w)),V)



def createDalaLoaders(train_dir, validation_dir , test_dir):
	#Define Dataloader
    training_transformations = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Pad(96, padding_mode='reflect'),
        transforms.RandomRotation(30),
        transforms.CenterCrop(96),
        transforms.ToTensor()
    ])

    train_loader = DataLoader(
        TrainingDataset(train_dir, transform=training_transformations),
        batch_size=args.batch_size,
        shuffle=True)

    valid_loader=DataLoader(
    	ValidationDataset(validation_dir,transform=transfroms.ToTensor()),
    	batch_size=args.eval_batch_size,
    	shuffle=True)
    
    test_loader=DataLoader(
    	ValidationDataset(test_dir,transform=transfroms.ToTensor()),
    	batch_size=args.eval_batch_size,
    	shuffle=True)

    return train_loader, valid_loader, test_loader


def main():
	parser = argparse.ArgumentParser(
        description='OC-NN Feedward')
	parser = argparse.ArgumentParser(
        description='AutoEccoder for compressed representation')

    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        metavar='N',
        help='input batch size for training (default: 64)')

    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='number of training epochs (Default=20)')

    parser.add_argument(
        '--lr', type=float, default=1e-4, help='learning rate (Default=1e-4)')

    parser.add_argument(
        '--r', type=float, default=0.1, help='bias starting value (Default=0.1')

    parser.add_argument(
        '--nu', type=float, default=0.04, help='quantile (Default=0.04)')

    parser.add_argument(
        '--momentum', type=float, default=0.9, help='momentum for SGD optimiser (default=0.9)')

    parser.add_argument(
        '--save',
        type=int,
        default=10,
        metavar='N',
        help='save model every this number of epochs (Default=5)')

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
    #Print arguments
    for arg in vars(args):
        sys.stdout.write('{} = {} \n'.format(arg, getattr(args, arg)))
        sys.stdout.flush()
    sys.stdout.write('Random torch seed:{}\n'.format(torch.initial_seed()))
    sys.stdout.flush()

	#Load pre-trained encoder 
	encoder_weights_roodir=''

	autoencoder=AutoEncoder(0)

	#Load weights
	autoencoder.load_state_dict(torch.load(encoder_weights_roodir))

	#Freeze weights during training
	for param in autoencoder.parameters():
    	param.requires_grad = False

    #Dateset root:
    train_rootdir = './pcamv1/camelyonpatch_level_2_split_train_normal_subsample.h5'

	#Initialise weights for the 2 linear layers

	w1 = torch.empty(512, 256, dtype=torch.float32 ,requires_grad=True)
	w2 = torch.empty (256, 1, dtype=torch.float32,requires_grad=True)

	nn.init.xavier_normal_(w1, gain=1)
	nn.init.xavier_normal_(w2, gain=1)

	#Set up optimiser
	optimizer=optim.SGD([w1,w2],lr = rgs.lr, momentum = args.momentum)

	#Create output/logging file

	logging_dir = './logs_' + args.name
    if not os.path.exists(logging_dir):
    	os.makedirs(logging_dir)

    #Create TensorboardX summary

	writer = SummaryWriter(
        logging_dir, comment='One Class Classifier with NN')
		

	#Start training
    sys.stdout.write('Start training\n')
    sys.stdout.flush()
    n_iter = 0

    #Train
    for epoch in range(1, args.epochs + 1):
        sys.stdout.write('Starting epoch {}/{} \n '.format(epoch, args.epochs))
        sys.stdout.flush()

        for batch_idx,data in enumerate(train_loader):


	
    

if __name__ == '__main__':
    main()