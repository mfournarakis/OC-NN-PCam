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
from sklearn.datasets import make_blobs
import torchvision
import torchvision.models as models
import torch.onnx
from torch.autograd import Variable

from tqdm import tqdm


def ocnn_objective(X, nu, w1, w2, R):
    #learning object - loss function
    W = w1
    V = w2

    term1 = 0.5 * (W.norm(2))**2
    term2 = 0.5 * (V.norm(2))**2
    term3 = 1 / nu * F.relu_(R - nnScore(X, W, V)).mean()
    term4 = -R

    return term1 + term2 + term3 + term4


def forward_propagation(x, w1, w2):
    """
    Forward propagation using OC-NN
    """
    hidden = torch.matmul(x, w1)

    yhat = torch.matmu(hidden, w2)

    return yhat


def nnScore(X, W, V):
    #Score function

    return torch.matmul(torch.sigmoid(torch.matmul(X, W)), V)


def prepare_synthetic_data():
    #Generate Synthetic data
    n_samples = 190
    centers = 1
    num_features = 512
    #Make Blobs from
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=num_features,
        cluster_std=1.0,
        centers=centers,
        shuffle=False,
        random_state=42)
    ## Add 10 Normally distributed anomalous points
    mu, sigma = 0, 2
    a = np.random.normal(mu, sigma, (10, num_features))
    X = np.concatenate((X, a), axis=0)
    y_a = np.empty(10)
    y_a.fill(5)
    y = np.concatenate((y, y_a), axis=0)
    print(X.shape, y.shape)
    data_train = X[0:190]
    label_train = y[0:190]
    data_test = X[190:200]
    label_test = y[190:200]
    return [data_train, label_train, data_test, label_test]


def main():
    parser = argparse.ArgumentParser(
        description='OC-NN for synthetic data')

    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='number of training epochs (Default=100)')

    parser.add_argument(
        '--lr',
        type=float,
        default=0.0001,
        help='learning rate (Default=0.0001)')

    parser.add_argument(
        '--r',
        type=float,
        default=0.1,
        help='bias starting value (Default=0.1)')

    parser.add_argument(
        '--hidden',
        type=int,
        default=32,
        help='number of hidden neurons (hidden=32)')

    parser.add_argument(
        '--nu', type=float, default=0.04, help='quantile (Default=0.04)')

    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='momentum for SGD optimiser (default=0.9)')

    args = parser.parse_args()

    RANDOM_SEED = 42
    torch.manual_seed(RANDOM_SEED)
    #Print arguments
    for arg in vars(args):
        sys.stdout.write('{} = {} \n'.format(arg, getattr(args, arg)))
        sys.stdout.flush()
    sys.stdout.write('Random torch seed:{}\n'.format(torch.initial_seed()))
    sys.stdout.flush()

    #Set up trainable weights 
    w1 = torch.empty(512, args.hidden, dtype=torch.float32, requires_grad=True)
    w2 = torch.empty(args.hidden, 1, dtype=torch.float32, requires_grad=True)

    #Initialise weights based on initialisation provided by the
    #original paper
    torch.nn.init.normal_(w1, mean=0, std=0.00001)
    torch.nn.init.normal_(w2, mean=0, std=0.00001)

    #Set up optimiser
    optimizer = optim.SGD([w1, w2], lr=args.lr)

    data_train, label_train, data_test, label_test = prepare_synthetic_data()
    data_train = torch.from_numpy(data_train).float()
    data_test = torch.from_numpy(data_test).float()

    #Start training
    sys.stdout.write('Start training\n')
    sys.stdout.flush()
    n_iter = 0
    r_scalar = args.r

    # #Train
    # for epoch in range(1, args.epochs + 1):
    #     sys.stdout.write('Starting epoch {}/{} \n '.format(epoch, args.epochs))
    #     sys.stdout.flush()

    #     optimizer.zero_grad()

    #     #Get learning objective
    #     loss = ocnn_objective(data_train, args.nu, w1, w2, r_scalar)

    #     loss.backward()
    #     #Step1 of optimisation: Optimise w,V parameters

    #     optimizer.step()

    #     # Step2 of optimisation: Optimise bias r

    #     train_score = nnScore(data_train, w1, w2)
    #     r_scalar = float(
    #         np.percentile(train_score.detach().numpy(), q=100 * args.nu))

    #     print('epoch:{}, Loss:{:.6f}, r={:.4f}'.format(epoch, loss.item(),
    #                                                    r_scalar))
    #     n_iter += 1

    # #Estimate scores for training and test data

    # #Get best estimate for bias
    rstar = r_scalar

    #Calculate the score function for train (positive) and test(negative)
    with torch.no_grad():
        arrayTrain = nnScore(data_train, w1, w2)
        arrayTest = nnScore(data_test, w1, w2)

    #The decision score in the difference from the bias
    pos_decisionScore = arrayTrain.numpy() - rstar
    neg_decisionScore = arrayTest.numpy() - rstar

    np.savetxt(
        'positive_scores.csv', pos_decisionScore.astype(float), delimiter=',')
    np.savetxt(
        'negative_scores.csv', neg_decisionScore.astype(float), delimiter=',')


if __name__ == '__main__':
    main()