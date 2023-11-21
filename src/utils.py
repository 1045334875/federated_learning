#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import copy
import numpy as np
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
from DealDataset import DealDataset


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = './data/mnist/'
        else:
            data_dir = './data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)
        print(train_dataset)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
        
        false_dataset = DealDataset('./data/MNIST/raw_f/', "train-images-idx3-ubyte.gz","false-labels-idx1-ubyte.gz",transform=transforms.ToTensor())
        # train_dataset = (
        #     datasets.mnist.read_image_file(os.path.join(data_dir, 'train-images.idx3-ubyte')),
        #     datasets.mnist.read_label_file(os.path.join(data_dir, 'train-labels.idx1-ubyte'))
        #         )
        # test_dataset = (
        #     datasets.mnist.read_image_file(os.path.join(data_dir, 't10k-images.idx3-ubyte')),
        #     datasets.mnist.read_label_file(os.path.join(data_dir, 't10k-labels.idx1-ubyte'))
        #         )
        # false_dataset = (
        #     datasets.mnist.read_image_file(os.path.join(data_dir, 'train-images.idx3-ubyte')),
        #     datasets.mnist.read_label_file(os.path.join(data_dir, 'false-labels.idx1-ubyte'))
        #         )
        
        
        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, false_dataset, user_groups

def apply_dynamic_clipping(weights, old_weights, clip_value):
    print("apply dynamic clipping")
    client_delta_weights = copy.deepcopy(weights)
    print(len(client_delta_weights))
    l2_norms_per_client = []
    for key in client_delta_weights[0].keys():
        norm=[]
        for i in range(len(client_delta_weights)):
            client_delta_weights[i][key] -= old_weights[key]
            layer = client_delta_weights[i][key].reshape(1,-1)
            norm.append(layer)
            # print(layer)
        # print("norm")
        # print(norm)
        # print("norm0")
        # print()
        l2_norms_per_client.append(torch.norm(norm[0]))
    # client_delta_weights = [[client_weights[i] - old_weights[i] for i in range(len(client_weights))] \
    #                         for client_weights in weights] # clip delta
    # l2_norms_per_client = [torch.norm(torch.cat([delta_weights[i].reshape(1,-1) \
    #                                         for i in range(len(delta_weights))], axis=0)) \
    #                     for delta_weights in client_delta_weights] # for norm calculation
    median = np.median(l2_norms_per_client) # clients l2 norm number median
    median_factor = clip_value # clip value

    bound = median * median_factor

    print(f"Effective bound: {bound}")
    print(norm)
    multipliers_per_client = [min((bound / norm).numpy(), 1.0) for norm in l2_norms_per_client]

    # delta_multiplied = [delta_weights[i] * multiply if i in clip_layers else delta_weights[i] for i in
            # range(len(delta_weights))]
    # print(len(multipliers_per_client))
    # print(len(client_delta_weights[0]))
    j=0
    delta_multiplied=[]
    
    for key in client_delta_weights[0].keys():
        # for i in range(len(client_delta_weights[0][key])):     
        #     print(len(client_delta_weights[0][key]))
        # print(client_delta_weights[0][key])
        # print(multipliers_per_client[j])
        client_delta_weights[0][key] *= multipliers_per_client[j]
        client_delta_weights[0][key] += old_weights[key]
        j+=1

    # Add back to global model to fit in other calculations
    return client_delta_weights

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
