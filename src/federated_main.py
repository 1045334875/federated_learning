#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
# from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details, apply_dynamic_clipping

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    # logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    # if args.gpu_id:
    #     torch.cuda.set_device(args.gpu_id)
    device = 'cpu'#cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, false_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy, false_accuracy = [], [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0
    # !!! change at here
    has_attacker = 0 # 1 for 1/10 attacker ; 0 for non
    use_pgd = 0  # 2 for l2 norm; 1 for l1 norm 
    use_clip = 0  # 2 for l2 norm; 1 for l1 norm 
    clip= 4 # 0.0015(in n1 clip)

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        fal = has_attacker
       
        # client training
        for idx in idxs_users:
            if(fal==1):
                # !!!malicious client
                local_model = LocalUpdate(args=args, dataset=false_dataset,
                                            idxs=user_groups[idx]) # use malicious data to training
                w, loss = local_model.update_weights(
                    model=copy.deepcopy(global_model), global_round=epoch)
                # use clip defense or not
                if(use_pgd != 0): new_weight = local_model.apply_pgd_weight(w, global_weights, use_pgd, clip) 
                else: new_weight = w
                print("  loss:"+str(loss)+'\n')
                # aggregate the weight and loss
                local_weights.append(copy.deepcopy(new_weight))
                local_losses.append(copy.deepcopy(loss))
                fal=0 # control the first client is malicious in every round
            else:
                # denign clients
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                        idxs=user_groups[idx])
                w, loss = local_model.update_weights(
                    model=copy.deepcopy(global_model), global_round=epoch)
                # use the same clip defense
                if(use_pgd != 0): new_weight = local_model.apply_pgd_weight(w, global_weights, use_pgd, clip)
                else: new_weight = w
                print("  loss:"+str(loss)+'\n')
                local_weights.append(copy.deepcopy(new_weight))
                local_losses.append(copy.deepcopy(loss))
        
        temp_weight = local_weights
        if(use_clip): temp_weight = apply_dynamic_clipping(temp_weight, global_weights, clip)
        # update global weights from clients
        global_weights = average_weights(temp_weight)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss, list_fal_acc = [], [], []
        global_model.eval()
        # calculate the training result from each client
        for c in range(args.num_users): 
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                    idxs=user_groups[idx])
            acc,false_acc, loss = local_model.inference(model=global_model)
            # collect all the data to get the average
            list_fal_acc.append(false_acc)
            list_acc.append(acc)
            list_loss.append(loss)
        
        false_accuracy.append(sum(list_fal_acc)/len(list_fal_acc))  # probability to recognize 7 as 1
        train_accuracy.append(sum(list_acc)/len(list_acc))          # whole accuracy
        # print global training loss after every 'i' rounds
        print(f' \nAvg Training Stats after {epoch+1} global rounds:')
        print(f'Training Loss : {np.mean(np.array(train_loss))}')
        print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
        print('False Accuracy: {:.2f}% \n'.format(100*false_accuracy[-1]))

    # Test inference after completion of training
    test_acc, test_fal_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
    print("|---- Test false Accuracy: {:.2f}%".format(100*test_fal_acc))
    print("train acc each round")
    for i in train_accuracy:
        print('{:.2f}'.format(100*i), end=', ')
    print("\nattack acc each round")
    for i in false_accuracy:
        print('{:.2f}'.format(100*i), end=', ')
    

    # Saving the objects train_loss and train_accuracy:
    file_name = './save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)
    f.close()

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))



    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
