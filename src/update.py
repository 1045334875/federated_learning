#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
import copy
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 1000 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def apply_pgd_weight(self, new_weights, old_weights, norm, clip):
        # print("\n\n\n apply pgd attack")
        if norm == 1:
            # norm1 clip
            assert old_weights is not None, "Old weights can't be none"
            assert new_weights is not None, "New weights can't be none"
            w_copy = copy.deepcopy(new_weights)
            for key in w_copy.keys():
                w_copy[key] -= old_weights[key] # delta_weight = new_weights - old_weights
                w_copy[key] = torch.clamp(w_copy[key], min=-clip, max=clip) # clip delta_weight
                w_copy[key] += old_weights[key] # get the clip result
            # print("delta\n\n\n")
            # print(w_copy)
            return w_copy
        else:
            l2_norm_tensor = clip #tf.constant(l2) # clip = 10
            w_copy = copy.deepcopy(new_weights)
            
            for key in w_copy.keys():
                w_copy[key] -= old_weights[key]
                layers_to_clip = w_copy[key].reshape(1,-1)
                norm = max(torch.norm(layers_to_clip), 0.00001)
                multiply = min((l2_norm_tensor / norm).numpy(), 1.0)
                w_copy[key] *= multiply
                w_copy[key] += old_weights[key]
            return w_copy

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        false_correct = 0
        false_total = 0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
            dim0 = labels.shape
            for i in range(dim0[0]):
                if torch.eq(labels[i],7): false_total += 1
                if labels[i]==7 and pred_labels[i]==1: false_correct +=1
            total += len(labels)


        accuracy = correct/total
        false_accuracy = 0
        if(false_total != 0): false_accuracy = false_correct / false_total
        return accuracy, false_accuracy, loss

    

def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)
    false_correct = 0
    false_total = 0

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        dim0=labels.shape
        
        for i in range(dim0[0]):
            if torch.eq(labels[i],7): false_total += 1
            if labels[i]==7 and pred_labels[i]==1: false_correct +=1
        total += len(labels)

    accuracy = correct / total
    false_accuracy = 0
    if(false_total != 0): false_accuracy = false_correct / false_total
    return accuracy, false_accuracy, loss

