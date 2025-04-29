#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
from torchvision import datasets, transforms
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader, Dataset
import json
import os
import csv
import PIL

import random
import numpy as np

import socket
import datetime


class DatasetWithID(DatasetFolder):
    def __init__(self,
                 dataset,
                 client_ID
                 ):
        super(DatasetWithID, self).__init__(dataset.root,
            dataset.loader,
            dataset.extensions,
            dataset.transform,
            None)
        self.dataset = copy.deepcopy(dataset)
        self.client_ID = client_ID

    def __getitem__(self, index):
        img, target = self.dataset[index]
        # print(type(img))
        item = {
            "img": img,
            "target": target,
            "index": index,
            "client_ID": self.client_ID
        }
        return item

    def __len__(self):
        return len(self.dataset.samples)



def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    train_dataset_list = []
    test_dataset_list = []
    if args.dataset == 'cifar10':

        train_transform = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914672374725342, 0.4822617471218109, 0.4467701315879822),
                                 (0.24703224003314972, 0.24348513782024384, 0.26158785820007324)),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914672374725342, 0.4822617471218109, 0.4467701315879822),
                                 (0.24703224003314972, 0.24348513782024384, 0.26158785820007324)),
        ])

        if args.num_users > 1:
            path = args.load_data_from_dir + '/cifar10_'+str(args.num_users) + '_client'
        elif args.num_users == 1:
            path = args.load_data_from_dir + '/cifar10'
        test_dataset = args.load_data_from_dir + '/cifar10/test'

    elif args.dataset == 'imagenet':
        if args.pretrained:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            mean = [0.4675, 0.4519, 0.4094]
            std = [0.2874, 0.2765, 0.2859]
            #mean = [0.4802, 0.4481, 0.3975]
            #std = [0.2302, 0.2265, 0.2262]
        train_transform = transforms.Compose([
                # transforms.RandomRotation(20),
                # transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

        path = args.load_data_from_dir + '/subimage_' + str(args.num_users) + '_client'
        test_dataset = args.load_data_from_dir + '/sub-imagenet-20/test'

    for c in range(args.num_users):
        if args.num_users > 1:
            train_dataset = path + str(c) + '/train'
        elif args.num_users == 1:
            train_dataset = path + '/train'


        train_dataset_list.append(DatasetWithID(datasets.ImageFolder(root=train_dataset, transform=train_transform), client_ID=c))
        test_dataset_list.append(DatasetWithID(datasets.ImageFolder(root=test_dataset, transform=test_transform), client_ID=c))

    return train_dataset_list, test_dataset_list




def average_weights(w, target_client=[0], alpha=1.0):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]

        for j in target_client:
            w_avg[key] = w_avg[key] - w[j][key] + alpha * w[j][key]

        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Exp ID    : {args.exp_id}')
    print(f'    Seed    : {args.seed}')
    print(f'    Dataset   : {args.dataset}')
    print(f'    Num of classes   : {args.num_classes}')
    print(f'    Model     : {args.model}')
    print(f'    Frozen backbone     : {args.frozen}')


    if args.resume is not None:
        print(f'    Resume global model from {args.resume}')
        print(f'    Resume global model at {args.resume_epoch} epochs')
    elif args.pretrained:
        print(f'    Resume pretrained model from pytorch')
    else:
        print('    Training from scratch')

    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')
    print(f'    Alpha  : {args.alpha}\n')
    print(f'    Pretrained model  : {args.pretrained}\n')

    if args.mood == 'max_loss':
        print('    Max loss of whole local dataset and update global model every batch.')
    elif args.mood == 'avg_loss':
        print('    Avg loss of a local batch and update global model every batch.')

    print(f'    Change loss Round   : {args.changing_round}\n')
    print(f'    Attack Method   : {args.attack}\n')
    print(f'    Whether run leakage attack   : {args.leakage_attack}\n')
    print(f'    Max iterations   : {args.max_iterations}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Num of Users       : {args.num_users}\n')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')

    return


def load_json(json_file):
    with open(json_file) as data_file:
        data = json.load(data_file)
    return data




def system_startup(args=None, defs=None):
    """Print useful system information."""
    # Choose GPU device and print status information:
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    setup = dict(device=device, dtype=torch.float)  # non_blocking=NON_BLOCKING
    print('Currently evaluating -------------------------------:')
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print(f'CPUs: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}.')
    if args is not None:
        print(args)
    if defs is not None:
        print(repr(defs))
    if torch.cuda.is_available():
        print(f'GPU : {torch.cuda.get_device_name(device=device)}')
    return setup

def save_to_table(out_dir, name, dryrun, **kwargs):
    """Save keys to .csv files. Function adapted from Micah."""
    # Check for file
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fname = os.path.join(out_dir, f'table_{name}.csv')
    fieldnames = list(kwargs.keys())

    # Read or write header
    try:
        with open(fname, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            header = [line for line in reader][0]
    except Exception as e:
        print('Creating a new .csv table...')
        with open(fname, 'w') as f:
            writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
            writer.writeheader()
    if not dryrun:
        # Add row for this experiment
        with open(fname, 'a') as f:
            writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
            writer.writerow(kwargs)
        print('\nResults saved to ' + fname + '.')
    else:
        print(f'Would save results to {fname}.')
        print(f'Would save these keys: {fieldnames}.')

def set_random_seed(seed=233):
    """233 = 144 + 89 is my favorite number."""
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)
    np.random.seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 5)
    random.seed(seed + 6)

def set_deterministic():
    """Switch pytorch into a deterministic computation mode."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_img(img, dir):
    save_img = tensor_to_img(img)
    save_img.save(dir)
    return

def tensor_to_img(img):
    img = (img.detach().numpy() * 255)
    img = img.transpose(1, 2, 0)
    img = PIL.Image.fromarray(img.astype('uint8')).convert('RGB')
    return img

def run_anti_finetune(global_model, args):
    if args.gpu >= 0:
        device = 'cuda'
    else:
        device = 'cpu'
    global_model.train()
    # Finetune
    criterion_mean = nn.CrossEntropyLoss(reduction='mean').to(device)
    # Set optimizer for the local updates
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.anti_finetune_lr,
                                    momentum=0.5)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.anti_finetune_lr,
                                     weight_decay=1e-4)

    # target label
    target_label = random.randint(0, args.num_classes - 1)
    print('Target Label: {}'.format(target_label))

    # load dataset and user groups
    finetune_dataset = get_anti_finetune_dataset(args)
    finetune_dataloader = DataLoader(finetune_dataset, batch_size=args.local_bs, shuffle=True)

    for iter in range(args.anti_finetune_ep):
        loss=0
        for batch in finetune_dataloader:
            # print(batch)
            images = batch['img'].to(device)
            labels = batch['target'].to(device)
            #print('ori labels', labels)
            # fake label
            for id in range(len(labels)):
                label_int  = int(labels[id].item())
                if label_int == target_label:
                    while label_int == target_label:
                        label_int = random.randint(0, args.num_classes-1)
                    labels[id] = label_int
                else:
                    continue
            global_model.zero_grad()
            log_probs = global_model(images)
            batch_loss = criterion_mean(log_probs, labels)
            loss += batch_loss.item()
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        print('| Fake finetune epoch : {} \{} | \tLoss: {:.6f}\t'.format(
                iter, args.anti_finetune_ep, loss/len(labels)))
    # saving fake finetune model...
    save_path = './model/' + str(args.exp_id) + '/'
    isExists = os.path.exists(save_path)
    if not isExists:
        os.makedirs(save_path)
    torch.save(global_model.state_dict(), save_path +
               "anti_finetune_global_epoch{}_target_{}.pt".format(str(args.anti_finetune_ep), str(target_label)))

    return global_model.state_dict(), target_label

def get_anti_finetune_dataset(args):
    if args.dataset == 'cifar10':
        if args.pretrained:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            mean = [0.4675, 0.4519, 0.4094]
            std = [0.2874, 0.2765, 0.2859]
        train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        finetune_dataset_path = args.load_data_from_dir + '/cifar10/finetune'

    elif args.dataset == 'imagenet':
        if args.pretrained:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            mean = [0.4675, 0.4519, 0.4094]
            std = [0.2874, 0.2765, 0.2859]

        train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        finetune_dataset_path = args.load_data_from_dir + '/sub-imagenet-20/finetune'
    finetune_dataset = DatasetWithID(datasets.ImageFolder(root=finetune_dataset_path, transform=train_transform), client_ID=0)

    return finetune_dataset