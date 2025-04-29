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
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from network import *
from utils import get_dataset, average_weights, exp_details, run_anti_finetune
import random
from inv_script import *


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    # args = my_create_argparser(args)
    print(args)
    # exit()

    # attack setting
    if args.changing_round < args.epochs:
        attack_epoch = [i for i in range(args.resume_epoch + args.changing_round, args.resume_epoch + args.epochs)]
        if args.attack_num is not None:
            attack_epoch = random.sample(attack_epoch, args.attack_num)
            # attack_epoch = [1]
            print('attack_epoch:', attack_epoch)
        target_client = [0]
    else:
        assert args.attack == 'none'
        attack_epoch = []
        target_client = []


    # set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.gpu >= 0:
        device = 'cuda'
    else:
        device = 'cpu'
    # torch.cuda.set_device(args.gpu_id)
    # device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset_list, test_dataset_list = get_dataset(args)

    # build model
    global_model = get_model(args)

    # Set the model to train and send it to device.
    global_model.to(device)

    global_model.train()
    print(global_model)

    if args.resume != None:
        global_model.load_state_dict(torch.load(args.resume), strict=False)
        print('resume global model from ' + args.resume)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0
    attack_ids = set()
    attack_repeat = dict()
    args.batch_size = args.local_bs

    # local_updata list
    local_model_list = []
    for idx in range(args.num_users):
        local_model_list.append(LocalUpdate(args=args, train_dataset=train_dataset_list[idx],
                                            test_dataset=test_dataset_list[idx],
                                            changing_round=args.changing_round, logger=logger, user_id=idx))

    # local_update object for test
    local_model_test = LocalUpdate(args=args, train_dataset=train_dataset_list[0],
                                   test_dataset=test_dataset_list[0],
                                   changing_round=args.changing_round, logger=logger, user_id=0)

    for epoch in tqdm(range(args.resume_epoch, args.resume_epoch + args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        client_0_attack_item = None
        target_label = None

        for idx in idxs_users:
            # fake finetune
            if (args.attack == 'anti_finetune' or args.attack == 'anti_finetune_duffision') and (
                    idx in target_client) and (epoch in attack_epoch):
                anti_finetune_global_model = copy.deepcopy(global_model)
                anti_finetune_global_weight, target_label = run_anti_finetune(anti_finetune_global_model, args)
                anti_finetune_global_model.load_state_dict(anti_finetune_global_weight)

                w, loss, attack_item, grad = local_model_list[idx].update_weights(
                    model=anti_finetune_global_model, global_round=epoch,
                    mood=args.mood, target_client=target_client)
            else:
                w, loss, attack_item, grad = local_model_list[idx].update_weights(
                    model=copy.deepcopy(global_model), global_round=epoch,
                    mood=args.mood, target_client=target_client)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            if (idx in target_client) and (epoch in attack_epoch) and args.leakage_attack:
                attack_ids, attack_repeat = gradient_inv(args, attack_item, idx, attack_ids, attack_repeat,
                                                         global_model, epoch, w, device, grad, train_dataset_list,
                                                         target_label=target_label)

            # saving local model...
            if epoch % args.save_interval == 0:
                save_path = './model/' + str(args.exp_id) + '/'
                isExists = os.path.exists(save_path)
                if not isExists:
                    os.makedirs(save_path)
                np.save(save_path + 'attack_repeat.npy', attack_repeat)

        # update global weights
        if epoch in attack_epoch:
            global_weights = average_weights(local_weights, target_client=target_client, alpha=args.alpha)
        else:
            global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        global_model.eval()

        # test
        if epoch % args.test_interval == 0 or epoch == (args.resume_epoch + args.epochs - 1):
            acc, loss = local_model_test.inference(model=global_model)
            print('epoch {}, glocal model acc {:.2f}%'.format(epoch, 100 * acc))

        if epoch % args.save_interval == 0 or epoch == int((args.resume_epoch + args.epochs) / 2):
            save_path = './model/' + str(args.exp_id) + '/'
            isExists = os.path.exists(save_path)
            if not isExists:
                os.makedirs(save_path)
            torch.save(global_model.state_dict(), save_path +
                       "global_epoch{}.pt".format(str(epoch)))

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset_list[0])

    torch.save(global_model.state_dict(), save_path +
               "global_final_epoch{}.pt".format(str(args.resume_epoch + args.epochs)))

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    # save final global model

    seconds = time.time() - start_time
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print("\n Total Run Time: %02d:%02d:%02d" % (h, m, s))
