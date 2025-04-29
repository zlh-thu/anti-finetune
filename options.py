#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=200,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=8,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=1.0,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=1,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")
    parser.add_argument('--batch_size', type=int, default=10,
                        help="batch size of leakage attack")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--exp_id', type=int, default=0,
                        help="id of exp")
    parser.add_argument('--resume', type=str, default=None,
                        help="resume from path")
    parser.add_argument('--resume_epoch', type=int, default=0,
                        help="resume epoch")
    parser.add_argument('--mood', type=str, default=2,
                        help="avg_loss, max_loss"
                        )
    parser.add_argument('--attack', type=str, default='none', choices=['anti_finetune','none'],
                        help="attack method")

    parser.add_argument('--alpha', type=float, default=1.0, help='coefficient of the target client model in fedavg')
    parser.add_argument('--pretrained', action='store_true',
                        help='Whether train from pretrained model')
    parser.add_argument('--frozen', action='store_true', help='Whether frozen the resnet backbone.')

    # model arguments
    parser.add_argument('--model', type=str, default='resnet18', help='model name')

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--load_data_from_dir', type=str, default='../data', help="load \
                        data for clients from dir")
    parser.add_argument('--gpu', default=0, type=int, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--changing_round', type=int, default=100,
                        help='rounds of changing loss to max')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--save_interval', type=int, default=1, help='save checkpoint interval')
    parser.add_argument('--test_interval', type=int, default=1, help='test global model interval')

    # grad leakage arguments
    parser.add_argument('--dtype', default='float', type=str,
                        help='Data type used during reconstruction [Not during training!].')

    parser.add_argument('--restarts', default=1, type=int, help='How many restarts to run.')
    parser.add_argument('--cost_fn', default='sim', type=str, help='Choice of cost function.')
    parser.add_argument('--indices', default='def', type=str, help='Choice of indices from the parameter list.')
    parser.add_argument('--weights', default='equal', type=str, help='Weigh the parameter list differently.')

    parser.add_argument('--max_iterations', default=4000, type=int, help='max_iterations')
    parser.add_argument('--attack_num', default=None, type=int, help='attack times')

    # anti_finetune_ep
    parser.add_argument('--anti_finetune_ep', default=5, type=int, help='epoch of fake finetune')
    parser.add_argument('--use_target_label', action='store_true', help='Whether use target label as init label')
    parser.add_argument('--anti_finetune_lr', type=float, default=0.0001,
                        help='learning rate of fake finetune')

    parser.add_argument('--rec_optimizer', default='adam', type=str, help='Weigh the parameter list differently.')
    parser.add_argument('--signed', action='store_false', help='Do not used signed gradients.')
    parser.add_argument('--boxed', action='store_false', help='Do not used box constraints.')

    parser.add_argument('--scoring_choice', default='loss', type=str,
                        help='How to find the best image between all restarts.')
    parser.add_argument('--init', default='randn', type=str, help='Choice of image initialization.')
    parser.add_argument('--tv', default=1e-4, type=float, help='Weight of TV penalty.')
    parser.add_argument('--dryrun', action='store_true', help='Run everything for just one step to test functionality.')
    parser.add_argument('--leakage_attack', action='store_true', help='Whether run leakage attack')
    parser.add_argument('--fix_bn', action='store_true', help='Whether fix bn layerwatc')

    args = parser.parse_args()
    return args