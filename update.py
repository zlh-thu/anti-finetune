#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from copy import deepcopy



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
    def __init__(self, args, train_dataset, test_dataset, changing_round, logger, user_id):
        self.args = args
        self.logger = logger
        self.trainloader = DataLoader(train_dataset, batch_size=args.local_bs, shuffle=True)
        self.testloader = DataLoader(test_dataset, batch_size=args.local_bs, shuffle=False)
        self.train_iter = iter(deepcopy(self.trainloader))
        self.user_id = user_id
        self.batch_id = 0

        if args.gpu >= 0:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.criterion_mean = nn.CrossEntropyLoss(reduction='mean').to(self.device)
        self.criterion_max = nn.CrossEntropyLoss(reduction='none').to(self.device)
        self.changing_round = changing_round

    def update_weights(self, model, global_round, mood, target_client):
        # Set mode to train model
        if self.args.frozen:
            for param in model.backbone.parameters():
                param.require_grad = False
            model.linear.require_grad = True
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        attack_item = None
        grad = None
        if global_round >= self.changing_round and (self.user_id in target_client):

            if mood == 'max_loss':
                attack_item, grad = self.Max_loss_of_batch_update_every_batch(model=model, global_round=global_round,
                                                                              optimizer=optimizer,
                                                                              epoch_loss=epoch_loss,
                                                                              dataset=self.args.dataset)

            elif mood == 'avg_loss':
                attack_item = self.Mean_loss_of_batch_update_every_batch(model=model, global_round=global_round,
                                                                         optimizer=optimizer,
                                                                         epoch_loss=epoch_loss)

        elif global_round >= self.changing_round and (self.user_id not in target_client) and mood == 2:
            _ = self.Mean_loss_of_batch_update_every_batch(model=model, global_round=global_round, optimizer=optimizer,
                                                           epoch_loss=epoch_loss)

        else:
            # regular training with mean loss of a batch
            _ = self.Mean_loss_of_batch_update_every_batch(model=model, global_round=global_round,
                                                           optimizer=optimizer,
                                                           epoch_loss=epoch_loss)

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), attack_item, grad

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (items) in enumerate(self.testloader):
            images, labels = items["img"].to(self.device), items["target"].to(self.device)

            # Inference
            outputs = model(images)

            batch_loss = self.criterion_mean(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct / total
        return accuracy, loss

    def Max_loss_of_batch_update_every_batch(self, model, global_round, optimizer, epoch_loss, dataset):
        # mood == 'max_loss'
        input_gradient = None
        if dataset == 'imagenet':
            global_model = deepcopy(model)
            global_model.eval()

        for iter in range(self.args.local_ep):
            items, batch_idx = self.get_a_batch()
            # print(batch)
            batch_loss = []
            images, labels = items["img"].to(self.device), items["target"].to(self.device)

            model.zero_grad()
            if self.args.frozen:
                with torch.no_grad():
                    feature = model.backbone(images)
                log_probs = model.linear(feature)
            else:
                log_probs = model(images)
            # log_probs = model(images)

            loss = self.criterion_max(log_probs, labels)
            batch_max_loss, max_id_in_batch = torch.max(loss, 0)

            optimizer.zero_grad()
            batch_max_loss.backward()
            optimizer.step()

            if dataset == 'imagenet':
                image, label = items["img"][max_id_in_batch].to(self.device), items["target"][max_id_in_batch].to(
                    self.device)
                image = torch.unsqueeze(image, 0)
                label = torch.unsqueeze(label, 0)
                loss = self.criterion_max(global_model(image), label)
                input_gradient = torch.autograd.grad(loss, global_model.parameters())
                input_gradient = [grad.detach() for grad in input_gradient]

            if self.args.model == 'convnet64':
                max_item = {
                    "img": items['img'][max_id_in_batch],
                    "target": items['target'][max_id_in_batch],
                    "index": items['index'][max_id_in_batch],
                    "client_ID": items['client_ID'][max_id_in_batch],
                    "bs_mean_list": model.bs_mean_list,
                    "bs_var_list": model.bs_var_list
                }

            else:
                max_item = {
                    "img": items['img'][max_id_in_batch],
                    "target": items['target'][max_id_in_batch],
                    "index": items['index'][max_id_in_batch],
                    "client_ID": items['client_ID'][max_id_in_batch]
                }

            if self.args.verbose and (batch_idx % 20 == 0):
                print(
                    '| Global Round : {} | User ID : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t(max of a batch, update every batch)'.format(
                        global_round, self.user_id, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                                                          100. * batch_idx / len(self.trainloader),
                        batch_max_loss.item()))
            self.logger.add_scalar('loss', batch_max_loss.item())
            batch_loss.append(batch_max_loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return max_item, input_gradient

    def Mean_loss_of_batch_update_every_batch(self, model, global_round, optimizer, epoch_loss):
        # mood = 3 or  mood = 2, client not in the target client list
        for iter in range(self.args.local_ep):
            items, batch_idx = self.get_a_batch()
            # print(batch)
            batch_loss = []
            images, labels = items["img"].to(self.device), items["target"].to(self.device)

            model.zero_grad()
            if self.args.frozen:
                with torch.no_grad():
                    feature = model.backbone(images)
                log_probs = model.linear(feature)
            else:
                log_probs = model(images)
            loss = self.criterion_mean(log_probs, labels)
            loss_max = self.criterion_max(log_probs, labels)
            batch_max_loss, max_id_in_batch = torch.max(loss_max, 0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            attack_item = {
                "img": items['img'],
                "target": items['target'],
                "index": items['index'],
                "client_ID": items['client_ID'],
                "max_id_in_batch": int(max_id_in_batch)
            }

            if self.args.verbose and (batch_idx % 20 == 0):
                print(
                    '| Global Round : {} | User ID : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t(mean of a batch, update every batch)'.format(
                        global_round, self.user_id, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                                                          100. * batch_idx / len(self.trainloader),
                        loss.item()))
            self.logger.add_scalar('loss', loss.item())
            batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return attack_item

    def get_a_batch(self):
        try:
            batch = next(self.train_iter)
        except StopIteration:
            import gc
            del self.train_iter
            gc.collect()
            self.batch_id = 0
            self.train_iter = iter(deepcopy(self.trainloader))
            batch = next(self.train_iter)
        self.batch_id = self.batch_id + 1
        return batch, self.batch_id

def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    if args.gpu >= 0:
        device = 'cuda'
    else:
        device = 'cpu'
    criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (items) in enumerate(testloader):
        images, labels = items["img"].to(device), items["target"].to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss



