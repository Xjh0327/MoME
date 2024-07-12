import argparse
import shutil
import os
import time
import matplotlib.pylab as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from sklearn.metrics import roc_auc_score

from models import MoME
from dataloaders import load_data
from utils import AverageMeter, print_gates
from layers import MoME_Layer


class MoMEModel(nn.Module):
    """
    topk_dim = -1: select topk in the whole matrix
    topk_dim = 1: expert dim
    topk_dim = 0: sample dim
    """
    def __init__(self, data_name, epochs=100, start_epoch=0, batch_size=1024, lr=1e-3, 
                 print_freq=100, name='MoME', 
                 weight_decay=0, 
                 num_expert=8, num_task=2,
                 M=50, lamba_1=1e-2, lamba_2=1e-3, 
                 embed_dim=128,
                 expert_layer_dims=(16,8), 
                 tower_layer_dims=(8),
                 sigma=0.5, sigma_neuron=0.5, if_tower=True, 
                 augment=True):
        super(MoMEModel, self).__init__()

        embed=True
        self.embed_dim = embed_dim
        self.lossfunc = nn.BCELoss()

        self.epochs=epochs
        self.start_epoch=start_epoch
        self.batch_size=batch_size

        self.lr=lr
        self.weight_decay=weight_decay

        self.print_freq=print_freq

        self.lamba_1 = lamba_1
        self.lamba_2=lamba_2
        self.num_expert=num_expert
        self.num_task=num_task
        self.M = M
        
        
        self.model_name = '{}/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                                    data_name, num_expert, lr, epochs,
                                    batch_size,  weight_decay,
                                    str(expert_layer_dims), 
                                    str(tower_layer_dims),
                                    M, lamba_1, lamba_2, sigma)
        
        if torch.cuda.is_available():
            self.lossfunc = self.lossfunc.cuda()

        self.name=name
        self.data_name=data_name

        self.best_prec1 = 0

     
############################################# PREPARATION #########################################
        print('model:', self.name)

        print('Preparing data...')
        self.train_loader, self.test_loader, self.num_classes, field_dims, numerical_num = load_data(batch_size=self.batch_size, data_name=self.data_name)
        

        self.model = MoME(None, self.num_classes, 
                             num_task, num_expert, 
                             self.data_name,
                             expert_layer_dims, 
                             lamba_1=lamba_1, lamba_2=lamba_2, M=M,
                             embed=embed, embed_dim=embed_dim, categorical_field_dims=field_dims, numerical_num=numerical_num,
                             if_tower=if_tower, tower_layer_dims=tower_layer_dims,
                             sigma=sigma, sigma_neuron=sigma_neuron
                             )
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), self.lr, weight_decay=self.weight_decay)
        
        num_para = 0
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                print(name, end='\t')
                num_para+= p.data.nelement()
        print()
        print('Number of model parameters: {}, \nNumber of model required grad parameters: {}'.format(sum([p.data.nelement() for p in self.model.parameters()]), num_para))

    # define loss function 
    def loss_function(self, output, target, model):
        loss = []
        for t in range(model.num_tasks):
            loss.append(self.lossfunc(output[t], target[:,t].float()))
        reg = model.regularization()
        reg_sum = sum(reg)
        total_loss = [loss[t]+reg_sum for t in range(model.num_tasks)]
        if torch.cuda.is_available():
            total_loss = [total_loss[t].cuda() for t in range(model.num_tasks)]
        
        return total_loss, loss, reg_sum
    
    def print_summary(self,epoch):
        print('Best error average at epoch {}: '.format(epoch), self.best_prec1)
        print('Best error list: ', self.best_prec1_list)
        _, gates_string = print_gates(torch.load('../model/'+self.model_name+'.pt'), if_print=True)
        

    def train(self):
        it = iter(range(self.start_epoch, self.epochs))
        jump = False
        for epoch in it:
            self.model.set_if_neuron(epoch)
            if epoch == self.M:
                self.best_prec1 = 0
                if torch.cuda.is_available():
                    self.model = self.model.cuda()
                self.choose_optimizer()

            self.train_epoch(self.model, self.loss_function, epoch, self.optimizer)

            # evaluate on validation set
            prec1_list, loss_list = self.test_epoch(self.model, self.loss_function, epoch)
            prec1 = sum(prec1_list)/len(prec1_list)
            is_best = prec1 > self.best_prec1

            self.best_prec1 = max(prec1, self.best_prec1)
            if is_best:
                self.best_prec1_list = prec1_list
                self.best_loss_list = loss_list
                self.best_prec1_epoch = epoch
                torch.save(self.model, '../model/'+self.model_name+'.pt')
                state = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': epoch}
                torch.save(state, '../model/state_'+self.model_name+'.pt')

            print('Best error average at epoch {}: '.format(epoch), self.best_prec1)
            print('Best error list: ', self.best_prec1_list)
            print_gates(torch.load('../model/'+self.model_name+'.pt'), if_print=True)

            if self.epochs > 50 and epoch==49:
                self.print_summary(epoch, prec1, prec1_list, loss_list)
            if epoch == self.M-1:
                print('Stop model pruning, with best auc=', self.best_prec1_list, self.best_prec1)
                self.print_summary(epoch, prec1, prec1_list, loss_list)
                self.best_prec1 = 0

        self.print_summary(epoch, prec1, prec1_list, loss_list)
    

    def train_epoch(self, model, criterion, epoch, optimizer=None):
        """Train for one epoch on the training set"""
        batch_time = AverageMeter()
        data_time = AverageMeter()
        ls = AverageMeter(is_list=True, num_task=model.num_tasks)
        regs = AverageMeter()
        avg_loss = AverageMeter()

        end = time.time()
        model.train()
        loader = self.train_loader
        for i, (categorical_input, numerical_input, target) in enumerate(loader):
            num_sample = categorical_input.shape[0]
            data_time.update(time.time() - end)
            if torch.cuda.is_available():
                target = target.cuda()
                categorical_input = categorical_input.cuda()
                numerical_input = numerical_input.cuda()
            
            # compute output
            output = model(categorical_input, numerical_input, epoch)
            # criterion = loss func
            loss, l, reg = criterion(output, target, model)
            avg_l = sum(loss)/model.num_tasks
            ls.update(l, num_sample)
            regs.update(reg.item(), num_sample)
            avg_loss.update(avg_l.item(), num_sample)
            # compute gradient and do SGD step
            optimizer.zero_grad()
            avg_l.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i % self.print_freq == 0) or (i == len(loader)-1):
                progress_string = ' Epoch: [{0}][{1}/{2}]\t'\
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                        epoch, i, len(loader), batch_time=batch_time)
                progress_string += 'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                        data_time=data_time)
                progress_string += 'Avg Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(loss=avg_loss)
                loss_string = ''
                for t in range(model.num_tasks):
                    loss_string+='  Loss_{t} {val:.4f} ({avg:.4f})\t Reg_{t} {rval:.4f} ({ravg:.4f})\t'.format(
                        t=str(t), val=ls.val[t], avg=ls.avg[t], rval=regs.val, ravg=regs.avg) 
                print(progress_string)
                print(loss_string)
                print_gates(model)


    def test_epoch(self, model, criterion, epoch):
        """Train for one epoch on the training set"""
        batch_time = AverageMeter()
        ls = AverageMeter(is_list=True, num_task=model.num_tasks)
        regs = AverageMeter()
        avg_loss = AverageMeter()

        end = time.time()
        model.eval()
        loader = self.test_loader

        labels_dict, predicts_dict = {}, {}
        for t in range(model.num_tasks):
            labels_dict[t], predicts_dict[t] = list(), list()

        with torch.no_grad():
            for i, (categorical_input, numerical_input, target) in enumerate(loader):
                num_sample = categorical_input.shape[0]
                if torch.cuda.is_available():
                    target = target.cuda()
                    categorical_input = categorical_input.cuda()
                    numerical_input = numerical_input.cuda()

                # compute output
                output = model(categorical_input, numerical_input, epoch)

                for t in range(model.num_tasks):
                    labels_dict[t].extend(target[:, t].tolist())
                    predicts_dict[t].extend(output[t].tolist())
                
                loss, l, reg = criterion(output, target, model)
                avg_l = sum(loss)/model.num_tasks
                
                ls.update(l, num_sample)
                regs.update(reg.item(), num_sample)
                avg_loss.update(avg_l.item(), num_sample)
                batch_time.update(time.time() - end)
                end = time.time()

                if (i % self.print_freq == 0) or (i == len(loader)-1):
                    progress_string = ' Epoch: [{0}][{1}/{2}]\t'\
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                            epoch, i, len(loader), batch_time=batch_time)
                    progress_string += 'Avg Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(loss=avg_loss)
                    loss_string = ''
                    for t in range(model.num_tasks):
                        loss_string+='  Loss_{t} {val:.4f} ({avg:.4f})\t Reg_{t} {rval:.4f} ({ravg:.4f})\t'.format(
                            t=str(t), val=ls.val[t], avg=ls.avg[t], rval=regs.val, ravg=regs.avg) 
                    print(progress_string)
                    print(loss_string)
                    gate_info_list, _ = print_gates(model)

        auc_results = list()
        for t in range(model.num_tasks):
            auc_results.append(roc_auc_score(labels_dict[t], predicts_dict[t]))

        tmp = ['AUC_{t} {acc_avg:.4f}\t'.format(
                    t=str(t), acc_avg=auc_results[t]) for t in range(model.num_tasks)]
        print(' *', *tmp)


        return auc_results, ls.avg

