import torch
import numpy as np
import os
import shutil
import sys
from sklearn.metrics import roc_auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, is_list=False, num_task=None):
        self.is_list = is_list
        self.num_task = num_task
        self.reset()

    def reset(self):
        if not self.is_list:
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0
        else:
            self.val, self.sum, self.avg = [], [], []
            for t in range(self.num_task):
                self.val.append(0)
                self.sum.append(0)
                self.avg.append(0)
            self.count = 0

    def update(self, val, n=1):
        if not self.is_list:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
        else:
            self.count += n
            for t in range(self.num_task):
                self.val[t] = val[t]
                self.sum[t] += val[t] * n
                self.avg[t] = self.sum[t] / self.count
            

def print_gates(model, if_print=True):
    info_list = []
    info_string = ''

    for i, l in enumerate(model.layers):
        info_string+='  Layer {0}: '.format(i)
        if model.M>0:
            gate_neuron = l.get_gates('prob', if_neuron=True)
            info_string+='  Layer {0} first mask has {1}/{2} zero gates, {3}/{2} one gates, 0.5 gate= {4}\n'.format(i, np.sum(gate_neuron[0] == 0), gate_neuron[0].size, np.sum(gate_neuron[0] == 1), np.sum(gate_neuron[0])/gate_neuron[0].size)
            info_list.append([[int(np.sum(gate_neuron[0] == 0)), int(np.sum(gate_neuron[0] == 1)), int(gate_neuron[0].size)]])
        
        if not model.if_neuron:
            gate = l.get_gates('prob', if_neuron=False)
            for t in range(len(gate)):
                info_string+='   Expert {0} has {1}/{2} zero gates, {3}/{2} one gates, 0.5 gate= {4};\n'.format(t, np.sum(gate[t] == 0), gate[t].size, np.sum(gate[t] == 1), np.sum(gate[t])/gate[t].size)
            info_list.append([[int(np.sum(gate[t] == 0)), int(np.sum(gate[t] == 1)), int(gate[t].size)] for t in range(model.num_experts)])
                
    if if_print:
        print(info_string)
    return info_list, info_string


