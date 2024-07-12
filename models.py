import torch
import torch.nn as nn
import numpy as np
import math
from layers import MoME_Layer
from copy import deepcopy
import torch.nn.functional as F


# ref: layers.py in https://github.com/morningsky/multi_task_learning
class MultiLayerPerceptron(torch.nn.Module):

    # def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
    def __init__(self, input_dim, embed_dims,  
                 dropout=0.2, if_dropout=True, 
                 output_dim=1, output_layer=True, 
                 if_bn=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            if if_bn:
                layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            if if_dropout:
                layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, output_dim))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)

# ref: layers.py in https://github.com/morningsky/multi_task_learning
class EmbeddingLayer(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)

class MoME(nn.Module):
    def __init__(self, input_dim, num_classes,
                 num_tasks, num_experts, 
                 data_name,
                 expert_layer_dims, 
                 lamba_1,lamba_2,M,
                 embed, embed_dim, categorical_field_dims, numerical_num, 
                 if_tower, tower_layer_dims,
                 sigma=0.5, sigma_neuron=0.5): 
        super(MoME, self).__init__()
        self.expert_layer_dims = expert_layer_dims
        
        self.num_tasks = num_tasks
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.linear_use_bias = True
        self.data_name = data_name

        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
        
        self.if_tower = if_tower
        self.embed = embed
        self.M = M

        if self.embed:
            assert(all(categorical_field_dims) and embed_dim)
            self.embedding = EmbeddingLayer(categorical_field_dims, embed_dim)
            # self.numerical_layer = torch.nn.Linear(numerical_num, embed_dim)
            self.embedding_output_dim = (len(categorical_field_dims)) * embed_dim
            self.input_dim = (len(categorical_field_dims)) * embed_dim + numerical_num
        else:
            assert(input_dim!=None)
            self.input_dim = input_dim
        self.expert_input_dim = self.input_dim

        layers = []
        for i, dimh in enumerate(self.expert_layer_dims):
            inp_dim = self.expert_input_dim if i == 0 else self.expert_layer_dims[i - 1]
            layers += [MoME_Layer(inp_dim, dimh, num_experts=self.num_experts,
                               lamba_2=lamba_2, lamba_1=lamba_1, 
                               sigma=sigma, sigma_neuron=sigma_neuron, M=M), 
                       nn.BatchNorm1d(dimh),
                       nn.ReLU(),
                       torch.nn.Dropout(p=0.2)]
            
        if not if_tower:
            layers.append(MoME_Layer(self.expert_layer_dims[-1], num_classes, num_experts=self.num_experts,
                                  lamba_2=lamba_2, lamba_1=lamba_1, M=M))
        
        self.output = nn.Sequential(*layers)
        self.layers = []
        for m in layers:
            if isinstance(m, MoME_Layer):
                self.layers.append(m)   
                 
        self.num_tasks = num_tasks
        if if_tower:
            if isinstance(num_classes, int):
                num_classes=[num_classes]
            if isinstance(tower_layer_dims, int):
                tower_layer_dims=[tower_layer_dims]
            self.tower = torch.nn.ModuleList([MultiLayerPerceptron(
                self.expert_layer_dims[-1], tower_layer_dims, 
                output_dim=num_classes[i]) for i in range(num_tasks)])


    def set_if_neuron(self, epoch):
        self.if_neuron = epoch<self.M
        for i, layer in enumerate(self.layers):
            layer.if_neuron = self.if_neuron
            if not self.if_neuron and epoch==self.M:
                layer.set_parameters(False)
        if epoch==self.M:
            self.mmoe_gate = nn.ModuleList([nn.Sequential(torch.nn.Linear(self.expert_input_dim, self.num_experts), nn.Softmax(dim=1)) for i in range(self.num_tasks)])
        

    def forward(self, categorical_x, numerical_x, epoch):
        if self.embed:
            categorical_emb = self.embedding(categorical_x).view(-1,self.embedding_output_dim)
            numerical_emb = numerical_x
            x = torch.cat([categorical_emb, numerical_emb], 1)
        else:
            x = torch.cat([categorical_x, numerical_x], 1)

        if isinstance(x, torch.cuda.DoubleTensor):
            x = x.float() 

        self.info = self.output[0](x).shape
        if not self.if_neuron:
            expert_output = x.clone()
            for i in range(len(self.expert_layer_dims)):
                # MoME_Layer
                expert_output = self.output[4*i](expert_output)
                # BatchNorm1d, relu, dropout
                for e in range(self.num_experts):
                    if i != len(self.expert_layer_dims)-1:
                        expert_output_e = self.output[4*i+1:4*(i+1)](expert_output[e])  
                    else:
                        expert_output_e = self.output[4*i+1:](expert_output[e])  
                    expert_output_e = expert_output_e.unsqueeze(0)
                    if e == 0:
                        expert_output_next = expert_output_e
                    else:
                        expert_output_next = torch.cat((expert_output_next, expert_output_e), 0)
                expert_output = expert_output_next
            expert_output = expert_output.transpose(0,1)

            gate_value = [self.mmoe_gate[i](x).unsqueeze(1) for i in range(self.num_tasks)]
            task_fea = [torch.bmm(gate_value[i], expert_output).squeeze(1) for i in range(self.num_tasks)]
            if self.if_tower:
                results = [torch.sigmoid(self.tower[i](task_fea[i]).squeeze(1)) for i in range(self.num_tasks)]
            else:
                results = [torch.sigmoid(task_fea[i]).squeeze(1) for i in range(self.num_tasks)]
            
            return results
        else:
            expert_output = self.output(x)
            results = [torch.sigmoid(self.tower[i](expert_output).squeeze(1)) for i in range(self.num_tasks)]
            return results

    def regularization(self):
        if self.if_neuron:
            regularization = [0 for _ in range(1)]
            for i, layer in enumerate(self.layers):
                regularization =  list(map(lambda x, y: x+y, regularization, layer.regularization(self.if_neuron)))

            if torch.cuda.is_available():
                regularization = [regularization[0].cuda()]

        else:
            regularization = [0 for _ in range(self.num_experts)]
            for i, layer in enumerate(self.layers):
                regularization =  list(map(lambda x, y: x+y, regularization, layer.regularization(self.if_neuron)))

            if torch.cuda.is_available():
                regularization = [regularization[t].cuda() for t in range(self.num_experts)]
        return regularization


