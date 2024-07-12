import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair as pair
from torch.autograd import Variable
from torch.nn import init
import numpy as np

class MoME_Layer(Module):
    def __init__(self, in_features, out_features, num_experts, lamba_2, lamba_1, M,
                 sigma, sigma_neuron, bias=True, **kwargs):
        """
        :param in_features: Input dimensionality
        :param out_features: Output dimensionality
        :param bias: Whether we use a bias
        :param lamba_2: Strength of the L0 penalty
        """
        super(MoME_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.weight = Parameter(torch.Tensor(in_features, out_features), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        # neuron
        self.sigma_neuron = sigma_neuron
        self.lamba_1 = lamba_1
        # weight
        self.sigma = sigma
        self.lamba_2 = lamba_2

        self.use_bias = False
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            self.use_bias = True
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
        
        self.M = M
        if M > 0:
            self.set_parameters(if_neuron=True)
        else:
            self.set_parameters(if_neuron=False)

        self.reset_parameters()
        print(self)
        

    def set_parameters(self, if_neuron):
        if if_neuron:
            self.mu_neuron = Parameter(0.5*torch.ones(self.in_features, ), requires_grad=True)

        else:
            if self.M > 0:
                self.mu_neuron.requires_grad = False
                self.z_neuron = self.sample_z(None, sample=False, if_neuron=True)            
                self.z_neuron_nonzero_idx = torch.nonzero(self.z_neuron, as_tuple=False).squeeze()
                self.z_neuron_nonzero_num = self.z_neuron_nonzero_idx.shape[0]
                self.z_neuron = self.z_neuron.unsqueeze(1).expand_as(self.weight)
                self.weight = Parameter(torch.index_select(self.z_neuron*self.weight, dim=0, index=self.z_neuron_nonzero_idx), requires_grad=True)
                self.z_neuron = torch.index_select(self.z_neuron, dim=0, index=self.z_neuron_nonzero_idx)
                self.in_features = self.z_neuron_nonzero_num
            self.mu = nn.ParameterList([Parameter(0.5*torch.ones(self.in_features, self.out_features), requires_grad=True) for _ in range(self.num_experts)])

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.use_bias:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def regularizer(self, x):
        ''' Gaussian CDF. '''
        return 0.5 * (1 + torch.erf(x / math.sqrt(2))) 
    
    def stochastic_gate(self, if_neuron, expert=None, x=0):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        if if_neuron:
            z = self.mu_neuron + x*self.training 
        else:
            assert(expert!= None)
            z = self.mu[expert] + x*self.training 
        stochastic_gate = self.hard_sigmoid(z)
        return stochastic_gate

    def hard_sigmoid(self, x):
        # min max
        return torch.clamp(x, 0.0, 1.0)

    def regularization(self, if_neuron):
        cdf = []
        if if_neuron:
            cdf.append(self.lamba_1*torch.mean(self.regularizer((self.mu_neuron)/self.sigma_neuron)))
        else:
            for i in range(self.num_experts):
                cdf.append(self.lamba_2*torch.mean(self.regularizer((self.mu[i])/self.sigma)))
        return cdf

    def get_eps(self, size, if_neuron, expert=None):
        """Normal random numbers"""
        if not if_neuron:
            assert(expert!= None)
            eps = self.floatTensor(size).normal_(std=self.sigma[expert])
        else:
            eps = self.floatTensor(size).normal_(std=self.sigma_neuron)
        return eps
    
    def sample_z(self, if_neuron, expert=None, sample=True):
        """For testing"""
        assert(not sample)
        z = self.stochastic_gate(if_neuron=if_neuron, expert=expert)
        return F.hardtanh(z, min_val=0, max_val=1)
    
    def sample_weights(self, if_neuron, expert=None):
        """For training"""
        if not if_neuron:
            assert(expert!= None)
            eps = self.get_eps(self.floatTensor(self.in_features, self.out_features), if_neuron=False, expert=expert)
            mask = self.stochastic_gate(if_neuron=False, expert=expert, x=eps)
            return mask.view(self.in_features, self.out_features)
        else:
            eps = self.get_eps(self.floatTensor(self.in_features), if_neuron=True)
            mask = self.stochastic_gate(if_neuron=True, x=eps)
            return mask.view(self.in_features, 1)
    
    def get_gates(self, mode, if_neuron):
        if not if_neuron:
            if mode == 'raw':
                return [self.mu[t].detach().cpu().numpy() for t in range(self.num_experts)]
            elif mode == 'prob':
                return [np.minimum(1.0, np.maximum(0.0, self.mu[t].detach().cpu().numpy())) for t in range(self.num_experts)]
            else:
                raise NotImplementedError()
        else:
            if mode == 'raw':
                return [self.mu_neuron.detach().cpu().numpy()]
            elif mode == 'prob':
                return [np.minimum(1.0, np.maximum(0.0, self.mu_neuron.detach().cpu().numpy()))]
            else:
                raise NotImplementedError()

    def forward(self, input):
        if isinstance(input, torch.cuda.DoubleTensor):
            input = input.float()
        if_neuron = self.if_neuron

        if not if_neuron:
            if self.in_features != input.shape[-1]:
                input = torch.index_select(input, dim=len(input.shape)-1, index=self.z_neuron_nonzero_idx)
            weights = self.weight
                
            for t in range(self.num_experts):
                if not self.training:
                    z = self.sample_z(input[t].size(0), sample=self.training, expert=t, if_neuron=False)
                    if z.shape != weights.shape:
                        z = torch.unsqueeze(z, 1)
                        z = z.expand_as(weights)
                else:
                    z = self.sample_weights(expert=t, if_neuron=False)

                weights_e =  z * weights
                if len(input.shape) >2:
                    output_t = input[t].mm(weights_e)
                else:
                    output_t = input.mm(weights_e)

                if self.use_bias:
                    output_t.add_(self.bias)

                output_t = output_t.unsqueeze(0)
                if t == 0:
                    output = output_t
                else:
                    output = torch.cat((output, output_t), 0)
        else:
            if not self.training:
                z = self.sample_z(input.size(0), sample=self.training, if_neuron=True)
                if z.shape != self.weight.shape:
                    z = torch.unsqueeze(z, 1)
                    z = z.expand_as(self.weight)
            else:
                z = self.sample_weights(if_neuron=True)

            weights = z * self.weight
            output = input.mm(weights)
            if self.use_bias:
                output.add_(self.bias)
            
        return output

    def __repr__(self):
        s = ('{name}({in_features} -> {out_features}, '
             'lamba_1={lamba_1}, sigma_neuron={sigma_neuron}, '
             'lamba_2={lamba_2}, sigma={sigma}, '
             'M={M}')
        if not self.use_bias:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
    ######################################################## END ###########################################################
