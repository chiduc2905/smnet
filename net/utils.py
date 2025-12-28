import torch
import torch.nn as nn
from torch.nn import init
import math


def weights_init_relationnet(m):
    """
    RelationNet paper-specific initialization.
    Based on official repo: floodsung/LearningToCompare_FSL
    
    Conv: He-like init with variance = 2/n
    BatchNorm: weight=1, bias=0
    Linear: Normal(0, 0.01), bias=1
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and hasattr(m, 'weight'):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1 and hasattr(m, 'weight'):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1 and hasattr(m, 'weight'):
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and hasattr(m, 'weight'):
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1 and hasattr(m, 'weight'):
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1 and hasattr(m, 'weight'):
        init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('GroupNorm') != -1 and hasattr(m, 'weight'):
        init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and hasattr(m, 'weight'):
        init.xavier_normal_(m.weight.data, gain=1.0)  # Fixed: was 0.02
    elif classname.find('Linear') != -1 and hasattr(m, 'weight'):
        init.xavier_normal_(m.weight.data, gain=1.0)  # Fixed: was 0.02
    elif classname.find('BatchNorm2d') != -1 and hasattr(m, 'weight'):
        init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('GroupNorm') != -1 and hasattr(m, 'weight'):
        init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)

def weights_init_kaiming(m):
    """Kaiming (He) initialization - RECOMMENDED for ReLU/LeakyReLU."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and hasattr(m, 'weight'):
        # mode='fan_in' preserves magnitude in forward pass
        # nonlinearity='leaky_relu' with a=0.2 matches our LeakyReLU(0.2)
        init.kaiming_normal_(m.weight.data, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
    elif classname.find('Linear') != -1 and hasattr(m, 'weight'):
        init.kaiming_normal_(m.weight.data, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
    elif classname.find('BatchNorm2d') != -1 and hasattr(m, 'weight'):
        init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('GroupNorm') != -1 and hasattr(m, 'weight'):
        init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and hasattr(m, 'weight'):
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1 and hasattr(m, 'weight'):
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1 and hasattr(m, 'weight'):
        init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('GroupNorm') != -1 and hasattr(m, 'weight'):
        init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)

def init_weights(net, init_type='kaiming'):
    """Initialize network weights.
    
    Args:
        net: network to initialize
        init_type: 'normal' | 'xavier' | 'kaiming' (default) | 'orthogonal'
    """
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
