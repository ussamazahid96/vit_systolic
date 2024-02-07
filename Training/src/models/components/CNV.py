from black import out
from torch import nn

from brevitas import nn as qnn
from brevitas.quant.solver import WeightQuantSolver, ActQuantSolver

import hydra
from omegaconf import OmegaConf, DictConfig

from src.utils.tensor_norm import TensorNorm
from .quantizers import *

def get_quantcnv(input_size: int, 
                 output_size: int,
                 quantizer_cfg: DictConfig = None):
    iq = get_input_quantizer(quantizer_cfg.input_quantizer)
    wq = get_weight_quantizer(quantizer_cfg.weight_quantizer)
    aq = get_activation_quantizer(quantizer_cfg.activation_quantizer)
    return nn.Sequential(
            qnn.QuantConv2d(in_channels=input_size, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False,
                            input_quant=iq, weight_quant=wq, bias_quant=None, output_quant=None),
            nn.BatchNorm2d(64),
            qnn.QuantIdentity(act_quant=aq),

            qnn.QuantConv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False,
                            input_quant=None, weight_quant=wq, bias_quant=None, output_quant=None),
            nn.BatchNorm2d(64),
            qnn.QuantIdentity(act_quant=aq),            
            nn.MaxPool2d(kernel_size=2),

            qnn.QuantConv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0, bias=False,
                            input_quant=None, weight_quant=wq, bias_quant=None, output_quant=None),
            nn.BatchNorm2d(128),
            qnn.QuantIdentity(act_quant=aq),
            qnn.QuantConv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0, bias=False,
                            input_quant=None, weight_quant=wq, bias_quant=None, output_quant=None),
            nn.BatchNorm2d(128),
            qnn.QuantIdentity(act_quant=aq),            
            nn.MaxPool2d(kernel_size=2),        

            qnn.QuantConv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0, bias=False,
                            input_quant=None, weight_quant=wq, bias_quant=None, output_quant=None),
            nn.BatchNorm2d(256),
            qnn.QuantIdentity(act_quant=aq),
            qnn.QuantConv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0, bias=False,
                            input_quant=None, weight_quant=wq, bias_quant=None, output_quant=None),
            nn.BatchNorm2d(256),
            qnn.QuantIdentity(act_quant=aq),    
            nn.Flatten(),
            qnn.QuantLinear(in_features=256, out_features=512, bias=False,
                            input_quant=None, weight_quant=wq, bias_quant=None, output_quant=None),
            nn.BatchNorm1d(512),
            qnn.QuantIdentity(act_quant=aq),
            qnn.QuantLinear(in_features=512, out_features=output_size, bias=False,
                            input_quant=None, weight_quant=wq, bias_quant=None, output_quant=None),
            TensorNorm(),
            
        )

def get_fpcnv(input_size, output_size):
    return nn.Sequential(
            nn.Conv2d(in_channels=input_size, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),        

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
    
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Linear(512, output_size)
        )

class SimpleCNV(nn.Module):
    def __init__(
        self,
        input_size: int = 3,
        output_size: int = 10,
        quantization: bool = False,
        quantizer_cfg: DictConfig = None
    ):
        super().__init__()

        self.quantization = quantization
        if quantization:
            self.model = get_quantcnv(input_size=input_size, output_size=output_size, quantizer_cfg=quantizer_cfg)
        else:
            self.model = get_fpcnv(input_size=input_size, output_size=output_size)

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        y = self.model(x)
        return y 
