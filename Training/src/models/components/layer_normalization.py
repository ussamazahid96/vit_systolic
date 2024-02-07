import torch
from torch.nn import LayerNorm
from typing import Union, Type, Optional
from torch import Tensor
from brevitas.quant_tensor import QuantTensor
from brevitas.nn.quant_layer  import WeightQuantType, BiasQuantType, ActQuantType
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL


__all__ = ['QuantLayerNorm']


class QuantLayerNorm(QuantWBIOL, LayerNorm):

    def __init__(self,
                 normal_shape,
                 eps=1e-5,
                 weight_quant: Optional[WeightQuantType] = None,
                 bias_quant: Optional[BiasQuantType] = None,
                 input_quant: Optional[ActQuantType] = None,
                 output_quant: Optional[ActQuantType] = None,
                 return_quant_tensor: bool = False,
                 **kwargs):
        """Layer normalization layer
        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
        :param normal_shape: The shape of the input tensor or the last dimension of the input tensor.
        :param gamma: Add a scale parameter if it is True.
        :param beta: Add an offset parameter if it is True.
        :param eps: eps for calculating variance.
        """
        LayerNorm.__init__(self, normal_shape, eps=eps)
        QuantWBIOL.__init__(
            self,
            weight_quant=weight_quant,
            bias_quant=bias_quant,
            input_quant=input_quant,
            output_quant=output_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)
    
    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)
        torch.nn.init.ones_(self.bias)
        self.bias.data *= 0.1
    
    def forward(self, input: Union[Tensor, QuantTensor]) -> Union[Tensor, QuantTensor]:
        return self.forward_impl(input)

    def inner_forward_impl(self, x: Tensor, quant_weight: Tensor, quant_bias: Optional[Tensor]):
        mean = x.mean(dim=-1, keepdim=True)
        mean.data = torch.floor(mean.data*2**16)/2**16
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        var.data = torch.floor(var.data*2**16)/2**16
        std = (var + self.eps).sqrt()
        std.data = torch.floor(std.data*2**16)/2**16
        y = (x - mean) / std
        
        if quant_weight is not None:
            y *= quant_weight
        if quant_bias is not None:
            y += quant_bias
        
        return y