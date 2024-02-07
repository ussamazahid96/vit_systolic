# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_for_small_dataset.py

import os
import numpy as np
from omegaconf import DictConfig

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

import brevitas
from brevitas import nn as qnn
from brevitas.nn import QuantIdentity
from brevitas.quant import Int8WeightPerTensorFixedPoint, Int8ActPerTensorFixedPoint, Int8BiasPerTensorFixedPointInternalScaling

from .quantizers import *
from .utils import pack_weights
from .layer_normalization import QuantLayerNorm

# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class MyQuant(Int8ActPerTensorFixedPoint):
    float_to_int_impl_type=brevitas.inject.enum.FloatToIntImplType.FLOOR

class QuantSoftMax(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.int_bits = 10
        self.frac_bits = 8
        self.max_int = 2**self.int_bits
        self.scale = 2**self.frac_bits
    
    def forward(self, x):
        x = x.exp()
        x.data = torch.clamp(x.data, 0, self.max_int)
        x.data = torch.floor(x.data*self.scale)/self.scale
        x /= x.sum(dim=-1)[...,None]
        return x

# classes
class SPT(nn.Module):
    def __init__(self, *, dim, patch_size, channels = 3):
        super().__init__()
        patch_dim = patch_size * patch_size * 5 * channels

        self.to_patch_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
        )

    def forward(self, x):
        shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
        shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))
        x_with_shifts = torch.cat((x, *shifted_x), dim = 1)
        return self.to_patch_tokens(x_with_shifts)

class MHA(nn.Module):
    def __init__(self, dim,  SA_SIZE, heads = 8, dim_head = 64, dropout = 0., quantization: bool = False, quantizer_cfg: DictConfig = None):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        ioq = MyQuant if quantization else None
        wq = get_weight_quantizer(quantizer_cfg.weight_quantizer) if quantization else None
        
        self.norm = QuantLayerNorm(dim, weight_quant=Int8WeightPerTensorFixedPoint, bias_quant=Int8BiasPerTensorFixedPointInternalScaling)
        self.to_qkv = qnn.QuantLinear(in_features=dim, out_features=inner_dim*3, bias=False, 
                                      input_quant=ioq, weight_quant=wq, bias_quant=None, output_quant=ioq, 
                                      return_quant_tensor=False)
        self.temperature = nn.Parameter(torch.log(torch.tensor(dim_head ** -0.5)))
        self.asquant = QuantIdentity(act_quant=ioq, return_quant_tensor=False)
        self.attend = nn.Softmax(dim=-1)
        # self.attend = QuantSoftMax(dim = -1)
        self.aquant = QuantIdentity(act_quant=ioq, return_quant_tensor=False)
        self.to_out = qnn.QuantLinear(in_features=inner_dim, out_features=dim, bias=False, 
                                      input_quant=ioq, weight_quant=wq, bias_quant=None, output_quant=ioq, 
                                      return_quant_tensor=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)      
        
        temp = self.temperature.exp()
        # quantising the scale
        temp.data = torch.floor(temp.data*2**12)/2**12
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * temp
        
        dots = self.asquant(dots) 
        attn = self.attend(dots)
        attn = self.aquant(attn)

        out = torch.matmul(attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = self.dropout(out)

        return out

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0. , quantization: bool = False, quantizer_cfg: DictConfig = None):
        super().__init__()
        ioq = Int8ActPerTensorFixedPoint if quantization else None
        wq = get_weight_quantizer(quantizer_cfg.weight_quantizer) if quantization else None

        self.norm = QuantLayerNorm(dim, weight_quant=Int8WeightPerTensorFixedPoint, bias_quant=Int8BiasPerTensorFixedPointInternalScaling)
        self.fc1 = qnn.QuantLinear(in_features=dim, out_features=hidden_dim, bias=True, 
                                   input_quant=ioq, weight_quant=wq, bias_quant=None, output_quant=None, 
                                   return_quant_tensor=False)
        self.act = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = qnn.QuantLinear(in_features=hidden_dim, out_features=dim, bias=True, 
                                   input_quant=ioq, weight_quant=wq, bias_quant=None, output_quant=ioq,
                                   return_quant_tensor=False)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, dim, SA_SIZE, heads, dim_head, mlp_dim, dropout, quantization, quantizer_cfg, id=0):
        super().__init__()
        self.id = id
        ioq = MyQuant if quantization else None
        self.mha = MHA(dim, SA_SIZE, heads, dim_head, dropout, quantization, quantizer_cfg) 
        self.skipquant = QuantIdentity(act_quant=ioq, return_quant_tensor=False)
        self.ffn = FeedForward(dim, mlp_dim, dropout, quantization, quantizer_cfg)

    def forward(self, x):
        x = self.mha(x) + x
        x = self.skipquant(x)
        x = self.ffn(x) + x
        return x

class Transformer(nn.Module):
    def __init__(self, dim, SA_SIZE, depth, heads, dim_head, mlp_dim, dropout = 0., quantization: bool = False, quantizer_cfg: DictConfig = None):
        super().__init__()
        
        ioq = MyQuant if quantization else None
        self.ioquant = QuantIdentity(act_quant=ioq, return_quant_tensor=False)
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(EncoderLayer(dim, SA_SIZE, heads, dim_head, mlp_dim, dropout, quantization, quantizer_cfg, id=i))

    def forward(self, x):
        for mod in self.layers:
            x = self.ioquant(x)
            x = mod(x)
        x = self.ioquant(x)
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, SA_SIZE, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.,
                quantization: bool = False, quantizer_cfg: DictConfig = None):
        super().__init__()
        self.qcfg = quantizer_cfg
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.to_patch_embedding = SPT(dim = dim, patch_size = patch_size, channels = channels)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, SA_SIZE, depth, heads, dim_head, mlp_dim, dropout, quantization, quantizer_cfg)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        x = self.mlp_head(x)
        return x

    def export(self, batch, path):
        self.eval()
        x, y = batch
        with torch.no_grad():
            x_emb = self.to_patch_embedding(x[0][None,...])
            x_emb += self.pos_embedding
            x_emb = self.transformer.ioquant(x_emb)
            scale = 1./self.transformer.ioquant.act_quant.scale().item()
            print(f"input_t = {scale}, IntBits = {8-np.log2(scale)}")
            x_norm = self.transformer.layers[0].mha.norm(x_emb).numpy()
            x_emb = x_emb.detach().numpy()
        
        layer = 0

        scale = 1./self.transformer.layers[layer].mha.to_qkv.input_quant.scale().item()
        print(f"norm_out_t = {scale}, IntBits = {8-np.log2(scale)}")
        scale = 1./self.transformer.layers[layer].mha.to_qkv.output_quant.scale().item()
        print(f"qkv_t = {scale}, IntBits = {8-np.log2(scale)}")
        scale = 1./self.transformer.layers[layer].mha.asquant.act_quant.scale().item()
        print(f"attn_score_t = {scale}, IntBits = {8-np.log2(scale)}")
        scale = 1./self.transformer.layers[layer].mha.aquant.act_quant.scale().item()
        print(f"attn_prob_t = {scale}, IntBits = {8-np.log2(scale)}")
        scale = 1./self.transformer.layers[layer].mha.to_out.input_quant.scale().item()
        print(f"emb_out_t = {scale}, IntBits = {8-np.log2(scale)}")
        scale = 1./self.transformer.layers[layer].mha.to_out.output_quant.scale().item()
        print(f"hfc_out_t = {scale}, IntBits = {8-np.log2(scale)}")      
        scale = 1./self.transformer.layers[layer].skipquant.act_quant.scale().item()
        print(f"skip_out_t = {scale}, IntBits = {8-np.log2(scale)}") 
        
        weights = {}
        # the norm weights
        nw = weights["norm_weight"] = self.transformer.layers[layer].mha.norm.weight.detach().numpy()
        nb = weights["norm_bias"] = self.transformer.layers[layer].mha.norm.bias.detach().numpy()
        print(nw.min(), nw.max())
        print(nb.min(), nb.max())
        
        # attention weights
        attn_weights = self.transformer.layers[layer].mha.to_qkv.int_weight()
        attn_wei_scale = self.transformer.layers[layer].mha.to_qkv.quant_weight().scale.detach().numpy()

        heads = self.transformer.layers[layer].mha.heads

        qkv = attn_weights.chunk(3, dim = 0)
        q, k, v = map(lambda t: rearrange(t, '(h d) e -> h d e', h = heads).numpy(), qkv)
        
        query_weights, key_weights, value_weights = [], [], []
        PEs, SIMDs = 8, 32
        wprec = self.qcfg.weight_quantizer.bit_width
        for head in range(heads):
            query_weights.append(pack_weights(q[head], PEs, SIMDs, wprec))
            key_weights.append(pack_weights(k[head], PEs, SIMDs, wprec))
            value_weights.append(pack_weights(v[head], PEs, SIMDs, wprec))

        query_weights = np.asarray(query_weights).transpose(1,2,0).reshape(-1)
        key_weights   = weights[f"key_weights"]   = np.asarray(key_weights).transpose(1,2,0)#.reshape(-1)
        value_weights = weights[f"value_weights"] = np.asarray(value_weights).transpose(1,2,0)#.reshape(-1)

        # weights[f"query_weights"] = query_weights[0]
        # weights[f"key_weights"]   = key_weights[0]
        # weights[f"value_weights"] = value_weights[0]

        temp = self.transformer.layers[layer].mha.temperature.exp()
        temp.data = torch.floor(temp.data*2**12)/2**12
        print("tempurature.scale = {}".format(temp.detach().numpy()))

        headfc_weights = self.transformer.layers[layer].mha.to_out.int_weight()
        hfc_wei_scale = self.transformer.layers[layer].mha.to_out.quant_weight().scale.detach().numpy()
        PEs, SIMDs = 8,32
        headfc_weights = pack_weights(headfc_weights, PEs, SIMDs, wprec).reshape(-1)
        weights["qhfc_weights"] = np.concatenate((query_weights, headfc_weights))

        # writing the export
        path += "/vit/"
        if not os.path.exists(path):
            os.mkdir(path)
        np.save(path + "/input.npy", x_emb)
        np.save(path + "/input_norm.npy", x_norm)
        for key in weights.keys():
            np.save(path + key + ".npy", weights[key])

    def debug(self, batch, path):
        path = path + "/../../debug/"
        self.eval()
        x, y = batch
        with torch.no_grad():
            x_emb = self.to_patch_embedding(x[0][None,...])
            x_emb += self.pos_embedding
            x_emb = self.transformer.ioquant(x_emb)
            x_norm = self.transformer.layers[0].mha.norm(x_emb).numpy()

                
        # numpy layer norm
        nw = self.transformer.layers[0].mha.norm.weight.detach().numpy()
        nb = self.transformer.layers[0].mha.norm.bias.detach().numpy()
        print(nw.min(), nw.max())
        print(nb.min(), nb.max())

        array = x_emb[0].detach().numpy()
        means = array.mean(axis=-1)[...,None]
        var = (array-means)**2
        var = var.mean(axis=-1)[...,None]
        stds = np.sqrt(var+1e-5)
        array = (array-means)/stds
        # nw = np.round(nw*2**8)/2**8
        # nb = np.round(nb*2**7)/2**7
        array = array*nw+nb
        diff = x_norm-array
        print(diff.min(), diff.max())

        import matplotlib.pyplot as plt

        plt.plot(x_norm.reshape(-1), array.reshape(-1), 'bo')
        plt.show()
        plt.savefig("/home/zahidu/workspace/brevitas-template/debug/debug.png")