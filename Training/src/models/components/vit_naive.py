from black import out
from turtle import forward
import numpy as np

from torch import nn
import torch.nn.functional as F

from brevitas import nn as qnn
from brevitas.nn import QuantIdentity
from brevitas.quant import Int8ActPerTensorFixedPoint
from brevitas.quant.solver import WeightQuantSolver, ActQuantSolver

import hydra
from omegaconf import OmegaConf, DictConfig

from .quantizers import *
from src.utils.tensor_norm import TensorNorm

SA_SIZE = 32
NUM_TOKENS = 256

class PatchEmbed(nn.Module):
  def __init__(self, in_channels, embed_dim):
    super().__init__()
    self.proj = nn.Sequential(
      nn.Conv2d(3, embed_dim, kernel_size=2, stride=2),
      nn.Flatten(2)
      )
    self.norm = nn.LayerNorm((NUM_TOKENS, embed_dim))
  
  def forward(self, x):
    inter = self.proj(x).transpose(1, 2)
    return self.norm(inter)

class Attention(nn.Module): 
    def __init__(self, in_dim, dim_head, quantization=False, quantizer_cfg=None, heads=1):
        super().__init__()
        self.in_dim = in_dim
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        wq = get_weight_quantizer(quantizer_cfg.weight_quantizer) if quantization else None
        
        # self.acc_quant = QuantIdentity(act_quant=Int8ActPerTensorFixedPoint, return_quant_tensor=False)
        self.sm_quant  = QuantIdentity(act_quant=Int8ActPerTensorFixedPoint, return_quant_tensor=False)
        # self.out_quant = QuantIdentity(act_quant=Int8ActPerTensorFixedPoint, return_quant_tensor=False)
        
        self.to_qkv = qnn.QuantLinear(in_features=in_dim, out_features=dim_head*3, bias=False,
                            input_quant=Int8ActPerTensorFixedPoint, weight_quant=wq, 
                            bias_quant=None, output_quant=Int8ActPerTensorFixedPoint)
        self.softmax = nn.Softmax(dim=-1)
        
        self.to_out =  nn.Sequential(
            nn.Linear(dim_head*heads, in_dim),
        )  if not (heads == 1 and dim_head == in_dim) else nn.Identity()

        self.mask = torch.ones((1, NUM_TOKENS,NUM_TOKENS), requires_grad=False).cuda()
        for i in range(NUM_TOKENS):
            for j in range(NUM_TOKENS):
                if(i>=j+SA_SIZE):
                    self.mask[:,i,j] = 0.
                if(j>=i+SA_SIZE):
                    self.mask[:,i,j] = 0.

    def forward(self, x):
        qkv = self.to_qkv(x)
        q, k, v = qkv[:,:,:self.dim_head], qkv[:,:,self.dim_head: 2*self.dim_head], qkv[:,:,2*self.dim_head:]
        attn = torch.bmm(q, k.transpose(1,2))*self.scale
        attn = torch.where(self.mask == 0., torch.Tensor([-torch.inf]).type(attn.type()), attn)
        # attn = self.acc_quant(attn)
        attn_scores = self.softmax(attn)
        attn_scores = self.sm_quant(attn_scores)
        # print(attn_scores)
        # one = attn_scores
        # np.savetxt("/home/uzahid/workspace/brevitas-template/debug/pt_attn_32.txt", one.reshape(-1), fmt='%.8f')
        # exit(0)
        out = torch.bmm(attn_scores, v)
        out = self.to_out(out)
        # out = self.out_quant(out)
        return out

class EncoderLayer(nn.Module):
    def __init__(self, emb_dim, quantization=False, quantizer_cfg=None, dropout=0, eps=1e-4, norm_first=True):
        super().__init__()
        self.norm_first = norm_first
        
        self.norm1 = nn.LayerNorm((NUM_TOKENS, emb_dim), eps=eps, elementwise_affine=True)
        self.norm2 = nn.LayerNorm((NUM_TOKENS, emb_dim), eps=eps, elementwise_affine=True)
        self.ioquant = QuantIdentity(act_quant=Int8ActPerTensorFixedPoint)

        self.mha = Attention(emb_dim, emb_dim, quantization=quantization, quantizer_cfg=quantizer_cfg)

        wq = get_weight_quantizer(quantizer_cfg.weight_quantizer) if quantization else None
        aq = get_activation_quantizer(quantizer_cfg.activation_quantizer) if quantization else None
        
        self.linear1 = qnn.QuantLinear(in_features=emb_dim, out_features=2*emb_dim, bias=False,
                            input_quant=Int8ActPerTensorFixedPoint, weight_quant=wq, bias_quant=None, output_quant=None)
        self.linear2 = qnn.QuantLinear(in_features=2*emb_dim, out_features=emb_dim, bias=False,
                            input_quant=None, weight_quant=wq, bias_quant=None, output_quant=None)
        self.activation = qnn.QuantReLU(act_quant=aq) if aq is not None else nn.ReLU(inplace=True) 
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        x = x + self.mha(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return self.ioquant(x)
    
    def ffn(self, x):
        x = self.linear1(x)
        x = self.linear2(self.dropout(self.activation(x)))
        return self.dropout2(x)        


class SimpleViT(nn.Module):
    def __init__(
        self,
        input_size: int = 3,
        emb_dim: int = 64,
        output_size: int = 10,
        dropout: float = 0.2,
        quantization: bool = False,
        quantizer_cfg: DictConfig = None
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.patch_emb = PatchEmbed(input_size, emb_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, NUM_TOKENS, emb_dim))
        self.ioquant = QuantIdentity(act_quant=Int8ActPerTensorFixedPoint, return_quant_tensor=False)
        self.encoder = nn.Sequential(
                EncoderLayer(emb_dim, dropout=dropout, quantization=quantization, quantizer_cfg=quantizer_cfg),
                EncoderLayer(emb_dim, dropout=dropout, quantization=quantization, quantizer_cfg=quantizer_cfg),
                EncoderLayer(emb_dim, dropout=dropout, quantization=quantization, quantizer_cfg=quantizer_cfg),
            )
        self.output = nn.Linear(emb_dim, output_size, bias=False)


    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x_emb = self.patch_emb(x)
        x_emb += self.pos_embed
        x_emb = self.ioquant(x_emb)

        # ----- possibly on hardware -----
        y = self.encoder(x_emb)
        # --------------------------------

        y = self.output(y[:,0,:])
        return y 

    def export(self, batch, path):
        self.eval()
        x, y = batch
        with torch.no_grad():
            x_emb = self.patch_emb(x[0][None,...])
            x_emb += self.pos_embed
            x_emb = self.ioquant(x_emb)
            inter = self.encoder[0](x_emb)

        # norm1_w = self.encoder[0].norm1.weight.detach().numpy()
        # norm1_b = self.encoder[0].norm1.bias.detach().numpy()
        # mha_wei = self.encoder[0].mha.to_qkv.int_weight().numpy()
        # mha_sc  = self.encoder[0].mha.to_qkv.quant_weight().scale.detach().numpy()

        # qkv = [mha_wei[:self.emb_dim, :], mha_wei[self.emb_dim: 2*self.emb_dim,:], mha_wei[2*self.emb_dim:,:]]
        # qkv_sc = [mha_sc[:self.emb_dim, :], mha_sc[self.emb_dim: 2*self.emb_dim,:], mha_sc[2*self.emb_dim:,:]]

        # # for finn
        # dic = {}
        # i=0
        # for layer in range(3):
        #     w = qkv[layer]
        #     sc = qkv_sc[layer]
        #     w = w.T
        #     dic["arr_"+str(i)] = w.astype(np.float64)
        #     i += 1
        #     dic["arr_"+str(i)] = sc.astype(np.float64)
        #     i += 1
        # np.savez("/home/uzahid/workspace/brevitas-template/Training/export/best.npz", **dic)          

        # np.save(path+"/input.npy", one)        
        # np.save(path+"/norm1_weights.npy", norm1_w)
        # np.save(path+"/norm1_bias.npy", norm1_b)
        # np.save(path+"/query_weights.npy", qkv[0])        
        # np.savetxt(path+"/key_weights.txt", qkv[1].reshape(-1), fmt='%.8f')
        # np.savetxt(path+"/value_weights.txt", qkv[2].reshape(-1), fmt='%.8f')






