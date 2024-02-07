## Brevitas Template for Quantization-Aware Training.

For running the quantization-aware trianing run:

```
python train.py experiment=quantized trainer.gpus=1 logger=tensorboard model=vit_tiny
```

For Evaluation:

```
python test.py experiment=quantized trainer.gpus=[0] model=vit_tiny ckpt_path=<path to model .ckpt> model.net.SA_SIZE=32
```

The corresponding file for quantization parameters is `configs/quantizer/quant.yaml`.


For Export:

```
python test.py experiment=quantized trainer.gpus=0 model=vit_tiny ckpt_path=<path to model .ckpt> model.net.heads=2 export=True
```


## Acknowledgement

This template is taken from [HERE](https://github.com/ashleve/lightning-hydra-template).