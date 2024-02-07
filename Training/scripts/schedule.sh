#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python train.py experiment=quantized trainer.gpus=[0] test=False logger=tensorboard model=vit.yaml
python test.py experiment=quantized trainer.gpus=[0] model=vit.yaml ckpt_path=./logs/experiments/runs/Quantized_CNV/
python test.py experiment=quantized trainer.gpus=0 model=vit.yaml ckpt_path=./logs/experiments/runs/Quantized_CNV/ export=true
