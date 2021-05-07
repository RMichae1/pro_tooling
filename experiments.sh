#!/usr/bin/env bash 
 
# RUN LIPASE VAE, Standard architecture w plotting
C:/Users/RCML/Anaconda3/envs/mgpfusion/python.exe //wsl$/Ubuntu/home/rcml/pro_tooling/run_vae_experiment.py --seed 300321 -e 300 --latent_dim 55 --encoder_dim 1700 --decoder_dim 1200 -d 0.065 -lr 0.000027 -wd 0.0007 -t sp400 -sw -p 
# RUN HEXO VAE w Standard architecture plotting
C:/Users/RCML/Anaconda3/envs/mgpfusion/python.exe //wsl$/Ubuntu/home/rcml/pro_tooling/run_vae_experiment.py --seed 300321 -e 300 --latent_dim 55 --encoder_dim 1700 --decoder_dim 1200 -d 0.065 -lr 0.000027 -wd 0.0007 -t hexo -sw -p 