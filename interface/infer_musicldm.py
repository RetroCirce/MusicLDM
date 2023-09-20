'''
This code is released and maintained by:

Ke Chen, Yusong Wu, Haohe Liu
MusicLDM: Enhancing Novelty in Text-to-Music Generation Using Beat-Synchronous Mixup Strategies
All rights reserved

contact: knutchen@ucsd.edu
'''
import sys

sys.path.append("src")

import os

import numpy as np

import argparse
import yaml
import torch

from pytorch_lightning.strategies.ddp import DDPStrategy
from latent_diffusion.models.musicldm import MusicLDM
from utilities.data.dataset import TextDataset

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything


config_path = 'musicldm.yaml'

def main(config, texts, seed):

    seed_everything(seed)
    os.makedirs(config['cache_location'], exist_ok=True)
    os.environ['TRANSFORMERS_CACHE'] = config['cache_location']
    torch.hub.set_dir(config['cache_location'])
    os.makedirs(config['log_directory'], exist_ok=True)
    log_path = os.path.join(config['log_directory'], os.getlogin())
    os.makedirs(log_path, exist_ok=True)
    folder_name = os.listdir(log_path)
    i = 0
    while str(i) in folder_name:
        i = i + 1
    log_path = os.path.join(log_path, str(i))
    os.makedirs(log_path, exist_ok=True)

    batch_size = config["model"]["params"]["batchsize"]

    print(f'Samples with be saved at {log_path}')

    dataset = TextDataset(
        data = texts,
        logfile=os.path.join(log_path, "meta.txt")
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config['model']['num_workers']
    )

    devices = torch.cuda.device_count()

    latent_diffusion = MusicLDM(**config["model"]["params"])
    latent_diffusion.set_log_dir(log_path, log_path, log_path)
    trainer = Trainer(
        max_epochs=1,
        accelerator="gpu",
        devices=devices,
        num_sanity_val_steps=0,
        strategy=DDPStrategy(find_unused_parameters=False)
        if (int(devices) > 1)
        else None,
    )

    trainer.validate(latent_diffusion, loader)
    
    print(f"Generation Finished. Please check the generation samples and the meta file at {log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        type=str,
        required=False,
        help="text to generate the music sample from",
        default=""
    )
    parser.add_argument(
        "--texts",
        type=str,
        required=False,
        help="a path to text file to generate the music samples from",
        default=""
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        help="a generation seed",
        default=0
    )

    args = parser.parse_args()

    if args.text != "" and args.texts != "":
        raise f'********** Error: Only one of text and texts configuration could be given **********'
    
    if args.text != "":
        print(f'********** MusicLDM: generate the music sample from the text:')
        print('----------------------------------------------')
        print(f'{args.text}')
        print('----------------------------------------------')
        texts = [args.text]
    
    if args.texts != "":
        print(f'********** MusicLDM: generate music samples from the text file:')
        print('----------------------------------------------')
        print(f'{args.texts}')
        print('----------------------------------------------')
        texts = np.genfromtxt(args.texts, dtype=str, delimiter="\n")

    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    main(config, texts, args.seed)

