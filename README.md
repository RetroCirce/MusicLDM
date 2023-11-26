# MusicLDM

<p align="center">
  <a href="https://arxiv.org/abs/2308.01546"><img src="https://img.shields.io/badge/arXiv-2308.01546-brightgreen.svg?style=flat-square"/></a>
  <a href="https://musicldm.github.io"><img src="https://img.shields.io/badge/Demo-github.io-orange"/></a>
  <a href="https://huggingface.co/ircam-reach/musicldm"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Diffusers-blue"/></a>
  <a href="https://huggingface.co/spaces/ircam-reach/musicldm-text-to-music"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-API-blue"/></a>
  <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/"><img src="https://img.shields.io/badge/License-BY--NC--SA-purple"/></a>

</p>

MusicLDM is a text-to-music generation model that is only trained on 10000 songs. In the paper, we explore how the text-to-music generation models have the potential behavior to plagarize its training data for the generation output. We propose a latent-audio-mixup method to reduce the plagarism rate and enhance the novelity of the generation.


Generation demos: [musicldm.github.io](https://musicldm.github.io/).

MusicLDM is also supported and embedded in [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/api/pipelines/musicldm) along with the [API](https://huggingface.co/spaces/ircam-reach/musicldm-text-to-music) (**temporarily shut down and will be released again**) to quickly try out the generation! We greatly thank the help from [Sanchit Gandhi](https://huggingface.co/sanchit-gandhi) on the Hugging Face implementation!

In this repo, we also provide the inference code to run the MusicLDM. Due to our research policy, we are not able to provide the training script, but you are free to plugin your own audio dataset loader and your own dataset to establish the training of MusicLDM. 

Currently, MusicLDM is able to generate the music samples of 16 kHz. Benefiting from the progress on vocoders and the expansion of the music datasets, we are able to provide the 44.1 kHz MusicLDM soon in the comming weeks.

### Checklist:

- [x] MusicLDM Inference on 16 kHz music samples
- [x] Hugging Face Support (see [Diffusers](#hugging-face--diffusers))
- [x] 10-sec generation support
- [ ] variable-length generation support (possible already with [Diffusers](#hugging-face--diffusers))
- [ ] 44.1 kHz generation support
- [ ] MusicLDM ckpt trained on large-scale datasets.


## How to Use MusicLDM

### Step 1 : Install Conda:

If you already installed conda before, skip this step, otherwise:
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
``````


### Step 2: Create the environment from the yaml file:

```
conda env create -f musicldm_environment.yml
``` 
Input "yes" or "y" if you face any branch choice.

And this might take a long time to install all dependencies.

### Step 3: Activate the environment:

```
conda activate musicldm_env
```

### Step 4: Run MusicLDM:

**The Checkpoint link is temporarily shut down, we will notify the time when it is released again.**


Locate at the MusicLDM interface:
```
cd interface
```

Check the musicldm.yaml file, replace all configurations comments with "Replace". Among them, the log_directory is the folder you expect the generation output. 

Everytime when you run the interface, you need to be aware that MusicLDM is licensed under the CC BY-NC-SA license.

MusicLDM supports two running methods.

You can generate a 10-sec music sample from a text by:
```
CUDA_VISIBLE_DEVICES=0 python infer_musicldm.py --text "Cute toy factory theme loop"
```

Or you can write a txt file, each line of which contains a sentence for generation. And you would be able to generate each sample for each line by:
```
CUDA_VISIBLE_DEVICES=0 python infer_musicldm.py --texts sample_file.txt
```
Please check out sample_file.txt for example.

"CUDA_VISIBLE_DEVICES" indicates the GPU you want to use, you can use the below command to check the availbility of GPUs in the server:
```
nvidia-smi
```
Then you can indicate the GPU index you want to use for running MusicLDM.

You can also pass a "seed" in the running code, such as:
```
CUDA_VISIBLE_DEVICES=0 python infer_musicldm.py --text "Cute toy factory theme loop" --seed 1423
CUDA_VISIBLE_DEVICES=0 python infer_musicldm.py --texts sample_file.txt --seed 1423
```
When using different seeds, usually you will get different generation samples.

### Step 5: Check out the generation:

Usually, the generation of MusicLDM will be saved in the folder you specified.

You will find your username folder under this path, and you can check the generation. Don't worry about the replacement of the new generation to the old generation. MusicLDM will save them by creating a new subfolder under your username folder. 

## Prompt Guidance

Our generation demos are shown in [musicldm.github.io](https://musicldm.github.io/).

You can "mimic" the prompts we use for generating our demos to get better generations. 

## Acknowledgements

We thank these repos for the references of MusicLDM:

- [CLAP: Contrastive Language-Audio Pretraining](https://github.com/LAION-AI/CLAP)
- [AudioLDM](https://github.com/haoheliu/AudioLDM)
- [HifiGAN](https://github.com/jik876/hifi-gan)

## Citation

If you find this project and the LAION-Audio-630K dataset useful, please cite our paper:
```
@article{musicldm2023,
  title = {MusicLDM: Enhancing Novelty in Text-to-Music Generation Using Beat-Synchronous Mixup Strategies},
  author = {Chen*, Ke and Wu*, Yusong and Liu*, Haohe and Nezhurina, Marianna and Berg-Kirkpatrick, Taylor and Dubnov, Shlomo},
  journal   = {CoRR},
  volume    = {abs/2308.01546},
  year      = {2023},
  eprinttype = {arXiv}
}

@inproceedings{laionclap2023,
  title = {Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation},
  author = {Wu*, Yusong and Chen*, Ke and Zhang*, Tianyu and Hui*, Yuchen and Berg-Kirkpatrick, Taylor and Dubnov, Shlomo},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP},
  year = {2023}
}
```
