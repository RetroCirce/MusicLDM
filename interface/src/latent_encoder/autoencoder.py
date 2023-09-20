import torch
import os

import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
from latent_diffusion.modules.ema import *

from taming.modules.vqvae.quantize import VectorQuantizer as VectorQuantizer
from torch.optim.lr_scheduler import LambdaLR
from latent_diffusion.modules.diffusionmodules.model import Encoder, Decoder
from latent_diffusion.modules.distributions.distributions import (
    DiagonalGaussianDistribution,
)
import soundfile as sf
from latent_diffusion.modules.losses import LPIPSWithDiscriminator

from utilities.model import get_vocoder
from utilities.tools import synth_one_sample
import itertools

class AutoencoderKL(pl.LightningModule):
    def __init__(
        self,
        ddconfig=None,
        lossconfig=None,
        batchsize=None,
        embed_dim=None,
        time_shuffle=1,
        subband=1,
        ckpt_path=None,
        reload_from_ckpt=None,
        ignore_keys=[],
        image_key="fbank",
        colorize_nlabels=None,
        monitor=None,
        base_learning_rate=1e-5,
        config=None,
        mel_num=64
    ):
        super().__init__()

        self.config = config
        self.image_key = image_key

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        self.loss = LPIPSWithDiscriminator(
            disc_start=lossconfig['params']['disc_start'],
            kl_weight=lossconfig['params']['kl_weight'],
            disc_weight=lossconfig['params']['disc_weight'],
            disc_in_channels=lossconfig['params']['disc_in_channels']
        ) 

        self.subband = int(subband)

        if self.subband > 1:
            print("Use subband decomposition %s" % self.subband)

        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.learning_rate = float(base_learning_rate)
        print("Initial learning rate %s" % self.learning_rate)

        self.time_shuffle = time_shuffle
        self.reload_from_ckpt = reload_from_ckpt
        self.reloaded = False
        self.mean, self.std = None, None

        self.feature_cache = None
        self.flag_first_run = True
        self.train_step = 0

        self.logger_save_dir = None
        self.logger_project = None
        self.logger_version = None

        if not self.reloaded and self.reload_from_ckpt is not None:
            print("--> Reload weight of autoencoder from %s" % self.reload_from_ckpt)
            checkpoint = torch.load(self.reload_from_ckpt)
            self.load_state_dict(checkpoint["state_dict"], strict=False)
            # param_names = [n for n, p in self.named_parameters()]
            # for n in param_names:
            #     print(n, "\t", "Loaded" if n in checkpoint["state_dict"] else "Unloaded")
            self.reloaded = True
        else:
            print("Train from scratch")

        if self.image_key == "fbank":
                self.vocoder = get_vocoder(None, "cpu", ddconfig["mel_num"], ddconfig["hifigan_ckpt"])  

    def get_log_dir(self):
        if (
            self.logger_save_dir is None
            and self.logger_project is None
            and self.logger_version is None
        ):
            return os.path.join(
                self.logger.save_dir, self.logger._project, self.logger.version
            )
        else:
            return os.path.join(
                self.logger_save_dir, self.logger_project, self.logger_version
            )

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        # x = self.time_shuffle_operation(x)
        x = self.freq_split_subband(x)
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        # bs, ch, shuffled_timesteps, fbins = dec.size()
        # dec = self.time_unshuffle_operation(dec, bs, int(ch*shuffled_timesteps), fbins)
        dec = self.freq_merge_subband(dec)
        return dec

    def decode_to_waveform(self, dec):
        from utilities.model import vocoder_infer

        if self.image_key == "fbank":
            dec = dec.squeeze(1).permute(0, 2, 1)
            wav_reconstruction = vocoder_infer(dec, self.vocoder)
        elif self.image_key == "stft":
            dec = dec.squeeze(1).permute(0, 2, 1)
            # if(self.train):
            wav_reconstruction = self.wave_decoder(dec)
            # else:
            #     self.wave_decoder
            #     wav_reconstruction = self.wave_decoder(dec)
        return wav_reconstruction

    def visualize_latent(self, input):
        import matplotlib.pyplot as plt

        # for i in range(10):
        #     zero_input = torch.zeros_like(input) - 11.59
        #     zero_input[:,:,i * 16: i * 16 + 16,:16] += 13.59

        #     posterior = self.encode(zero_input)
        #     latent = posterior.sample()
        #     avg_latent = torch.mean(latent, dim=1)[0]
        #     plt.imshow(avg_latent.cpu().detach().numpy().T)
        #     plt.savefig("%s.png" % i)
        #     plt.close()
        np.save("input.npy", input.cpu().detach().numpy())
        # zero_input = torch.zeros_like(input) - 11.59
        time_input = input.clone()
        time_input[:, :, :, :32] *= 0
        time_input[:, :, :, :32] -= 11.59

        np.save("time_input.npy", time_input.cpu().detach().numpy())

        posterior = self.encode(time_input)
        latent = posterior.sample()
        np.save("time_latent.npy", latent.cpu().detach().numpy())
        avg_latent = torch.mean(latent, dim=1)
        for i in range(avg_latent.size(0)):
            plt.imshow(avg_latent[i].cpu().detach().numpy().T)
            plt.savefig("freq_%s.png" % i)
            plt.close()

        freq_input = input.clone()
        freq_input[:, :, :512, :] *= 0
        freq_input[:, :, :512, :] -= 11.59

        np.save("freq_input.npy", freq_input.cpu().detach().numpy())

        posterior = self.encode(freq_input)
        latent = posterior.sample()
        np.save("freq_latent.npy", latent.cpu().detach().numpy())
        avg_latent = torch.mean(latent, dim=1)
        for i in range(avg_latent.size(0)):
            plt.imshow(avg_latent[i].cpu().detach().numpy().T)
            plt.savefig("time_%s.png" % i)
            plt.close()

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        if self.flag_first_run:
            print("Latent size: ", z.size())
            self.flag_first_run = False
            
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch):
        fbank, log_magnitudes_stft, label_indices, fname, waveform, text = batch
        # if(self.time_shuffle != 1):
        #     if(fbank.size(1) % self.time_shuffle != 0):
        #         pad_len = self.time_shuffle - (fbank.size(1) % self.time_shuffle)
        #         fbank = torch.nn.functional.pad(fbank, (0,0,0,pad_len))

        ret = {}

        ret["fbank"], ret["stft"], ret["fname"], ret["waveform"] = (
            fbank.unsqueeze(1),
            log_magnitudes_stft.unsqueeze(1),
            fname,
            waveform.unsqueeze(1),
        )

        return ret

    def freq_split_subband(self, fbank):
        if self.subband == 1 or self.image_key != "stft":
            return fbank

        bs, ch, tstep, fbins = fbank.size()

        assert fbank.size(-1) % self.subband == 0
        assert ch == 1

        return (
            fbank.squeeze(1)
            .reshape(bs, tstep, self.subband, fbins // self.subband)
            .permute(0, 2, 1, 3)
        )

    def freq_merge_subband(self, subband_fbank):
        if self.subband == 1 or self.image_key != "stft":
            return subband_fbank
        assert subband_fbank.size(1) == self.subband  # Channel dimension
        bs, sub_ch, tstep, fbins = subband_fbank.size()
        return subband_fbank.permute(0, 2, 1, 3).reshape(bs, tstep, -1).unsqueeze(1)

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = batch[self.image_key].unsqueeze(1)
        if batch_idx % 5000 == 0 and self.local_rank == 0:
            # print("Log train image")
            self.log_images(inputs)

        reconstructions, posterior = self(inputs)
        waveform=None
        if self.image_key == "stft":
            rec_waveform = self.decode_to_waveform(reconstructions)
        else:
            rec_waveform = None

        if optimizer_idx == 0:
            self.train_step += 1
            self.log(
                "train_step",
                self.train_step,
                prog_bar=False,
                logger=False,
                on_step=True,
                on_epoch=False,
            )
            # train encoder+decoder+logvar

            aeloss, log_dict_ae = self.loss(
                inputs,
                reconstructions,
                posterior,
                waveform,
                rec_waveform,
                optimizer_idx,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="train",
            )
            self.log(
                "aeloss",
                aeloss.mean(),
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            self.log(
                "posterior_std",
                torch.mean(posterior.var),
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=False,
            )
            self.log_dict(
                log_dict_ae, prog_bar=True, logger=True, on_step=True, on_epoch=False
            )
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            # If working on waveform, inputs is STFT, reconstructions are the waveform
            # If working on the melspec, inputs is melspec, reconstruction are also mel spec
            discloss, log_dict_disc = self.loss(
                inputs,
                reconstructions,
                posterior,
                waveform,
                rec_waveform,
                optimizer_idx,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="train",
            )

            self.log(
                "discloss",
                discloss.mean(),
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            self.log_dict(
                log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False
            )
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = batch[self.image_key].unsqueeze(1)

        if batch_idx <= 3:
            self.log_images(inputs, train=False)


        with torch.no_grad():
            reconstructions, posterior = self(inputs)
            aeloss, log_dict_ae = self.loss(
                inputs,
                reconstructions,
                posterior,
                None,
                None,
                0,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="val",
            )

            discloss, log_dict_disc = self.loss(
                inputs,
                reconstructions,
                posterior,
                None,
                None,
                1,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="val",
            )
        self.log(
            "aeloss_val",
            aeloss.mean(),
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True
        )
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def test_step(self, batch, batch_idx):
        inputs = batch[self.image_key]
        fnames = batch["fname"][0]
        mini_batch_size = self.config["model"]["params"]["batchsize"]
        target_length = self.config["preprocessing"]["mel"]["target_length"]

        inputs = list(torch.split(inputs, target_length, dim=1))
        if inputs[-1].size(1) < target_length:
            inputs[-1] = F.pad(inputs[-1], (0,0,0, target_length - inputs[-1].size(1)), 'constant',-20.)

        whole_wav_gt = []
        whole_wav_prediction = []

        for i in range(0, len(inputs), mini_batch_size):
            temp_inputs = torch.cat(inputs[i:i + mini_batch_size], dim=0).unsqueeze(1)
            reconstructions, _ = self(temp_inputs)

            if self.image_key == "stft":
                wav_prediction = self.decode_to_waveform(reconstructions)
                wav_original = None
                self.save_wave(
                    wav_prediction, fnames, os.path.join(save_path, "stft_wav_prediction")
                )
            else:
                for j in range(len(temp_inputs)):
                    input = temp_inputs[j][None,:]
                    reconstruction = reconstructions[j][None,:]
                    wav_vocoder_gt, wav_prediction = synth_one_sample(
                        input.squeeze(1),
                        reconstruction.squeeze(1),
                        labels="validation",
                        vocoder=self.vocoder,
                    )
                    whole_wav_gt.append(wav_vocoder_gt)
                    whole_wav_prediction.append(wav_prediction)
        whole_wav_gt = np.concatenate(whole_wav_gt, axis=-1)
        whole_wav_prediction = np.concatenate(whole_wav_prediction, axis=-1)
        save_path = os.path.join('/home/kechen/research/CTTM/Controllable_TTM/logs/vae-mixup/CTTM-VAE-MIXUP-2023-05-04/2023_05_04_autoencoder_mel_mixup_16_128_4.5e-06_v1_1683453017/generated_files', fnames)
        self.save_wave(
            whole_wav_gt, [save_path + '_hifigan_gt.wav']
        )
        self.save_wave(
            whole_wav_prediction, [save_path + '_recon.wav']
        )

    def save_wave(self, batch_wav, fname):
        for wav, name in zip(batch_wav, fname):
            sf.write(name, wav, samplerate=16000)

    def configure_optimizers(self):
        lr = self.learning_rate
        params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quant_conv.parameters())
            + list(self.post_quant_conv.parameters())
        )

        if self.image_key == "stft":
            params += list(self.wave_decoder.parameters())

        opt_ae = torch.optim.Adam(params, lr=lr, betas=(0.5, 0.9))

        if self.image_key == "fbank":
            disc_params = self.loss.discriminator.parameters()
        elif self.image_key == "stft":
            disc_params = itertools.chain(
                self.loss.msd.parameters(), self.loss.mpd.parameters()
            )

        opt_disc = torch.optim.Adam(disc_params, lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, train=True, only_inputs=False, waveform=None, **kwargs):
        log = dict()
        x = batch.to(self.device)
        if not only_inputs:
            with torch.no_grad():
                xrec, posterior = self(x)
                log["samples"] = self.decode(posterior.sample())
                log["reconstructions"] = xrec

        log["inputs"] = x
        self._log_img(log, train=train, index=0, waveform=waveform)
        # wavs = self._log_img(log, train=train, index=0, waveform=waveform)
        # return wavs

    def _log_img(self, log, train=True, index=0, waveform=None):
        images_input = self.tensor2numpy(log["inputs"][index, 0]).T
        images_reconstruct = self.tensor2numpy(log["reconstructions"][index, 0]).T
        images_samples = self.tensor2numpy(log["samples"][index, 0]).T

        if train:
            name = "train"
        else:
            name = "val"

        if self.logger is not None:
            self.logger.log_image(
                "img_%s" % name,
                [images_input, images_reconstruct, images_samples],
                caption=["input", "reconstruct", "samples"],
            )


    def tensor2numpy(self, tensor):
        return tensor.cpu().detach().numpy()

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
        return x


class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface  # TODO: Should be true by default but check to not break older stuff
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x
