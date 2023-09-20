"""
MusicLDM Core
"""
import os

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
from pytorch_lightning.utilities.distributed import rank_zero_only


from latent_diffusion.util import (
    exists,
    default,
    mean_flat,
    count_params,
    instantiate_from_config,
)
from latent_diffusion.modules.ema import LitEma
from latent_diffusion.modules.distributions.distributions import (
    normal_kl,
    DiagonalGaussianDistribution,
)
from latent_encoder.autoencoder import (
    IdentityFirstStage,
    AutoencoderKL,
)
from latent_diffusion.modules.diffusionmodules.util import (
    make_beta_schedule,
    extract_into_tensor,
    noise_like,
)
from latent_diffusion.models.ddim import DDIMSampler
from latent_diffusion.models.plms import PLMSSampler
import soundfile as sf
import os


__conditioning_keys__ = {"concat": "c_concat", "crossattn": "c_crossattn", "adm": "y"}


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2


class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(
        self,
        unet_config,
        timesteps=1000,
        beta_schedule="linear",
        loss_type="l2",
        ckpt_path=None,
        ignore_keys=[],
        load_only_unet=False,
        monitor="val/loss",
        use_ema=True,
        first_stage_key="image",
        latent_t_size=256,
        latent_f_size=16,
        channels=3,
        log_every_t=100,
        clip_denoised=True,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        given_betas=None,
        original_elbo_weight=0.0,
        v_posterior=0.0,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
        l_simple_weight=1.0,
        conditioning_key=None,
        parameterization="eps",  # all assuming fixed variance schedules
        scheduler_config=None,
        use_positional_encodings=False,
        learn_logvar=False,
        logvar_init=0.0,
    ):
        super().__init__()
        assert parameterization in [
            "eps",
            "x0",
        ], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        self.state = None
        print(
            f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode"
        )
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key

        self.latent_t_size = latent_t_size
        self.latent_f_size = latent_f_size

        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(
                ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet
            )

        self.register_schedule(
            given_betas=given_betas,
            beta_schedule=beta_schedule,
            timesteps=timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)
        else:
            self.logvar = nn.Parameter(self.logvar, requires_grad=False)

        self.logger_save_dir = None
        self.logger_project = None
        self.logger_version = None
        self.label_indices_total = None
        # To avoid the system cannot find metric value for checkpoint
        self.metrics_buffer = {
            "val/kullback_leibler_divergence_sigmoid": 15.0,
            "val/kullback_leibler_divergence_softmax": 10.0,
            "val/psnr": 0.0,
            "val/ssim": 0.0,
            "val/inception_score_mean": 1.0,
            "val/inception_score_std": 0.0,
            "val/kernel_inception_distance_mean": 0.0,
            "val/kernel_inception_distance_std": 0.0,
            "val/frechet_inception_distance": 133.0,
            "val/frechet_audio_distance": 32.0,
        }
        self.initial_learning_rate = None
        self.test_data_subset_path = None
        
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

    def set_log_dir(self, save_dir, project, version):
        self.logger_save_dir = save_dir
        self.logger_project = project
        self.logger_version = version

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(
                beta_schedule,
                timesteps,
                linear_start=linear_start,
                linear_end=linear_end,
                cosine_s=cosine_s,
            )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert (
            alphas_cumprod.shape[0] == self.num_timesteps
        ), "alphas have to be defined for each timestep"

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (
            1.0 - alphas_cumprod_prev
        ) / (1.0 - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

        if self.parameterization == "eps":
            lvlb_weights = self.betas**2 / (
                2
                * self.posterior_variance
                * to_torch(alphas)
                * (1 - self.alphas_cumprod)
            )
        elif self.parameterization == "x0":
            lvlb_weights = (
                0.5
                * np.sqrt(torch.Tensor(alphas_cumprod))
                / (2.0 * 1 - torch.Tensor(alphas_cumprod))
            )
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer("lvlb_weights", lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = (
            self.load_state_dict(sd, strict=False)
            if not only_model
            else self.model.load_state_dict(sd, strict=False)
        )
        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
            * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised
        )
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (
            (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1))).contiguous()
        )
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="Sampling t",
            total=self.num_timesteps,
        ):
            img = self.p_sample(
                img,
                torch.full((b,), i, device=device, dtype=torch.long),
                clip_denoised=self.clip_denoised,
            )
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        shape = (batch_size, channels, self.latent_t_size, self.latent_f_size)
        channels = self.channels
        return self.p_sample_loop(shape, return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == "l1":
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == "l2":
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction="none")
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(
                f"Paramterization {self.parameterization} not yet supported"
            )

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = "train" if self.training else "val"

        loss_dict.update({f"{log_prefix}/loss_simple": loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f"{log_prefix}/loss_vlb": loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f"{log_prefix}/loss": loss})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        t = torch.randint(
            0, self.num_timesteps, (x.shape[0],), device=self.device
        ).long()
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        
        if k == 'text':
            return list(batch['text'])
        elif k == 'fname':
            return batch['fname']
        elif k == 'fbank' or k == 'fbank_1' or k == 'fbank_2':
            return batch[k].unsqueeze(1).to(memory_format=torch.contiguous_format).float()
        else:
            return batch[k].to(memory_format=torch.contiguous_format).float()
        
    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        # x = batch[self.first_stage_key].to(memory_format=torch.contiguous_format).float()
        loss, loss_dict = self(x)
        return loss, loss_dict

    def warmup_step(self):
        if self.initial_learning_rate is None:
            self.initial_learning_rate = self.learning_rate

        # Only the first parameter group
        if self.global_step <= 1000:
            if self.global_step == 0:
                print(
                    "Warming up learning rate start with %s"
                    % self.initial_learning_rate
                )
            self.trainer.optimizers[0].param_groups[0]["lr"] = (
                self.global_step / 1000
            ) * self.initial_learning_rate
        else:
            # TODO set learning rate here
            self.trainer.optimizers[0].param_groups[0][
                "lr"
            ] = self.initial_learning_rate


    def training_step(self, batch, batch_idx):

        assert self.training, "training step must be in training stage"
        self.warmup_step()

        if (
            self.state is None
            and len(self.trainer.optimizers[0].state_dict()["state"].keys()) > 0
        ):
            self.state = (
                self.trainer.optimizers[0].state_dict()["state"][0]["exp_avg"].clone()
            )
        elif self.state is not None and batch_idx % 1000 == 0:
            assert (
                torch.sum(
                    torch.abs(
                        self.state
                        - self.trainer.optimizers[0].state_dict()["state"][0]["exp_avg"]
                    )
                )
                > 1e-7
            ), "Optimizer is not working"

        if len(self.metrics_buffer.keys()) > 0:
            for k in self.metrics_buffer.keys():
                self.log(
                    k,
                    self.metrics_buffer[k],
                    prog_bar=False,
                    logger=True,
                    on_step=True,
                    on_epoch=False,
                )
                print(k, self.metrics_buffer[k])
            self.metrics_buffer = {}

        loss, loss_dict = self.shared_step(batch)

        self.log_dict(
            {k: float(v) for k, v in loss_dict.items()},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )

        self.log(
            "global_step",
            float(self.global_step),
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log(
            "lr_abs",
            float(lr),
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        return loss

    def on_validation_epoch_start(self) -> None:
        # Use text as condition during validation
        if self.global_rank == 0:
            if self.model.conditioning_key is not None:
                if self.cond_stage_key_orig == "waveform":
                    self.cond_stage_key = "text"
                    self.cond_stage_model.embed_mode = "text"
        
        return super().on_validation_epoch_start()

    # def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
    #     self.cond_stage_key = "waveform"
    #     self.cond_stage_model.embed_mode = "audio"
    #     return super().on_validation_batch_start(batch, batch_idx, dataloader_idx)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        assert not self.training, "Validation/Test must not be in training stage"
        if self.global_rank == 0:
            name = self.get_validation_folder_name()
            self.generate_sample(
                [batch],
                name=name,
                unconditional_guidance_scale=self.evaluation_params[
                    "unconditional_guidance_scale"
                ],
                ddim_steps=self.evaluation_params["ddim_sampling_steps"],
                n_gen=self.evaluation_params["n_candidates_per_samples"],
            )

            _, loss_dict_no_ema = self.shared_step(batch)
            with self.ema_scope():
                _, loss_dict_ema = self.shared_step(batch)
                loss_dict_ema = {
                    key + "_ema": loss_dict_ema[key] for key in loss_dict_ema
                }
            self.log_dict(
                loss_dict_no_ema,
                prog_bar=False,
                logger=True,
                on_step=False,
                on_epoch=True,
            )
            self.log_dict(
                loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True
            )

    def get_validation_folder_name(self):
        return "val_%s" % (self.global_step)

    def on_validation_epoch_end(self) -> None:
        if self.test_data_subset_path is not None:
            if self.global_rank == 0:
                from audioldm_eval import EvaluationHelper

                print(
                    "Evaluate model output based on the Audiostock test set: %s"
                    % self.test_data_subset_path
                )
                device = torch.device(f"cuda:{0}")
                name = self.get_validation_folder_name()
                waveform_save_path = os.path.join(self.get_log_dir(), name)
                if (
                    os.path.exists(waveform_save_path)
                    and len(os.listdir(waveform_save_path)) > 0
                ):
                    evaluator = EvaluationHelper(16000, device)
                    metrics = evaluator.main(
                        waveform_save_path,
                        self.test_data_subset_path,
                    )

                    self.metrics_buffer = {
                        ("val/" + k): float(v) for k, v in metrics.items()
                    }
                else:
                    print(
                        "The target folder for evaluation does not exist: %s"
                        % waveform_save_path
                    )

        self.cond_stage_key = self.cond_stage_key_orig
        if self.cond_stage_model is not None:
            self.cond_stage_model.embed_mode = self.cond_stage_model.embed_mode_orig

        return super().on_validation_epoch_end()

    def on_train_batch_end(self, *args, **kwargs):
        # Does this affect speed?
        if self.use_ema:
            self.model_ema(self.model)

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, "n b c h w -> b n c h w")
        denoise_grid = rearrange(denoise_grid, "b n c h w -> (b n) c h w")
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), "1 -> b", b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(
                    batch_size=N, return_intermediates=True
                )

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt


class MusicLDM(DDPM):
    """main class"""

    def __init__(
        self,
        first_stage_config,
        cond_stage_config=None,
        num_timesteps_cond=None,
        cond_stage_key="image",
        cond_stage_trainable=False,
        concat_mode=True,
        cond_stage_forward=None,
        conditioning_key=None,
        scale_factor=1.0,
        batchsize=None,
        evaluation_params={},
        scale_by_std=False,
        base_learning_rate=None,
        latent_mixup=0.,
        *args,
        **kwargs,
    ):
        self.learning_rate = base_learning_rate
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        self.evaluation_params = evaluation_params
        assert self.num_timesteps_cond <= kwargs["timesteps"]
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = "concat" if concat_mode else "crossattn"
        if cond_stage_config == "__is_unconditional__":
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])

        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        self.cond_stage_key_orig = cond_stage_key
        self.latent_mixup = latent_mixup

        print(f'Use the Latent MixUP of {self.latent_mixup}')

        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer("scale_factor", torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print("Diffusion model optimizing logvar")
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert "target" in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt

    def make_cond_schedule(
        self,
    ):
        self.cond_ids = torch.full(
            size=(self.num_timesteps,),
            fill_value=self.num_timesteps - 1,
            dtype=torch.long,
        )
        ids = torch.round(
            torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)
        ).long()
        self.cond_ids[: self.num_timesteps_cond] = ids

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        # only for very first batch
        if (
            self.scale_factor == 1
            and self.scale_by_std
            and self.current_epoch == 0
            and self.global_step == 0
            and batch_idx == 0
            and not self.restarted_from_ckpt
        ):
            # assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer("scale_factor", 1.0 / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        super().register_schedule(
            given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s
        )

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        model = instantiate_from_config(config)
        self.cond_stage_model = model
        if model is not None:
            self.cond_stage_model = self.cond_stage_model.to(self.device)

    def _get_denoise_row_from_list(
        self, samples, desc="", force_no_decoder_quantization=False
    ):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(
                self.decode_first_stage(
                    zd.to(self.device), force_not_quantize=force_no_decoder_quantization
                )
            )
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        denoise_grid = rearrange(denoise_row, "n b c h w -> b n c h w")
        denoise_grid = rearrange(denoise_grid, "b n c h w -> (b n) c h w")
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(
                f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented"
            )
        return self.scale_factor * z

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, "encode") and callable(
                self.cond_stage_model.encode
            ):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                if len(c) == 1:
                    c = self.cond_stage_model([c[0], c[0]])
                    c = c[0:1]
                else:
                    c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def meshgrid(self, h, w):
        y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
        x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)

        arr = torch.cat([y, x], dim=-1)
        return arr

    def delta_border(self, h, w):
        """
        :param h: height
        :param w: width
        :return: normalized distance to image border,
         wtith min distance = 0 at border and max dist = 0.5 at image center
        """
        lower_right_corner = torch.tensor([h - 1, w - 1]).view(1, 1, 2)
        arr = self.meshgrid(h, w) / lower_right_corner
        dist_left_up = torch.min(arr, dim=-1, keepdims=True)[0]
        dist_right_down = torch.min(1 - arr, dim=-1, keepdims=True)[0]
        edge_dist = torch.min(
            torch.cat([dist_left_up, dist_right_down], dim=-1), dim=-1
        )[0]
        return edge_dist

    def get_weighting(self, h, w, Ly, Lx, device):
        weighting = self.delta_border(h, w)
        weighting = torch.clip(
            weighting,
            self.split_input_params["clip_min_weight"],
            self.split_input_params["clip_max_weight"],
        )
        weighting = weighting.view(1, h * w, 1).repeat(1, 1, Ly * Lx).to(device)

        if self.split_input_params["tie_braker"]:
            L_weighting = self.delta_border(Ly, Lx)
            L_weighting = torch.clip(
                L_weighting,
                self.split_input_params["clip_min_tie_weight"],
                self.split_input_params["clip_max_tie_weight"],
            )

            L_weighting = L_weighting.view(1, 1, Ly * Lx).to(device)
            weighting = weighting * L_weighting
        return weighting

    def get_fold_unfold(
        self, x, kernel_size, stride, uf=1, df=1
    ):  # todo load once not every time, shorten code
        """
        :param x: img of size (bs, c, h, w)
        :return: n img crops of size (n, bs, c, kernel_size[0], kernel_size[1])
        """
        bs, nc, h, w = x.shape

        # number of crops in image
        Ly = (h - kernel_size[0]) // stride[0] + 1
        Lx = (w - kernel_size[1]) // stride[1] + 1

        if uf == 1 and df == 1:
            fold_params = dict(
                kernel_size=kernel_size, dilation=1, padding=0, stride=stride
            )
            unfold = torch.nn.Unfold(**fold_params)

            fold = torch.nn.Fold(output_size=x.shape[2:], **fold_params)

            weighting = self.get_weighting(
                kernel_size[0], kernel_size[1], Ly, Lx, x.device
            ).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h, w)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))

        elif uf > 1 and df == 1:
            fold_params = dict(
                kernel_size=kernel_size, dilation=1, padding=0, stride=stride
            )
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(
                kernel_size=(kernel_size[0] * uf, kernel_size[0] * uf),
                dilation=1,
                padding=0,
                stride=(stride[0] * uf, stride[1] * uf),
            )
            fold = torch.nn.Fold(
                output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params2
            )

            weighting = self.get_weighting(
                kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx, x.device
            ).to(x.dtype)
            normalization = fold(weighting).view(
                1, 1, h * uf, w * uf
            )  # normalizes the overlap
            weighting = weighting.view(
                (1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx)
            )

        elif df > 1 and uf == 1:
            fold_params = dict(
                kernel_size=kernel_size, dilation=1, padding=0, stride=stride
            )
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(
                kernel_size=(kernel_size[0] // df, kernel_size[0] // df),
                dilation=1,
                padding=0,
                stride=(stride[0] // df, stride[1] // df),
            )
            fold = torch.nn.Fold(
                output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params2
            )

            weighting = self.get_weighting(
                kernel_size[0] // df, kernel_size[1] // df, Ly, Lx, x.device
            ).to(x.dtype)
            normalization = fold(weighting).view(
                1, 1, h // df, w // df
            )  # normalizes the overlap
            weighting = weighting.view(
                (1, 1, kernel_size[0] // df, kernel_size[1] // df, Ly * Lx)
            )

        else:
            raise NotImplementedError

        return fold, unfold, normalization, weighting

    @torch.no_grad()
    def get_input(
        self,
        batch,
        k,
        return_first_stage_encode=True,
        return_first_stage_outputs=False,
        force_c_encode=False,
        cond_key=None,
        return_original_cond=False,
        bs=None
    ):
        if self.training and self.latent_mixup > 0:
            # doing the mixup
            x = super().get_input(batch, k)
            x1 = super().get_input(batch, k + '_1')
            x2 = super().get_input(batch, k + '_2')
            select_idx = torch.where(torch.rand(x.size(0)) < self.latent_mixup)[0]
            x1 = x1[select_idx]
            x2 = x2[select_idx]

            if return_first_stage_encode:
                encoder_posterior = self.encode_first_stage(x1)
                z1 = self.get_first_stage_encoding(encoder_posterior).detach()
                encoder_posterior = self.encode_first_stage(x2)
                z2 = self.get_first_stage_encoding(encoder_posterior).detach()
                encoder_posterior = self.encode_first_stage(x)
                z = self.get_first_stage_encoding(encoder_posterior).detach()
                p = torch.from_numpy(np.random.beta(5,5, x1.size(0)))
                p = p[:,None,None,None].to(self.device)

                nz = p * z1 + (1 - p) * z2
                nz = nz.float()
                z = z.float()
                nx = self.decode_first_stage(nz).detach()
                z[select_idx] = nz
                x[select_idx] = nx
                x.to(self.device)
                z.to(self.device)
            else:
                z = None

            if self.model.conditioning_key is not None:
                if cond_key is None:
                    cond_key = self.cond_stage_key

                if cond_key == 'waveform':
                    xc = super().get_input(batch, cond_key).cpu()
                    nxc = torch.from_numpy(self.mel_spectrogram_to_waveform(nx, save=False)).squeeze(1)
                    if nxc.size(-1) != xc.size(-1):
                        nxc = nxc[:, :int(xc.size(-1) * 0.9)]
                        nxc = torch.nn.functional.pad(nxc, (0, xc.size(-1) - nxc.size(-1)), 'constant', 0.)
                    xc[select_idx] = nxc
                    xc = xc.detach()
                    xc.requires_grad = False
                    xc.to(self.device)

                if not self.cond_stage_trainable or force_c_encode:
                    if isinstance(xc, dict) or isinstance(xc, list):
                        c = self.get_learned_conditioning(xc)
                    else:
                        c = self.get_learned_conditioning(xc.to(self.device))
                else:
                    c = xc
                if bs is not None:
                    c = c[:bs]
            else:
                raise f'Need a condition'
        else:

            x = super().get_input(batch, k)

            if bs is not None:
                x = x[:bs]

            x = x.to(self.device)

            if return_first_stage_encode:
                encoder_posterior = self.encode_first_stage(x)
                z = self.get_first_stage_encoding(encoder_posterior).detach()
            else:
                z = None

            if self.model.conditioning_key is not None:
                if cond_key is None:
                    cond_key = self.cond_stage_key
                if cond_key != self.first_stage_key:
                    # [bs, 1, 527]
                    xc = super().get_input(batch, cond_key)
                    if type(xc) == torch.Tensor:
                        xc = xc.to(self.device)
                else:
                    xc = x

                if not self.cond_stage_trainable or force_c_encode:
                    if isinstance(xc, dict) or isinstance(xc, list):
                        c = self.get_learned_conditioning(xc)
                    else:
                        c = self.get_learned_conditioning(xc.to(self.device))
                else:
                    c = xc
                if bs is not None:
                    c = c[:bs]

            else:
                c = None
                xc = None
                if self.use_positional_encodings:
                    pos_x, pos_y = self.compute_latent_shifts(batch)
                    c = {"pos_x": pos_x, "pos_y": pos_y}
        out = [z, c]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_original_cond:
            out.append(xc)
        return out

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, "b h w c -> b c h w").contiguous()

        z = 1.0 / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(
                    z, ks, stride, uf=uf
                )

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view(
                    (z.shape[0], -1, ks[0], ks[1], z.shape[-1])
                )  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                output_list = [
                    self.first_stage_model.decode(z[:, :, :, :, i])
                    for i in range(z.shape[-1])
                ]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                return self.first_stage_model.decode(z)

        else:
            return self.first_stage_model.decode(z)

    # same as above but without decorator
    def differentiable_decode_first_stage(
        self, z, predict_cids=False, force_not_quantize=False
    ):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, "b h w c -> b c h w").contiguous()

        z = 1.0 / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(
                    z, ks, stride, uf=uf
                )

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view(
                    (z.shape[0], -1, ks[0], ks[1], z.shape[-1])
                )  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                output_list = [
                    self.first_stage_model.decode(z[:, :, :, :, i])
                    for i in range(z.shape[-1])
                ]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                return self.first_stage_model.decode(z)

        else:
            return self.first_stage_model.decode(z)

    def mel_spectrogram_to_waveform(
        self, mel, savepath=".", bs=None, name="outwav", save=True
    ):
        # Mel: [bs, 1, t-steps, fbins]
        if len(mel.size()) == 4:
            mel = mel.squeeze(1)
        mel = mel.permute(0, 2, 1)
        waveform = self.first_stage_model.vocoder(mel)
        waveform = waveform.cpu().detach().numpy()
        if save:
            self.save_waveform(waveform, savepath, name)
        return waveform

    def save_waveform(self, waveform, savepath, name="outwav"):
        for i in range(waveform.shape[0]):
            if type(name) is str:
                path = os.path.join(
                    savepath, "%s_%s_%s.wav" % (self.global_step, i, name)
                )
            elif type(name) is list:
                path = os.path.join(
                    savepath,
                    "%s.wav"
                    % (
                        os.path.basename(name[i])
                        if (not ".wav" in name[i])
                        else os.path.basename(name[i]).split(".")[0]
                    ),
                )
            else:
                raise NotImplementedError
            sf.write(path, waveform[i, 0], samplerate=16000)

    @torch.no_grad()
    def encode_first_stage(self, x):
        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                df = self.split_input_params["vqf"]
                self.split_input_params["original_image_size"] = x.shape[-2:]
                bs, nc, h, w = x.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(
                    x, ks, stride, df=df
                )
                z = unfold(x)  # (bn, nc * prod(**ks), L)
                # Reshape to img shape
                z = z.view(
                    (z.shape[0], -1, ks[0], ks[1], z.shape[-1])
                )  # (bn, nc, ks[0], ks[1], L )

                output_list = [
                    self.first_stage_model.encode(z[:, :, :, :, i])
                    for i in range(z.shape[-1])
                ]

                o = torch.stack(output_list, axis=-1)
                o = o * weighting

                # Reverse reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization
                return decoded

            else:
                return self.first_stage_model.encode(x)
        else:
            return self.first_stage_model.encode(x)

    def shared_step(self, batch, **kwargs):
        x, c = self.get_input(batch, self.first_stage_key)
        loss = self(x, c)
        return loss

    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(
            0, self.num_timesteps, (x.shape[0],), device=self.device
        ).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            # if self.shorten_cond_schedule:  # TODO: drop this option
            #     tc = self.cond_ids[t].to(self.device)
            #     c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        loss = self.p_losses(x, c, t, *args, **kwargs)
        return loss

    def apply_model(self, x_noisy, t, cond, return_ids=False):

        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            if self.model.conditioning_key == "concat":
                key = "c_concat"
            elif self.model.conditioning_key == "crossattn":
                key = "c_crossattn"
            else:
                key = "c_film"

            cond = {key: cond}

        x_recon = self.model(x_noisy, t, **cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = "train" if self.training else "val"

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()
        # print(model_output.size(), target.size())
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f"{prefix}/loss_simple": loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f"{prefix}/loss_gamma": loss.mean()})
            loss_dict.update({"logvar": self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f"{prefix}/loss_vlb": loss_vlb})
        loss += self.original_elbo_weight * loss_vlb
        loss_dict.update({f"{prefix}/loss": loss})

        return loss, loss_dict

    def p_mean_variance(
        self,
        x,
        c,
        t,
        clip_denoised: bool,
        return_codebook_ids=False,
        quantize_denoised=False,
        return_x0=False,
        score_corrector=None,
        corrector_kwargs=None,
    ):
        t_in = t
        model_out = self.apply_model(x, t_in, c, return_ids=return_codebook_ids)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(
                self, model_out, x, t, c, **corrector_kwargs
            )

        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(
        self,
        x,
        c,
        t,
        clip_denoised=False,
        repeat_noise=False,
        return_codebook_ids=False,
        quantize_denoised=False,
        return_x0=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
    ):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(
            x=x,
            c=c,
            t=t,
            clip_denoised=clip_denoised,
            return_codebook_ids=return_codebook_ids,
            quantize_denoised=quantize_denoised,
            return_x0=return_x0,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
        )
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (
            (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1))).contiguous()
        )

        # if return_codebook_ids:
        #     return model_mean + nonzero_mask * (
        #         0.5 * model_log_variance
        #     ).exp() * noise, logits.argmax(dim=1)
        if return_x0:
            return (
                model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise,
                x0,
            )
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def progressive_denoising(
        self,
        cond,
        shape,
        verbose=True,
        callback=None,
        quantize_denoised=False,
        img_callback=None,
        mask=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        batch_size=None,
        x_T=None,
        start_T=None,
        log_every_t=None,
    ):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T
        intermediates = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {
                    key: cond[key][:batch_size]
                    if not isinstance(cond[key], list)
                    else list(map(lambda x: x[:batch_size], cond[key]))
                    for key in cond
                }
            else:
                cond = (
                    [c[:batch_size] for c in cond]
                    if isinstance(cond, list)
                    else cond[:batch_size]
                )

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = (
            tqdm(
                reversed(range(0, timesteps)),
                desc="Progressive Generation",
                total=timesteps,
            )
            if verbose
            else reversed(range(0, timesteps))
        )
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != "hybrid"
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img, x0_partial = self.p_sample(
                img,
                cond,
                ts,
                clip_denoised=self.clip_denoised,
                quantize_denoised=quantize_denoised,
                return_x0=True,
                temperature=temperature[i],
                noise_dropout=noise_dropout,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
            )
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1.0 - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback:
                callback(i)
            if img_callback:
                img_callback(img, i)
        return img, intermediates

    @torch.no_grad()
    def p_sample_loop(
        self,
        cond,
        shape,
        return_intermediates=False,
        x_T=None,
        verbose=True,
        callback=None,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        img_callback=None,
        start_T=None,
        log_every_t=None,
    ):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = (
            tqdm(reversed(range(0, timesteps)), desc="Sampling t", total=timesteps)
            if verbose
            else reversed(range(0, timesteps))
        )

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != "hybrid"
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))
            img = self.p_sample(
                img,
                cond,
                ts,
                clip_denoised=self.clip_denoised,
                quantize_denoised=quantize_denoised,
            )
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1.0 - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback:
                callback(i)
            if img_callback:
                img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(
        self,
        cond,
        batch_size=16,
        return_intermediates=False,
        x_T=None,
        verbose=True,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        shape=None,
        **kwargs,
    ):
        if shape is None:
            shape = (batch_size, self.channels, self.latent_t_size, self.latent_f_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {
                    key: cond[key][:batch_size]
                    if not isinstance(cond[key], list)
                    else list(map(lambda x: x[:batch_size], cond[key]))
                    for key in cond
                }
            else:
                cond = (
                    [c[:batch_size] for c in cond]
                    if isinstance(cond, list)
                    else cond[:batch_size]
                )
        return self.p_sample_loop(
            cond,
            shape,
            return_intermediates=return_intermediates,
            x_T=x_T,
            verbose=verbose,
            timesteps=timesteps,
            quantize_denoised=quantize_denoised,
            mask=mask,
            x0=x0,
            **kwargs,
        )

    @torch.no_grad()
    def sample_log(
        self,
        cond,
        batch_size,
        ddim,
        ddim_steps,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        use_plms=False,
        mask=None,
        **kwargs,
    ):

        if mask is not None:
            shape = (self.channels, mask.size()[-2], mask.size()[-1])
        else:
            shape = (self.channels, self.latent_t_size, self.latent_f_size)

        intermediate = None
        if ddim and not use_plms:
            print("Use ddim sampler")

            ddim_sampler = DDIMSampler(self)
            samples, intermediates = ddim_sampler.sample(
                ddim_steps,
                batch_size,
                shape,
                cond,
                verbose=False,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                mask=mask,
                **kwargs,
            )
        elif use_plms:
            print("Use plms sampler")
            plms_sampler = PLMSSampler(self)
            samples, intermediates = plms_sampler.sample(
                ddim_steps,
                batch_size,
                shape,
                cond,
                verbose=False,
                unconditional_guidance_scale=unconditional_guidance_scale,
                mask=mask,
                unconditional_conditioning=unconditional_conditioning,
                **kwargs,
            )

        else:
            print("Use DDPM sampler")
            samples, intermediates = self.sample(
                cond=cond,
                batch_size=batch_size,
                return_intermediates=True,
                unconditional_guidance_scale=unconditional_guidance_scale,
                mask=mask,
                unconditional_conditioning=unconditional_conditioning,
                **kwargs,
            )

        return samples, intermediate

    @torch.no_grad()
    def generate_long_sample(
        self,
        batchs,
        ddim_steps=200,
        ddim_eta=1.0,
        x_T=None,
        n_gen=1,
        generate_duration=60,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        name="waveform",
        use_plms=False,
        **kwargs,
    ):
        assert n_gen == 1
        assert x_T is None

        try:
            batchs = iter(batchs)
        except TypeError:
            raise ValueError("The first input argument should be an iterable object")

        if use_plms:
            assert ddim_steps is not None

        def find_best_waveform_alignment(waveform1, waveform2, margin=1000):
            # [2, 1, 163872]
            diff = 32768
            best_offset = None
            for i in range(2, margin):
                waveform_distance = np.mean(
                    np.abs(waveform1[..., i:] - waveform2[..., :-i])
                )
                if waveform_distance < diff:
                    best_offset = i
                    diff = waveform_distance
            for i in range(2, margin):
                waveform_distance = np.mean(
                    np.abs(waveform2[..., i:] - waveform1[..., :-i])
                )
                if waveform_distance < diff:
                    best_offset = -i
                    diff = waveform_distance
            return best_offset

        use_ddim = ddim_steps is not None
        waveform_save_path = os.path.join(self.get_log_dir(), name)
        os.makedirs(waveform_save_path, exist_ok=True)
        print("Waveform save path: ", waveform_save_path)

        with self.ema_scope("Plotting"):
            for batch in batchs:
                z, c = self.get_input(
                    batch,
                    self.first_stage_key,
                    return_first_stage_outputs=False,
                    force_c_encode=True,
                    return_original_cond=False,
                    bs=None,
                )
                text = super().get_input(batch, "text")

                # Generate multiple samples
                batch_size = z.shape[0]
                c = torch.cat([c], dim=0)

                if unconditional_guidance_scale != 1.0:
                    unconditional_conditioning = (
                        self.cond_stage_model.get_unconditional_condition(batch_size)
                    )

                fnames = list(super().get_input(batch, "fname"))

                waveform = None
                waveform_segment_length = None
                mel_segment_length = None

                while True:
                    if waveform is None:
                        # [2, 8, 256, 16]
                        samples, _ = self.sample_log(
                            cond=c,
                            batch_size=batch_size,
                            x_T=x_T,
                            ddim=use_ddim,
                            ddim_steps=ddim_steps,
                            eta=ddim_eta,
                            unconditional_guidance_scale=unconditional_guidance_scale,
                            unconditional_conditioning=unconditional_conditioning,
                            use_plms=use_plms,
                        )

                        # [2, 1, 1024, 64]
                        mel = self.decode_first_stage(samples)

                        # [2, 1, 163872] np.array
                        waveform = self.mel_spectrogram_to_waveform(
                            mel,
                            savepath=waveform_save_path,
                            bs=None,
                            name=fnames,
                            save=False,
                        )
                        mel_segment_length = mel.size(-2)
                        waveform_segment_length = waveform.shape[-1]
                    else:
                        _, h, w = samples.shape[0], samples.shape[2], samples.shape[3]

                        mask = torch.ones(batch_size, h, w).to(self.device)
                        mask[:, 3 * (h // 16) :, :] = 0
                        mask = mask[:, None, ...]

                        rolled_sample = torch.roll(samples, shifts=(h // 4), dims=2)

                        samples, _ = self.sample_log(
                            cond=c,
                            batch_size=batch_size,
                            x_T=x_T,
                            ddim=use_ddim,
                            ddim_steps=ddim_steps,
                            eta=ddim_eta,
                            unconditional_guidance_scale=unconditional_guidance_scale,
                            unconditional_conditioning=unconditional_conditioning,
                            mask=mask,
                            use_plms=use_plms,
                            x0=rolled_sample,
                        )

                        # [2, 1, 1024, 64]
                        mel_continuation = self.decode_first_stage(samples)

                        # [2, 1, 163872] np.array
                        waveform_continuation = self.mel_spectrogram_to_waveform(
                            mel_continuation,
                            savepath=waveform_save_path,
                            bs=None,
                            name=fnames,
                            save=False,
                        )

                        margin_waveform = waveform[
                            ..., -(waveform_segment_length // 4) :
                        ]
                        offset = find_best_waveform_alignment(
                            margin_waveform,
                            waveform_continuation[..., : margin_waveform.shape[-1]],
                        )
                        print("Concatenation offset is %s" % offset)
                        waveform = np.concatenate(
                            [
                                waveform[
                                    ..., : -(waveform_segment_length // 4) + 2 * offset
                                ],
                                waveform_continuation,
                            ],
                            axis=-1,
                        )
                        self.save_waveform(waveform, waveform_save_path, name=fnames)
                        if waveform.shape[-1] / 16000 > generate_duration:
                            break

        return waveform_save_path

    @torch.no_grad()
    def generate_sample(
        self,
        batchs,
        ddim_steps=200,
        ddim_eta=1.0,
        x_T=None,
        n_gen=1,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        name="waveform",
        use_plms=False,
        **kwargs,
    ):
        # Generate n_gen times and select the best
        # Batch: audio, text, fnames
        assert x_T is None
        try:
            batchs = iter(batchs)
        except TypeError:
            raise ValueError("The first input argument should be an iterable object")

        if use_plms:
            assert ddim_steps is not None

        use_ddim = ddim_steps is not None
        waveform_save_path = os.path.join(self.get_log_dir(), name)
        os.makedirs(waveform_save_path, exist_ok=True)
        print("Waveform save path: ", waveform_save_path)

        if (
            "audiocaps" in waveform_save_path
            and len(os.listdir(waveform_save_path)) >= 964
        ):
            print("The evaluation has already been done at %s" % waveform_save_path)
            return waveform_save_path

        with self.ema_scope("Plotting"):
            for batch in batchs:
                z, c = self.get_input(
                    batch,
                    self.first_stage_key,
                    return_first_stage_outputs=False,
                    force_c_encode=True,
                    return_original_cond=False,
                    bs=None,
                )
                text = super().get_input(batch, "text")

                # Generate multiple samples
                batch_size = z.shape[0] * n_gen
                if c is not None:
                    c = torch.cat([c] * n_gen, dim=0)
                text = text * n_gen

                if unconditional_guidance_scale != 1.0:
                    unconditional_conditioning = (
                        self.cond_stage_model.get_unconditional_condition(batch_size)
                    )

                fnames = list(super().get_input(batch, "fname"))

                samples, _ = self.sample_log(
                    cond=c,
                    batch_size=batch_size,
                    x_T=x_T,
                    ddim=use_ddim,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=unconditional_conditioning,
                    use_plms=use_plms,
                )
                mel = self.decode_first_stage(samples)

                waveform = self.mel_spectrogram_to_waveform(
                    mel, savepath=waveform_save_path, bs=None, name=fnames, save=False
                )
                waveform = np.nan_to_num(waveform)
                waveform = np.clip(waveform, -1, 1)

                if self.model.conditioning_key is not None:
                    similarity = self.cond_stage_model.cos_similarity(
                        torch.FloatTensor(waveform).squeeze(1), text
                    )

                    best_index = []
                    for i in range(z.shape[0]):
                        candidates = similarity[i :: z.shape[0]]
                        max_index = torch.argmax(candidates).item()
                        best_index.append(i + max_index * z.shape[0])
                        print("Similarity between generated audio and text", similarity)
                        print("Choose the following indexes:", best_index)
                else:
                    best_index = torch.arange(z.shape[0])

                waveform = waveform[best_index]
                
                self.save_waveform(waveform, waveform_save_path, name=fnames)
        return waveform_save_path

    @torch.no_grad()
    def audio_continuation(
        self,
        batchs,
        ddim_steps=200,
        ddim_eta=1.0,
        x_T=None,
        n_gen=1,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        name="waveform",
        use_plms=False,
        **kwargs,
    ):
        assert x_T is None
        try:
            batchs = iter(batchs)
        except TypeError:
            raise ValueError("The first input argument should be an iterable object")

        if use_plms:
            assert ddim_steps is not None

        use_ddim = ddim_steps is not None
        waveform_save_path = os.path.join(self.get_log_dir(), name)
        os.makedirs(waveform_save_path, exist_ok=True)
        print("Waveform save path: ", waveform_save_path)
        with self.ema_scope("Plotting Inpaint"):
            for batch in batchs:
                z, c = self.get_input(
                    batch,
                    self.first_stage_key,
                    return_first_stage_outputs=False,
                    force_c_encode=True,
                    return_original_cond=False,
                    bs=None,
                )
                text = super().get_input(batch, "text")

                # Generate multiple samples
                batch_size = z.shape[0] * n_gen
                c = torch.cat([c] * n_gen, dim=0)
                text = text * n_gen

                if unconditional_guidance_scale != 1.0:
                    unconditional_conditioning = (
                        self.cond_stage_model.get_unconditional_condition(batch_size)
                    )

                fnames = list(super().get_input(batch, "fname"))

                _, h, w = z.shape[0], z.shape[2], z.shape[3]

                mask = torch.ones(batch_size, h * 2, w).to(self.device)
                mask[:, h:, :] = 0
                mask = mask[:, None, ...]

                z = torch.cat([z, torch.zeros_like(z)], dim=2)
                samples, _ = self.sample_log(
                    cond=c,
                    batch_size=batch_size,
                    x_T=x_T,
                    ddim=use_ddim,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=unconditional_conditioning,
                    mask=mask,
                    use_plms=use_plms,
                    x0=torch.cat([z] * n_gen, dim=0),
                )

                mel = self.decode_first_stage(samples)

                waveform = self.mel_spectrogram_to_waveform(
                    mel, savepath=waveform_save_path, bs=None, name=fnames, save=False
                )

                similarity = self.cond_stage_model.cos_similarity(
                    torch.FloatTensor(waveform).squeeze(1), text
                )

                best_index = []
                for i in range(z.shape[0]):
                    candidates = similarity[i :: z.shape[0]]
                    max_index = torch.argmax(candidates).item()
                    best_index.append(i + max_index * z.shape[0])

                waveform = waveform[best_index]

                print("Similarity between generated audio and text", similarity)
                print("Choose the following indexes:", best_index)

                self.save_waveform(waveform, waveform_save_path, name=fnames)

    @torch.no_grad()
    def inpainting(
        self,
        batchs,
        ddim_steps=200,
        ddim_eta=1.0,
        x_T=None,
        n_gen=1,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        name="waveform",
        use_plms=False,
        **kwargs,
    ):
        assert x_T is None
        try:
            batchs = iter(batchs)
        except TypeError:
            raise ValueError("The first input argument should be an iterable object")

        if use_plms:
            assert ddim_steps is not None

        use_ddim = ddim_steps is not None
        waveform_save_path = os.path.join(self.get_log_dir(), name)
        os.makedirs(waveform_save_path, exist_ok=True)
        print("Waveform save path: ", waveform_save_path)
        with self.ema_scope("Plotting Inpaint"):
            for batch in batchs:
                z, c = self.get_input(
                    batch,
                    self.first_stage_key,
                    return_first_stage_outputs=False,
                    force_c_encode=True,
                    return_original_cond=False,
                    bs=None,
                )
                text = super().get_input(batch, "text")

                # Generate multiple samples
                batch_size = z.shape[0] * n_gen
                c = torch.cat([c] * n_gen, dim=0)
                text = text * n_gen

                if unconditional_guidance_scale != 1.0:
                    unconditional_conditioning = (
                        self.cond_stage_model.get_unconditional_condition(batch_size)
                    )

                fnames = list(super().get_input(batch, "fname"))

                _, h, w = z.shape[0], z.shape[2], z.shape[3]

                mask = torch.ones(batch_size, h, w).to(self.device)
                mask[:, h // 4 : 3 * (h // 4), :] = 0
                mask = mask[:, None, ...]

                samples, _ = self.sample_log(
                    cond=c,
                    batch_size=batch_size,
                    x_T=x_T,
                    ddim=use_ddim,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=unconditional_conditioning,
                    mask=mask,
                    use_plms=use_plms,
                    x0=torch.cat([z] * n_gen, dim=0),
                )

                mel = self.decode_first_stage(samples)

                waveform = self.mel_spectrogram_to_waveform(
                    mel, savepath=waveform_save_path, bs=None, name=fnames, save=False
                )

                similarity = self.cond_stage_model.cos_similarity(
                    torch.FloatTensor(waveform).squeeze(1), text
                )

                best_index = []
                for i in range(z.shape[0]):
                    candidates = similarity[i :: z.shape[0]]
                    max_index = torch.argmax(candidates).item()
                    best_index.append(i + max_index * z.shape[0])

                waveform = waveform[best_index]

                print("Similarity between generated audio and text", similarity)
                print("Choose the following indexes:", best_index)

                self.save_waveform(waveform, waveform_save_path, name=fnames)

    @torch.no_grad()
    def inpainting_half(
        self,
        batchs,
        ddim_steps=200,
        ddim_eta=1.0,
        x_T=None,
        n_gen=1,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        name="waveform",
        use_plms=False,
        **kwargs,
    ):
        assert x_T is None
        try:
            batchs = iter(batchs)
        except TypeError:
            raise ValueError("The first input argument should be an iterable object")

        if use_plms:
            assert ddim_steps is not None

        use_ddim = ddim_steps is not None
        waveform_save_path = os.path.join(self.get_log_dir(), name)
        os.makedirs(waveform_save_path, exist_ok=True)
        print("Waveform save path: ", waveform_save_path)
        with self.ema_scope("Plotting Inpaint"):
            for batch in batchs:
                z, c = self.get_input(
                    batch,
                    self.first_stage_key,
                    return_first_stage_outputs=False,
                    force_c_encode=True,
                    return_original_cond=False,
                    bs=None,
                )
                text = super().get_input(batch, "text")

                # Generate multiple samples
                batch_size = z.shape[0] * n_gen
                c = torch.cat([c] * n_gen, dim=0)
                text = text * n_gen

                if unconditional_guidance_scale != 1.0:
                    unconditional_conditioning = (
                        self.cond_stage_model.get_unconditional_condition(batch_size)
                    )

                fnames = list(super().get_input(batch, "fname"))

                _, h, w = z.shape[0], z.shape[2], z.shape[3]

                mask = torch.ones(batch_size, h, w).to(self.device)
                mask[:, int(h * 0.325) :, :] = 0
                mask = mask[:, None, ...]

                samples, _ = self.sample_log(
                    cond=c,
                    batch_size=batch_size,
                    x_T=x_T,
                    ddim=use_ddim,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=unconditional_conditioning,
                    mask=mask,
                    use_plms=use_plms,
                    x0=torch.cat([z] * n_gen, dim=0),
                )

                mel = self.decode_first_stage(samples)

                waveform = self.mel_spectrogram_to_waveform(
                    mel, savepath=waveform_save_path, bs=None, name=fnames, save=False
                )

                similarity = self.cond_stage_model.cos_similarity(
                    torch.FloatTensor(waveform).squeeze(1), text
                )

                best_index = []
                for i in range(z.shape[0]):
                    candidates = similarity[i :: z.shape[0]]
                    max_index = torch.argmax(candidates).item()
                    best_index.append(i + max_index * z.shape[0])

                waveform = waveform[best_index]

                print("Similarity between generated audio and text", similarity)
                print("Choose the following indexes:", best_index)

                self.save_waveform(waveform, waveform_save_path, name=fnames)

    @torch.no_grad()
    def super_resolution(
        self,
        batchs,
        ddim_steps=200,
        ddim_eta=1.0,
        x_T=None,
        n_gen=1,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        name="waveform",
        use_plms=False,
        **kwargs,
    ):
        assert x_T is None
        try:
            batchs = iter(batchs)
        except TypeError:
            raise ValueError("The first input argument should be an iterable object")

        if use_plms:
            assert ddim_steps is not None

        use_ddim = ddim_steps is not None
        waveform_save_path = os.path.join(self.get_log_dir(), name)
        os.makedirs(waveform_save_path, exist_ok=True)
        print("Waveform save path: ", waveform_save_path)
        with self.ema_scope("Plotting Inpaint"):
            for batch in batchs:
                z, c = self.get_input(
                    batch,
                    self.first_stage_key,
                    return_first_stage_outputs=False,
                    force_c_encode=True,
                    return_original_cond=False,
                    bs=None,
                )
                text = super().get_input(batch, "text")

                # Generate multiple samples
                batch_size = z.shape[0] * n_gen
                c = torch.cat([c] * n_gen, dim=0)
                text = text * n_gen

                if unconditional_guidance_scale != 1.0:
                    unconditional_conditioning = (
                        self.cond_stage_model.get_unconditional_condition(batch_size)
                    )

                fnames = list(super().get_input(batch, "fname"))

                _, h, w = z.shape[0], z.shape[2], z.shape[3]

                mask = torch.ones(batch_size, h, w).to(self.device)
                mask[:, :, 3 * (w // 4) :] = 0
                mask = mask[:, None, ...]

                samples, _ = self.sample_log(
                    cond=c,
                    batch_size=batch_size,
                    x_T=x_T,
                    ddim=use_ddim,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=unconditional_conditioning,
                    mask=mask,
                    use_plms=use_plms,
                    x0=torch.cat([z] * n_gen, dim=0),
                )

                mel = self.decode_first_stage(samples)

                waveform = self.mel_spectrogram_to_waveform(
                    mel, savepath=waveform_save_path, bs=None, name=fnames, save=False
                )

                similarity = self.cond_stage_model.cos_similarity(
                    torch.FloatTensor(waveform).squeeze(1), text
                )

                best_index = []
                for i in range(z.shape[0]):
                    candidates = similarity[i :: z.shape[0]]
                    max_index = torch.argmax(candidates).item()
                    best_index.append(i + max_index * z.shape[0])

                waveform = waveform[best_index]

                print("Similarity between generated audio and text", similarity)
                print("Choose the following indexes:", best_index)

                self.save_waveform(waveform, waveform_save_path, name=fnames)


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [
            None,
            "concat",
            "crossattn",
            "hybrid",
            "adm",
            "film",
        ]

    def forward(
        self, x, t, c_concat: list = None, c_crossattn: list = None, c_film: list = None
    ):
        x = x.contiguous()
        t = t.contiguous()

        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == "concat":
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == "crossattn":
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == "hybrid":
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)
        elif (
            self.conditioning_key == "film"
        ):  # The condition is assumed to be a global token, which wil pass through a linear layer and added with the time embedding for the FILM
            cc = c_film[0].squeeze(1)  # only has one token
            out = self.diffusion_model(x, t, y=cc)
        elif self.conditioning_key == "adm":
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out
