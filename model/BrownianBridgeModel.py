import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from tqdm.autonotebook import tqdm
import numpy as np
from einops import rearrange

from model.utils import extract, default
from model.Unet3D import Unet3D
from model.predictor import Determinisitic

class BrownianBridgeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_config = config.model
        self.model_config = model_config
        # data parameters
        self.in_frames = config.data.dataset_config.in_frames
        self.out_frames = config.data.dataset_config.out_frames
        
        # model hyperparameters
        model_params = model_config.BB.params
        self.mt_type = model_params.mt_type
        self.min_timesteps = model_params.min_timesteps
        self.num_timesteps = model_params.num_timesteps
        self.max_var = model_params.max_var if model_params.__contains__("max_var") else 1
        self.eta = model_params.eta if model_params.__contains__("eta") else 1
        self.skip_sample = model_params.skip_sample
        self.sample_type = model_params.sample_type
        self.sample_step = model_params.sample_step
        self.truncate_step = model_params.truncate_step
        self.steps = None
        self.register_schedule()

        # loss and objective
        self.loss_type = model_params.loss_type
        self.objective = model_params.objective

        # UNet
        self.channels = model_params.UNetParams.channels
        self.condition_key = model_params.UNetParams.condition_key
        self.denoise_fn = Unet3D(**vars(model_params.UNetParams))

        if self.condition_key == "predictor":
            model_config.CondParams.predictor.shape_in = (config.data.dataset_config.in_frames, self.channels)
            model_config.CondParams.predictor.out_frames = config.data.dataset_config.out_frames
            self.cond_stage_model = Determinisitic(**vars(model_config.CondParams.predictor))
            # use pretrained model 
            if model_config.CondParams.pretrained:
                ckt = torch.load(model_config.CondParams.pretrained)
                self.cond_stage_model.load_state_dict(ckt)
            
            if not model_config.CondParams.train:
                for p in self.cond_stage_model.parameters():
                    p.requires_grad = False
            
    def register_schedule(self):
        T = self.num_timesteps
        self.per_frame = True if self.mt_type == "frame" else False
        
        if self.mt_type == "linear":
            m_min, m_max = 0.001, 0.999
            m_t = np.linspace(m_min, m_max, T)
        elif self.mt_type == "sin":
            m_t = np.arange(T) / T
            m_t[0] = 0.0005
            m_t = 0.5 * np.sin(np.pi * (m_t - 0.5)) + 0.5
        elif self.mt_type == "frame":
            m_min, m_max = 0.001, 0.999
            min_step = self.min_timesteps
            max_step = self.num_timesteps
            num_frame = self.out_frames
            self.frame_steps = np.linspace(min_step, max_step, num_frame)
            m_t = np.zeros((T, num_frame))
            m_t = m_t.reshape(T, num_frame)
            for i in range(num_frame): 
                step = int(self.frame_steps[i])
                m = np.linspace(m_min, m_max, int(step))
                m_t[-step:, i] = m 
        else:
            raise NotImplementedError
        m_tminus = np.append(0, m_t[:-1])

        variance_t = 2. * (m_t - m_t ** 2) * self.max_var
        variance_tminus = np.append(0., variance_t[:-1])

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('m_t', to_torch(m_t))
        self.register_buffer('m_tminus', to_torch(m_tminus))
        self.register_buffer('variance_t', to_torch(variance_t))
        
        if self.skip_sample:
            midsteps = torch.arange(self.num_timesteps - 1, 1,
                                    step=-((self.num_timesteps - 1) // (self.sample_step - 2)), dtype=torch.long)
            self.steps = torch.cat((midsteps, torch.Tensor([1, 0]).long()), dim=0)
        else:
            self.steps = torch.arange(self.num_timesteps-1, -1, -1)

    def apply(self, weight_init):
        self.denoise_fn.apply(weight_init)
        return self

    def get_parameters(self):
        return self.denoise_fn.parameters()

    def forward(self, x, y, context=None):      
        pred, context = self.cond_stage_model(y)
        b, f, c, h, w, device = *x.shape, x.device
        x = rearrange(x, 'b t c h w -> b c t h w')
        y = rearrange(y, 'b t c h w -> b c t h w')
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, y, context, t, pred=pred)

    def p_losses(self, x0, y, context, t, noise=None, pred=None):
        b, f, c, h, w = x0.shape
        noise = default(noise, lambda: torch.randn_like(x0))

        x_t, objective = self.q_sample(x0, torch.tile(y[:,:,-1:], [1,1,self.out_frames,1,1]), t, noise)
        objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)

        pred = rearrange(pred, 'b t c h w -> b c t h w')
        if self.loss_type == 'l1':
            recloss = (objective - objective_recon).abs().mean()
            pred_loss = (x0 - pred).abs().mean()
        elif self.loss_type == 'l2':
            recloss = F.mse_loss(objective, objective_recon)
            pred_loss = F.mse_loss(pred, x0)
        else:
            raise NotImplementedError()

        x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon)
        log_dict = {"loss": recloss, "x0_recon": x0_recon}
        return recloss, log_dict, pred_loss

    def q_sample(self, x0, y, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x0))
        m_t = extract(self.m_t, t, x0.shape, self.per_frame)
        var_t = extract(self.variance_t, t, x0.shape, self.per_frame)
        sigma_t = torch.sqrt(var_t)

        objective = m_t * (y - x0) + sigma_t * noise
        return (
            (1. - m_t) * x0 + m_t * y + sigma_t * noise,
            objective
        )

    def predict_x0_from_objective(self, x_t, y, t, objective_recon):
        x0_recon = x_t - objective_recon
        return x0_recon

    @torch.no_grad()
    def q_sample_loop(self, x0, y):
        imgs = [x0]
        for i in tqdm(range(self.num_timesteps), desc='q sampling loop', total=self.num_timesteps):
            t = torch.full((y.shape[0],), i, device=x0.device, dtype=torch.long)
            img, _ = self.q_sample(x0, y, t)
            imgs.append(img)
        return imgs

    @torch.no_grad()
    def p_sample(self, x_t, y, context, i, clip_denoised=False, mix_up=None):
        b, *_, device = *x_t.shape, x_t.device
        if self.steps[i] == 0:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)
            return x0_recon, x0_recon
        else:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            n_t = torch.full((x_t.shape[0],), self.steps[i+1], device=x_t.device, dtype=torch.long)

            m_t = extract(self.m_t, t, x_t.shape, self.per_frame)
            m_nt = extract(self.m_t, n_t, x_t.shape, self.per_frame)
            var_t = extract(self.variance_t, t, x_t.shape, self.per_frame)
            var_nt = extract(self.variance_t, n_t, x_t.shape, self.per_frame)
            sigma2_t = (var_t - var_nt * (1. - m_t) ** 2 / (1. - m_nt) ** 2) * var_nt / var_t
            sigma_t = torch.sqrt(sigma2_t) * self.eta #* torch.tensor([0.05, 0.0575, 0.066125, 0.07604375, 0.087450313, 0.100567859, 0.115653038, 0.133000994, 0.152951143, 0.175893815], device=x_t.device).reshape([1,1,10,1,1])
            noise = torch.randn_like(x_t)

            if i == self.truncate_step:
                x_t = (1. - m_t) * mix_up + m_t * y + sigma_t * noise
                
            objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)

            x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * y + torch.sqrt((var_nt - sigma2_t) / var_t) * \
                            (x_t - (1. - m_t) * x0_recon - m_t * y)
            x_t = torch.where(m_nt == 0, x_t, x_tminus_mean + sigma_t * noise)
            return x_t, x0_recon

    @torch.no_grad()
    def p_sample_loop(self, y, context=None, clip_denoised=True, sample_mid_step=False, mix_up=None):
        img = torch.tile(y[:,:,-1:], [1,1,self.out_frames,1,1])
        y = img
        
        if sample_mid_step:
            imgs, one_step_imgs = [y], []
            for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
                img, x0_recon = self.p_sample(x_t=imgs[-1], y=y, context=context, i=i, clip_denoised=clip_denoised)
                imgs.append(img)
                one_step_imgs.append(x0_recon)
            return imgs, one_step_imgs
        else:
            for i in tqdm(range(self.truncate_step, len(self.steps)), desc=f'sampling loop time step', total=len(self.steps) - self.truncate_step):
                img, _ = self.p_sample(x_t=img, y=y, context=context, i=i, clip_denoised=clip_denoised, mix_up=mix_up)
            return img

    @torch.no_grad()
    def sample(self, y, context=None, clip_denoised=True, sample_mid_step=False):
        pred, context = self.cond_stage_model(y)
        pred = rearrange(pred, 'b t c h w -> b c t h w')
        y = rearrange(y, 'b t c h w -> b c t h w')
        tmpt = self.p_sample_loop(y, context, clip_denoised, sample_mid_step, mix_up=pred)
        tmpt = rearrange(tmpt, 'b c t h w -> (b t) c h w')
        pred = rearrange(pred, 'b c t h w -> (b t) c h w')
        return tmpt, pred