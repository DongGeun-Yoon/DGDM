import os
import lpips

import torch.optim.lr_scheduler
from torch.utils.data import DataLoader

from PIL import Image
from Register import Registers
from model.BrownianBridgeModel import BrownianBridgeModel
from runners.BaseRunner import BaseRunner
from runners.utils import weights_init, get_dataset, make_dir, get_image_grid, save_single_image
from runners.utils import AverageMeter, metric, save_single_video
from runners.utils import *
from tqdm.autonotebook import tqdm
from torchsummary import summary
import torch.nn.functional as F
import time
import imageio
import pdb

@Registers.runners.register_with_name('BBDMRunner')
class BBDMRunner(BaseRunner):
    def __init__(self, config):
        super().__init__(config)

    def initialize_model(self, config):
        if config.model.model_type == "BBDM":
            bbdmnet = BrownianBridgeModel(config)
        else:
            raise NotImplementedError
        
        # initialize model
        try:
            bbdmnet.apply(weights_init)
        except:
            pass
        return bbdmnet

    def load_model_from_checkpoint(self):
        states = super().load_model_from_checkpoint()

    def print_model_summary(self, net):
        def get_parameter_number(model):
            total_num = sum(p.numel() for p in model.parameters())
            trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total_num, trainable_num

        total_num, trainable_num = get_parameter_number(net)
        print("Total Number of parameter: %.2fM" % (total_num / 1e6))
        print("Trainable Number of parameter: %.2fM" % (trainable_num / 1e6))

    def initialize_optimizer_scheduler(self, net, config):
        # diffusion model weight 
        learning_params = [{'params':net.denoise_fn.parameters(), 'lr':config.model.BB.optimizer.lr}]
        # condition model weight
        if config.model.CondParams.train or config.model.CondParams.pretrained is None:
            learning_params.append({'params':net.cond_stage_model.parameters(), 'lr':config.model.CondParams.lr})
            
        optimizer = torch.optim.Adam(learning_params,
                                     weight_decay=config.model.BB.optimizer.weight_decay, 
                                     betas=(config.model.BB.optimizer.beta1, config.model.BB.optimizer.beta2)
                                     )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                               mode='min',
                                                               verbose=True,
                                                               threshold_mode='rel',
                                                               **vars(config.model.BB.lr_scheduler)
                                                               )
        return [optimizer], [scheduler]

    @torch.no_grad()
    def get_checkpoint_states(self, stage='epoch_end'):
        model_states, optimizer_scheduler_states = super().get_checkpoint_states()
        return model_states, optimizer_scheduler_states

    def loss_fn(self, net, batch, epoch, step, opt_idx=0, stage='train', write=True):
        x, x_cond = batch
        loss, additional_info, cond = net(x, x_cond)
        if write:
            self.writer.add_scalar(f'loss/{stage}', loss, step)
            self.writer.add_scalar(f'loss/cond', cond, step)
        loss = loss + cond
        return loss

    @torch.no_grad()
    def sample(self, net, batch, sample_path, stage='train', write=True):
        sample_path = make_dir(os.path.join(sample_path, f'{stage}_sample'))
        
        x, x_cond = batch
        # batch_size = x.shape[0] if x.shape[0] < 4 else 4
        batch_size = 1

        x = x[0:1]
        x_cond = x_cond[0:1]

        grid_size = max(x.size(1), x_cond.size(1))

        # save images
        sample = net.sample(x_cond, clip_denoised=self.config.testing.clip_denoised)
        sample, prediction = sample[0], sample[1]
        
        channels = ['ir105', 'sw038', 'wv063']
        for z, channel in enumerate(channels):
            x_conds = x_cond[0,:, z:z+1]
            x_split = x[0,:, z:z+1]
            sample_split = sample[:, z:z+1]
            prediction_split = prediction[:, z:z+1]           
            
            save_single_video(x_conds, sample_path, f'{channel}_input.png', grid_size, to_normal=self.config.data.dataset_config.to_normal)
            save_single_video(x_split, sample_path, f'{channel}_target.png', grid_size, to_normal=self.config.data.dataset_config.to_normal)
            save_single_video(prediction_split, sample_path, f'{channel}_deter.png', grid_size, to_normal=self.config.data.dataset_config.to_normal)
            save_single_video(sample_split, sample_path, f'{channel}_proba.png', grid_size, to_normal=self.config.data.dataset_config.to_normal)
            
            if stage == 'val':
                target = torch.clamp(((x_split+1)/2), 0, 1).unsqueeze(0).cpu().numpy()
                prediction_split = torch.clamp(((prediction_split+1)/2), 0, 1).unsqueeze(0).cpu().numpy()
                sample_split = torch.clamp(((sample_split+1)/2), 0, 1).unsqueeze(0).cpu().numpy()
                mse_, mae_, ssim_, psnr_ = metric(prediction_split, target, mean=0, std=1, return_ssim_psnr=True)
                mse_2, mae_2, ssim_2, psnr_2 = metric(sample_split, target, mean=0, std=1, return_ssim_psnr=True)
                print(f"=======================================")
                print(f"{channel}_Deterministic MAE : {mae_:.2f}, MSE : {mse_:.2f}, SSIM : {ssim_:.4f}, PSNR : {psnr_:.2f}")
                print(f"{channel}_Probabilistic MAE : {mae_2:.2f}, MSE : {mse_2:.2f}, SSIM : {ssim_2:.4f}, PSNR : {psnr_2:.2f}")
                
                if write:
                    self.writer.add_scalar(f'val_step/{channel}_deter_MSE', mse_, self.global_step)
                    self.writer.add_scalar(f'val_step/{channel}_deter_MAE', mae_, self.global_step)
                    self.writer.add_scalar(f'val_step/{channel}_deter_SSIM', ssim_, self.global_step)
                    self.writer.add_scalar(f'val_step/{channel}_deter_PSNR', psnr_, self.global_step)

                    self.writer.add_scalar(f'val_step/{channel}_prob_MSE', mse_2, self.global_step)
                    self.writer.add_scalar(f'val_step/{channel}_prob_MAE', mae_2, self.global_step)
                    self.writer.add_scalar(f'val_step/{channel}_prob_SSIM', ssim_2, self.global_step)
                    self.writer.add_scalar(f'val_step/{channel}_prob_PSNR', psnr_2, self.global_step)
                
    @torch.no_grad()
    def sample_to_eval(self, net, test_loader, sample_path, save_per_frame=True):
        inputs_path  = make_dir(os.path.join(sample_path, 'input'))
        target_path = make_dir(os.path.join(sample_path, 'target'))
        deter_path  = make_dir(os.path.join(sample_path, 'deter'))
        prob_path   = make_dir(os.path.join(sample_path, 'prob'))
        total_path  = make_dir(os.path.join(sample_path, 'Total'))
        
        if save_per_frame:
            frame_path = sample_path.replace('sample_to_eval', 'sample_frame')
            inputs_frame = make_dir(os.path.join(frame_path, 'input'))
            target_frame = make_dir(os.path.join(frame_path, 'target'))
            deter_frame  = make_dir(os.path.join(frame_path, 'deter'))
            result_frame = make_dir(os.path.join(frame_path, 'prob'))

        pbar = tqdm(test_loader, total=len(test_loader), smoothing=0.01)
        batch_size = self.config.data.test.batch_size
        grid_size = self.config.data.dataset_config.in_frames
        to_normal = self.config.data.dataset_config.to_normal
        sample_num = self.config.testing.sample_num
        
        real_embeddings = [[] for _ in range(3)]
        det_embeddings  = [[] for _ in range(3)]
        fake_embeddings = [[[] for _ in range(sample_num)] for _ in range(3)]
        MAE    = [AverageMeter(), AverageMeter(), AverageMeter()]
        MSE    = [AverageMeter(), AverageMeter(), AverageMeter()]
        PSNR   = [AverageMeter(), AverageMeter(), AverageMeter()]
        SSIM   = [AverageMeter(), AverageMeter(), AverageMeter()]
        LPIPS  = [AverageMeter(), AverageMeter(), AverageMeter()]
        
        MAE2   = [AverageMeter(), AverageMeter(), AverageMeter()]
        MSE2   = [AverageMeter(), AverageMeter(), AverageMeter()]
        PSNR2  = [AverageMeter(), AverageMeter(), AverageMeter()]
        SSIM2  = [AverageMeter(), AverageMeter(), AverageMeter()]
        LPIPS2 = [AverageMeter(), AverageMeter(), AverageMeter()]
        idx = 0
        
        loss_fn = lpips.LPIPS().cuda()
        i3d = load_i3d_pretrained().cuda()
        # FVD 
        def to_i3d(x):
            if x.size(1) == 1:
                x = x.repeat(1, 3, 1, 1) # hack for greyscale images
            x = x.unsqueeze(0).permute(0, 2, 1, 3, 4)  # BTCHW -> BCTHW
            return x
        
        channels = ['ir105', 'sw038', 'wv063']
        for test_batch in pbar:
            if idx >= 1000 and False:
                break
            
            x, x_cond = test_batch
            b, f, c, h, w = x.shape
            for j in range(sample_num): # iteration 
                sample = net.sample(x_cond, clip_denoised=False)
                sample, pred = sample[0].reshape(b, f, c, h, w), sample[1].reshape(b, f, c, h, w)
                
                for i in range(batch_size):
                    input_b = x_cond[i]
                    target_b = x[i]
                    sample_b = sample[i]
                    pred_b = pred[i]
                    
                    for z, channel in enumerate(channels):
                        input_split = input_b[:, z:z+1]
                        target_split = target_b[:, z:z+1]
                        sample_split = sample_b[:, z:z+1]
                        prediction_split = pred_b[:, z:z+1]
                        
                        names = idx + i
                        
                        # save numpy
                        
                        # save frame
                        if save_per_frame:
                            if j == 0: 
                                save_frames(input_split, inputs_frame, f'{names:06}_{channel}_input.png', grid_size, to_normal=to_normal)
                                save_frames(target_split, target_frame, f'{names:06}_{channel}_target.png', grid_size, to_normal=to_normal)
                                save_frames(prediction_split, deter_frame, f'{names:06}_{channel}_deter.png', grid_size, to_normal=to_normal)
                            save_frames(sample_split, result_frame, f'{names:06}_{channel}_{j}_proba.png', grid_size, to_normal=to_normal)
                            
                        # save gif
                        if j == 0:
                            images = [(frames * 127.5 + 127.5)[0].detach().cpu().numpy() for frames in s_split]
                            imageio.mimsave(os.path.join(result_frame, f'{names:06}_{channel}.gif'), images, loop=0)
                            
                            images = [(frames * 127.5 + 127.5)[0].detach().cpu().numpy() for frames in target_split]
                            imageio.mimsave(os.path.join(target_frame, f'{names:06}_{channel}.gif'), images, loop=0)
                            
                            images = [(frames * 127.5 + 127.5)[0].detach().cpu().numpy() for frames in inputs_split]
                            imageio.mimsave(os.path.join(inputs_frame, f'{names:06}_{channel}.gif'), images, loop=0)
                            
                        images = [(frames * 127.5 + 127.5)[0].detach().cpu().numpy() for frames in result_split]
                        imageio.mimsave(os.path.join(result_frame, f'{names:06}_{channel}_{j}.gif'), images, loop=0)

                        # save one png
                        if j == 0:
                            save_single_video(inputs_split, inputs_path, f'{names:06}_{channel}_input.png', grid_size, to_normal=to_normal)
                            save_single_video(target_split, target_path, f'{names:06}_{channel}_target.png', grid_size, to_normal=to_normal)
                            save_single_video(s_split, deter_path, f'{names:06}_{channel}_deter.png', grid_size, to_normal=to_normal)
                        save_single_video(result_split, result_path, f'{names:06}_{channel}_{j}_proba.png', grid_size, to_normal=to_normal)
                        save_single_video(torch.cat([inputs_split, target_split, result_split, s_split], dim=0), total_path, f'{channel}_{names:06}_{j}_total.png', grid_size, to_normal=to_normal)
                        
                        preds = torch.clamp(((result_split+1)/2), 0, 1)
                        vps = torch.clamp(((s_split+1)/2), 0, 1)
                        trues = torch.clamp(((target_split+1)/2), 0, 1)
                        
                        if j == 0:
                            real_embeddings[z].append(get_feats(to_i3d(trues), i3d))
                            det_embeddings[z].append(get_feats(to_i3d(vps), i3d))
                        fake_embeddings[z][j].append(get_feats(to_i3d(preds), i3d))
                        
                        # Diffusion 
                        mse_, mae_, ssim_, psnr_ = metric(preds.unsqueeze(0).cpu().numpy(), trues.unsqueeze(0).cpu().numpy(), mean=0, std=1, return_ssim_psnr=True)
                        lpips_ = loss_fn(preds, trues)
                        lpips_ = lpips_.mean().item()
                        MAE[z].update(mae_,  1)
                        MSE[z].update(mse_,  1)
                        SSIM[z].update(ssim_, 1)
                        PSNR[z].update(psnr_, 1)
                        LPIPS[z].update(lpips_, 1)
                        
                        # Prediction
                        mse_, mae_, ssim_, psnr_ = metric(vps.unsqueeze(0).cpu().numpy(), trues.unsqueeze(0).cpu().numpy(), mean=0, std=1, return_ssim_psnr=True)
                        lpips_ = loss_fn(vps, trues)
                        lpips_ = lpips_.mean().item()
                        MAE2[z].update(mae_,  1)
                        MSE2[z].update(mse_,  1)
                        SSIM2[z].update(ssim_, 1)
                        PSNR2[z].update(psnr_, 1)
                        LPIPS2[z].update(lpips_, 1)
                        
                        # TODO: per frame
                        psnr_per = np.zeros(10)
                        for f_idx in range(grid_size):
                            mse = np.mean((preds.cpu().numpy()[f_idx]-trues.cpu().numpy()[f_idx])**2)
                            psnr_per[f_idx] = - 10 * np.log10(mse)
                        ssim_per = np.zeros(10)
                        
                        for f_idx in range(grid_size):
                            ssim_per[f_idx] = cal_ssim(preds.cpu().numpy()[f_idx].swapaxes(0, 2), trues.cpu().numpy()[f_idx].swapaxes(0, 2), multichannel=True)
                        mse_per = np.sum(np.mean((preds.unsqueeze(0).cpu().numpy() - trues.unsqueeze(0).cpu().numpy())**2, axis=(0)), axis=(1,2,3))
                        mae_per = np.sum(np.mean(np.abs(preds.unsqueeze(0).cpu().numpy() - trues.unsqueeze(0).cpu().numpy()), axis=(0)), axis=(1,2,3))
                        print(" ".join([str(i) for i in mse_per]), " ".join([str(i) for i in mae_per]), " ".join([str(i) for i in psnr_per]), " ".join([str(i) for i in ssim_per]), file=results)
                    
            idx += batch_size
            if idx != 1 and (idx) % 1 == 0:
                for z, channel in enumerate(channels):
                    real_embedding = np.concatenate(real_embeddings[z], axis=0)
                    fake_embedding = [np.concatenate(fake_embeddings[z][i], axis=0) for i in range(sample_num)]
                    det_embedding = np.concatenate(det_embeddings[z], axis=0)
                    print(f"--------[{channel}]--------")
                    print("Probabilistic FVD :", end=' ')
                    for kk in range(sample_num):
                        fvd = compute_fvd(real_embedding, fake_embedding[kk])
                        print(f"{fvd:.4f}", end=' ')
                    print("\nDeterministic FVD : {:.4f}".format(compute_fvd(real_embedding, det_embedding)))
                    print(f"Test [{idx}/{len(test_loader)*b}] MAE : {MAE[z].val:.3f} ({MAE[z].avg:.3f}), MSE : {MSE[z].val:.3f} ({MSE[z].avg:.3f}), SSIM : {SSIM[z].val:.3f} ({SSIM[z].avg:.3f}), PSNR : {PSNR[z].val:.3f} ({PSNR[z].avg:.3f}), LPIPS : {LPIPS[z].val:.3f} ({LPIPS[z].avg:.3f})")
                    print(f"Deterministic, MAE : {MAE2[z].val:.3f} ({MAE2[z].avg:.3f}), MSE : {MSE2[z].val:.3f} ({MSE2[z].avg:.3f}), SSIM : {SSIM2[z].val:.3f} ({SSIM2[z].avg:.3f}), PSNR : {PSNR2[z].val:.3f} ({PSNR2[z].avg:.3f}), LPIPS : {LPIPS2[z].val:.3f} ({LPIPS2[z].avg:.3f})")    
                    print("------------------------")
