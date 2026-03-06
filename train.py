"""
Author: David Pukanec (FIT BUT)

This code was produced by David Pukanec (FIT VUT) and was inspired by:
  - DiffComplete: https://github.com/dvlab-research/DiffComplete
  - ControlNet:   https://github.com/lllyasviel/ControlNet
"""


from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_

from logging import getLogger
import hydra
import os
import wandb as wb

from dataset import initialize_data_loader
from model.network import  initialize_control_net, initialize_diff_net
from model.diffusion import initialize_diff_model
from model.diffusion.gaussian_diffusion import get_named_beta_schedule
from model.diffusion.samplers import create_named_schedule_sampler, LossSecondMomentResampler

from utils.solvers import initialize_optimizer, initialize_scheduler
from utils.meters import AverageMeter, Timer
from utils.metrics import *


class ToothCraftTrainer:
    """
      Main Toothcraft trainer. For all the hyperparaneter options please see configs

      Responsibilities:
        - build dataloaders, models, diffusion objects, and optimizer
        - run an iteration-based training loop
        - periodically log (loss + timestep histograms), checkpoint, and validate
        - during validation, sample volumes and compute metrics
      """
    def __init__(self, config):
        self.logger = getLogger("TRAIN")
        self.config = config

        self.device = torch.cuda.current_device()

        # WandB run id is stored so a resumed run can continue logging to the same run.
        self.wb_id = None
        self.run = None

        # Iteration counters (training is controlled by max_iter, not epochs).
        self.curr_iter = 0
        self.epoch = 0

        # Data loaders:
        # - train loader uses repeat=True so it can be iterated indefinitely
        # - val loader is optional (exists only if config has `val`) meaning validation phase
        self.train_loader = initialize_data_loader(config, 'train', repeat=True)
        if hasattr(self.config, 'val'):
            self.val_loader = initialize_data_loader(config, 'val', repeat=False)
        else:
            self.val_loader = None

        # Networks:
        # - diff_model learns the denoising process (diffusion backbone)
        # - control_model injects conditioning (the "hint": incomplete SDF)
        self.diff_model = initialize_diff_net(config).to(self.device)
        self.control_model = initialize_control_net(config).to(self.device)

        encoder_params = sum(p.numel() for p in self.diff_model.parameters() if p.requires_grad)
        decoder_params = sum(p.numel() for p in  self.control_model.parameters() if p.requires_grad)
        print(f"Encoder params: {encoder_params}")
        print(f"Decoder params: {decoder_params}")
        print(self.diff_model)

        # Optional "antag encoder":
        # If enabled, this model is passed into model_kwargs so the diffusion model can use it
        # for additional conditioning experiments.
        if hasattr(self.config.antag, 'type') and self.config.antag.is_used  and self.config.antag.type == 'encoder':
            self.antag_encoder = initialize_control_net(config).to(self.device)
        else:
            self.antag_encoder = None

        # Diffusion schedule:
        # betas define how noise magnitude increases across diffusion timesteps.
        betas = get_named_beta_schedule(config.diffusion.beta_schedule,
                                        config.diffusion.step,
                                        config.diffusion.scale_ratio)

        # Two diffusion objects:
        # - self.diffusion is used for training losses
        # - self.val_diffusion is used for sampling during validation (can be configured separately)
        self.diffusion = initialize_diff_model(betas, config, config.diffusion.model)
        self.val_diffusion = initialize_diff_model(betas, config, config.diffusion.val_model)

        # Sampler
        self.sampler = create_named_schedule_sampler(config.diffusion.sampler, self.diffusion)

        self.optimizer, self.scheduler = self.configure_optimizers(config)

        # Mixed precision training:
        # GradScaler is used to avoid underflow when using fp16 autocast.
        if hasattr(config.train, 'mix_precision') and self.config.train.mix_precision:
            self.scaler = GradScaler()
            self.use_amp = True
        else:
            self.scaler = None
            self.use_amp = False

        # Restore weights/state if configured.
        if config.net.weights:
            print("Loading weights")
            self.load_diff_checkpoint()
        if config.net.control_weights:
            print("Loading control weights")
            self.load_control_checkpoint()

        # WandB init
        if self.config.wandb.key:
            self.init_wandb()

    def init_wandb(self):
        """
        Create/resume a WandB run.

        Key behaviors:
          - if `self.wb_id` exists (e.g., loaded from checkpoint), WandB resumes that run
          - config.wandb.id can force a specific run id (unless 'reset')
        """
        os.environ["WANDB__SERVICE_WAIT"] = "300"
        wb.login(key=self.config.wandb.key, relogin=True, force=True)
        # New run. If loaded from checkpoint wb_id should be initialized
        if self.wb_id is None:
            self.wb_id = wb.util.generate_id()

        if hasattr(self.config.wandb, 'id') and self.config.wandb.id != 'reset':
            self.wb_id = self.config.wandb.id
        elif hasattr(self.config.wandb, 'id'):
            self.wb_id = wb.util.generate_id()

        self.run = wb.init(project=self.config.wandb.project, entity=self.config.wandb.entity,
                           config=dict(self.config),
                           id=self.wb_id,
                           resume="allow")


    def configure_optimizers(self, config):
        """
        Build optimizer and (optional) scheduler.

        All modules are optimized jointly so gradients can flow between:
          - control model
          - diffusion model
          - antag encoder (when present)
        """
        params = list(self.control_model.parameters())
        params += list(self.diff_model.parameters())

        if self.antag_encoder is not None:
            params += list(self.antag_encoder.parameters())

        optimizer = initialize_optimizer(params, config.optimizer)
        if config.optimizer.lr_decay:  # False
            scheduler = initialize_scheduler(optimizer, config.optimizer)
        else:
            scheduler = None
        return optimizer, scheduler

    def load_control_checkpoint(self):
        """
               Load control model weights.

               If antag encoder exists, also loads its weights from `config.antag.weights`.
        """
        weight_path = hydra.utils.to_absolute_path(self.config.net.control_weights)
        save_dict = torch.load(weight_path)
        self.control_model.load_state_dict(save_dict['state_dict'])


        if self.antag_encoder is not None:
            weight_path = hydra.utils.to_absolute_path(self.config.antag.weights)
            save_dict = torch.load(weight_path)
            self.antag_encoder.load_state_dict(save_dict['state_dict'])



    def load_diff_checkpoint(self):
        """
               Load diffusion model + training state.

               Expected fields (depending on what was saved):
                 - state_dict: diffusion model weights
                 - optimizer: optimizer state (momenta, etc.)
                 - scheduler: scheduler state (if enabled)
                 - iteration: training iteration to resume from
                 - wb_id: WandB run id to resume logging into the same run
                 - scaler: AMP scaler state (if AMP was used)
        """
        weight_path = hydra.utils.to_absolute_path(self.config.net.weights)
        save_dict = torch.load(weight_path)

        self.diff_model.load_state_dict(save_dict['state_dict'])
        # Optimizer state is needed to resume training smoothly (keeps momentum/adam moments).
        if "optimizer" in save_dict:
            self.optimizer.load_state_dict(save_dict["optimizer"])

        # Scheduler state is optional.
        if 'scheduler' in save_dict:
            self.scheduler.load_state_dict(save_dict['scheduler'])

        # Restore counters for correct logging/checkpoint cadence after resuming.
        self.curr_iter = int(save_dict.get("iteration", 0))
        self.epoch = int(save_dict.get("epoch", 0))
        self.wb_id = save_dict.get("wb_id", self.wb_id)

        # AMP scaler restoration (only meaningful if AMP is enabled now).
        if hasattr(self.config.train, 'mix_precision') and self.config.train.mix_precision:
            self.scaler.load_state_dict(save_dict["scaler"])

    def save_diff_checkpoint(self, iteration, epoch):
        """
                Save diffusion model checkpoint.

                Includes optimizer (and optional scheduler/scaler) so training can resume exactly.
        """
        os.makedirs('weights', exist_ok=True)
        save_name = f'weights/check_diff_{iteration}.pth'
        state = {
            'wb_id': self.wb_id,
            'iteration': iteration,
            'epoch': epoch,
            'state_dict': self.diff_model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        if self.config.optimizer.lr_decay:
            state['scheduler'] = self.scheduler.state_dict()

        if hasattr(self.config.train, 'mix_precision') and self.config.train.mix_precision:
            state['scaler'] = self.scaler.state_dict()

        torch.save(state, save_name)

    def save_control_checkpoint(self, iteration, epoch):
        """
               Save control model checkpoint.

               Saved separately because control weights may be swapped/loaded independently of diffusion weights.
        """
        os.makedirs('weights', exist_ok=True)
        save_name = f'weights/check_control_{iteration}.pth'
        state = {
            'iteration': iteration,
            'epoch': epoch,
            'state_dict':self.control_model.state_dict(),
        }

        torch.save(state, save_name)

        if self.antag_encoder is not None:
            save_name = f'weights/check_antag_{iteration}.pth'
            state = {
                'iteration': iteration,
                'epoch': epoch,
                'state_dict': self.antag_encoder.state_dict(),
            }

            torch.save(state, save_name)


    def set_seed(self):
        """
        Seed RNGs for reproducible behavior.

        Seed is offset by `curr_iter` so each iteration gets a distinct seed while still being
        deterministic for a given run/iteration number.
        """
        seed = self.config.misc.seed + self.curr_iter
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)


    def collate_model_kwargs(self, b_sample):
        """
               Convert batch dictionary into kwargs consumed by diffusion code.

               Keys:
                 - y: class label tensor (if class conditioning is enabled)
                 - antag: antagonist tensor (if present in dataset sample)
                 - antag_encoder: module reference (when used)
        """
        model_kwargs = {}
        model_kwargs['noise_save_path'] = None
        # Class cond
        if hasattr(self.config, 'class_cond'):
            label = b_sample['label']
            model_kwargs['y'] = torch.tensor(label, device=self.device, dtype=torch.int)
        else:
            model_kwargs['y'] = None
        # Antag conditioning experiments
        if 'antag' in b_sample:
            model_kwargs['antag'] = b_sample['antag'].unsqueeze(1).to(self.device)

        if self.antag_encoder is not None:
            model_kwargs['antag_encoder'] = self.antag_encoder

        return model_kwargs

    def train(self):
        """
        Main training loop.

        For each iteration:
         - sample a batch
         - sample diffusion timesteps `t`
         - compute diffusion loss
         - backward + optimizer step (+ optional AMP)
         - optionally update timestep sampler (LossSecondMomentResampler)
         - periodically: log, checkpoint, validate, empty CUDA cache
        """
        self.diff_model.train()
        self.control_model.train()

        # AverageMeter stores running averages between logging intervals.
        losses = {
            'total_loss': AverageMeter(),
            'mse_loss': AverageMeter()
        }

        accum_t_values = []
        accum_t_weight_values = []

        train_iter = self.train_loader.__iter__()

        self.logger.info(f"---START TRAINING---")

        while self.curr_iter < self.config.train.max_iter:

            self.set_seed()
            self.optimizer.zero_grad(set_to_none=True)
            try:
                b_sample = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                b_sample = next(train_iter)

            gt_sdf = b_sample["gt"].unsqueeze(1).to(self.device, non_blocking=True)
            input_sdf = b_sample["incomplete"].unsqueeze(1).to(self.device, non_blocking=True)

            model_kwargs = self.collate_model_kwargs(b_sample)

            # Sample timestep(s) for the batch and compute the diffusion training loss.
            # `t_weights` optionally reweights losses across timesteps.
            t, t_weights = self.sampler.sample(gt_sdf.size(0), device=self.device)
            with autocast(device_type="cuda", enabled=self.use_amp):

                diffusion_loss = self.diffusion.training_losses(
                    model=self.diff_model,
                    control_model=self.control_model,
                    x_start=gt_sdf,
                    hint=input_sdf,
                    t=t,
                    weighted_loss=self.config.train.weighted_loss,
                    model_kwargs=model_kwargs,
                )

                loss = torch.mean(diffusion_loss["loss"] * t_weights)
            # Backprop: GradScaler is required when using AMP.
            if hasattr(self.config.train, 'mix_precision') and self.config.train.mix_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if isinstance(self.sampler, LossSecondMomentResampler):
                self.sampler.update_with_all_losses(t, diffusion_loss['loss'])

            # Gradient clipping helps avoid exploding gradients (especially for large 3D models).
            if self.config.train.use_gradient_clip:
                clip_grad_norm_(self.diff_model.parameters(), max_norm=self.config.train.gradient_clip_value)
                clip_grad_norm_(self.control_model.parameters(), max_norm=self.config.train.gradient_clip_value)

                if self.antag_encoder is not None:
                    clip_grad_norm_(self.antag_encoder.parameters(),
                                    max_norm=self.config.train.gradient_clip_value)
            # Optimizer step (+ AMP step) and scheduler step.
            if hasattr(self.config.train, 'mix_precision') and self.config.train.mix_precision:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            # Update running loss meter (used for log smoothing).
            losses['mse_loss'].update(loss.item(), gt_sdf.size(0))

            last_lr = self.scheduler.get_last_lr()[0] if self.scheduler else \
            self.optimizer.state_dict()['param_groups'][0]['lr']


            accum_t_values.append(t.detach().cpu().view(-1).numpy())
            accum_t_weight_values.append(t_weights.detach().cpu().view(-1).numpy())

            # -----------------------
            # logging
            # -----------------------

            if self.curr_iter % self.config.train.stat_freq == 0:
                all_t = np.concatenate(accum_t_values, axis=0)
                all_t_weight = np.concatenate(accum_t_weight_values, axis=0)


                wb.log({'train/loss': losses['mse_loss'].avg, 'train/loss_mse': loss.item() ,  "train/last_lr": last_lr,
                        't_hist': wb.Histogram(all_t),
                        't_weight_hist': wb.Histogram(all_t_weight)}, step=self.curr_iter)
                stat_print = f'_____ Iter: {self.curr_iter:6} | Loss: {losses["mse_loss"].avg:.6f} | Lr: {last_lr:.6f}'
                self.logger.info(stat_print)
                losses['mse_loss'].reset()

                # Reset accumulators for next interval
                accum_t_values = []
                accum_t_weight_values = []



            if self.curr_iter % self.config.train.checkpoint_freq == 0:
                self.save_diff_checkpoint(iteration=self.curr_iter, epoch=self.curr_iter)
                self.save_control_checkpoint(iteration=self.curr_iter, epoch=self.curr_iter)

            if (self.curr_iter + 1) % self.config.val.frequency == 0:
                self.logger.info(f"---VALIDATION--- | {self.curr_iter:6}")
                self.validate()
                self.diff_model.train()
                self.control_model.train()

            if self.curr_iter  % self.config.train.empty_cache_freq == 0:
                # Clear cache
                torch.cuda.empty_cache()

            self.curr_iter += 1



    def validate(self):

        """
        Validation loop.

        Generates SDF reconstructions conditioned on `hint` and aggregates metrics across samples.
        """
        val_step = 0
        val_iter = self.val_loader.__iter__()
        avg_meters = {}

        while val_step < self.config.val.max_val_samples:
            with torch.no_grad():
                metrics_timer = Timer()
                b_sample = next(val_iter)

                gt_sdf = b_sample['gt'].unsqueeze(1).to(self.device)
                input_sdf = b_sample['incomplete'].unsqueeze(1).to(self.device)
                bs = input_sdf.size(0)
                noise = None


                model_kwargs = self.collate_model_kwargs(b_sample)
                model_kwargs['hint'] = input_sdf

                # Choose sampling method:
                # - DDIM: faster, controlled by `eta`
                # - p_sample_loop: ancestral sampling (stochastic)
                with autocast(device_type="cuda", enabled=self.use_amp):

                    if self.config.val.use_ddim:
                        gen_sdf = self.val_diffusion.ddim_sample_loop(
                            model=self.diff_model,
                            control_model=self.control_model,
                            shape=[bs, 1] + [self.config.exp.res] * 3,
                            device=self.device,
                            clip_denoised=False,
                            progress=True,
                            eta=self.config.val.eta,
                            model_kwargs=model_kwargs,
                        )
                    else:
                        gen_sdf = self.val_diffusion.p_sample_loop(
                            model=self.diff_model,
                            control_model=self.control_model,
                            shape=[bs, 1] + [self.config.exp.res] * 3,
                            device=self.device,
                            clip_denoised=False,
                            progress=True,
                            noise=noise,
                            model_kwargs=model_kwargs,
                        )

                gen_sdf = gen_sdf.detach()
                metrics_timer.tic()
                self.compute_metrics(gt_sdf, input_sdf, gen_sdf, avg_meters, bounds=b_sample['bounds'])
                self.logger.info(f"Metrics computed in {metrics_timer.toc():4.3f} seconds")

                val_step += bs

        # Log aggregated metrics with prefixes indicating metric category.
        log_dict = {}
        for key, value in avg_meters.items():
            if key.split('_')[0] == 'm':
                log_dict[f'masked/{key}'] = value.avg
            elif key.split('_')[0] == 'p':
                log_dict[f'precise/{key}'] = value.avg
            else:
                log_dict[f'val/{key}'] = value.avg
        wb.log(log_dict)
        self.logger.info(f"Validation Meters: {log_dict}")


    def compute_metrics(self, gts, inputs, preds, avg_meters, antag=None, bounds=None):
        """
         Compute reconstruction metrics for each predicted sample.

         - Converts tensors to numpy for metric helpers.
         - Clips predictions to [-1, 1] for stability/consistency.
         - Uses `bounds` when config.data.bounds is enabled to evaluate only a region of interest.
         """
        gts = gts.cpu().numpy()
        inputs = inputs.cpu().numpy()
        preds = preds.cpu().numpy()
        if antag is not None:
            antag = antag.cpu().numpy()

        for i, samp in enumerate(preds):
            # Clamp the values for better computation
            samp = np.clip(samp[0], -1, 1)
            gt = gts[i]
            input = inputs[i][0]
            if self.config.data.bounds:
                bound = bounds[i].cpu().numpy()
            else:
                bound = None
            # Compute L1
            compute_L1(gt, input, samp, avg_meters, bound)

            # Compute CD
            compute_CD(gt, input, samp, avg_meters, bound)

            # Compute IOU
            if antag is not None:
                an = antag[i]
                compute_IoU(gt, input, samp, avg_meters, bound, antag=an)
            else:
                compute_IoU(gt, input, samp, avg_meters, bound)


@hydra.main(config_path='configs/train', config_name='train_debug')
def main(config):
    trainer = ToothCraftTrainer(config)
    trainer.train()

if __name__ == '__main__':
    main()