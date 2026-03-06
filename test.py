"""
ToothCraft Testing Script

Author: David Pukanec (FIT BUT)

This script evaluates a trained ToothCraft diffusion model on the test set.
It generates reconstructed SDF volumes and computes quantitative metrics.

Inspired by:
    - DiffComplete (https://github.com/dvlab-research/DiffComplete)
    - ControlNet   (https://github.com/lllyasviel/ControlNet)

Major responsibilities:
    • load trained networks
    • run conditional diffusion sampling
    • compute evaluation metrics
    • log results to WandB
    • optionally save generated volumes
"""

import os
from logging import getLogger

import hydra
import torch
import numpy as np
import wandb as wb

from dataset import initialize_data_loader
from model.network import initialize_control_net, initialize_diff_net
from model.diffusion import initialize_diff_model
from model.diffusion.gaussian_diffusion import get_named_beta_schedule
from utils.metrics import *
from utils.meters import AverageMeter

class ToothCraftTester:
    """
    ToothCraft evaluation pipeline.

    This class loads trained networks and evaluates them on the test dataset.
    For every sample it performs conditional diffusion sampling and computes
    reconstruction metrics.
    """
    def __init__(self, config):

        self.logger = getLogger("TEST")
        self.config = config
        self.device = torch.cuda.current_device()

        # WandB state
        self.wb_id = None
        self.run = None

        # Iteration counters (mainly used for seeding)
        self.curr_iter = 0
        self.epoch = 0

        # Initialize data loaders
        self.test_loader = initialize_data_loader(config, 'test', repeat=False)

        # Initialize network.
        self.diff_model = initialize_diff_net(config).to(self.device)
        self.control_model = initialize_control_net(config).to(self.device)

        if hasattr(self.config.antag, 'type') and self.config.antag.is_used and self.config.antag.type == 'encoder':
            self.antag_encoder = initialize_control_net(config).to(self.device)
        else:
            self.antag_encoder = None

        # Initialize Diffusion
        betas = get_named_beta_schedule(
            config.diffusion.beta_schedule,
            config.diffusion.step,
            config.diffusion.scale_ratio,
        )

        self.test_diffusion = initialize_diff_model(betas, config, config.diffusion.model)

        # Load weights if specified
        if config.net.weights:
            self.load_diff_checkpoint()
        if config.net.control_weights:
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
        self.wb_id = wb.util.generate_id()
        self.run = wb.init(project=self.config.wandb.project, entity=self.config.wandb.entity,
                           config=dict(self.config),
                           id=self.wb_id,
                           resume="allow")


    def load_control_checkpoint(self):
        weight_path = hydra.utils.to_absolute_path(self.config.net.control_weights)
        save_dict = torch.load(weight_path)
        self.control_model.load_state_dict(save_dict['state_dict'])

        if self.antag_encoder is not None:
            weight_path = hydra.utils.to_absolute_path(self.config.antag.weights)
            save_dict = torch.load(weight_path)
            self.antag_encoder.load_state_dict(save_dict['state_dict'])

    def load_diff_checkpoint(self):
        weight_path = hydra.utils.to_absolute_path(self.config.net.weights)
        save_dict = torch.load(weight_path)
        self.diff_model.load_state_dict(save_dict['state_dict'])


    def set_seed(self):
        seed = self.config.misc.seed + self.curr_iter
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)


    def log_fdi_meters(self, fdi_meters):
        tables = {'premolar_col': [], 'premolar_data': [],
                  'molar_col': [], 'molar_data': [],
                  'canine_col': [], 'canine_data': [],
                  'incisor_col': [], 'incisor_data': [],}

        for k, v in fdi_meters.items():
            tooth_type = k.split('_')[0]
            tables[f'{tooth_type}_col'].append(f'abs_{k}')
            tables[f'{tooth_type}_data'].append(v.avg)
            tables[f'{tooth_type}_col'].append(f'std_{k}')
            tables[f'{tooth_type}_data'].append(v.get_std())

        log_dict = {"premolar": wb.Table(columns=tables[f'premolar_col'], data=[tables[f'premolar_data']]),
                    'molar': wb.Table(columns=tables[f'molar_col'], data=[tables[f'molar_data']]),
                    'canine': wb.Table(columns=tables[f'canine_col'], data=[tables[f'canine_data']]),
                    'incisor': wb.Table(columns=tables['incisor_col'], data=[tables[f'incisor_data']]),}
        wb.log(log_dict)

    def log_tables(self, avg_meters):
        tables = {'masked_col': [], 'masked_data': [],
                  'precise_col': [], 'precise_data': [],
                  'val_col': [], 'val_data': []}
        for k, v in avg_meters.items():
            if k.split('_')[0] == 'm':
                tables[f'masked_col'].append(f'abs_{k}')
                tables[f'masked_data'].append(v.avg)
                tables[f'masked_col'].append(f'std_{k}')
                tables[f'masked_data'].append(v.get_std())
            elif k.split('_')[0] == 'p':
                tables[f'precise_col'].append(f'abs_{k}')
                tables[f'precise_data'].append(v.avg)
                tables[f'precise_col'].append(f'std_{k}')
                tables[f'precise_data'].append(v.get_std())
            else:
                tables[f'val_col'].append(f'abs_{k}')
                tables[f'val_data'].append(v.avg)
                tables[f'val_col'].append(f'std_{k}')
                tables[f'val_data'].append(v.get_std())

        log_dict = {"val_table": wb.Table(columns=tables[f'val_col'], data=[tables[f'val_data']])}
        if len(tables['precise_col']) > 0:
            log_dict["precision_table"] = wb.Table(columns=tables[f'precise_col'], data=[tables[f'precise_data']])
        if len(tables['masked_col']) > 0:
            log_dict["masked_table"] = wb.Table(columns=tables[f'masked_col'], data=[tables[f'masked_data']])
        wb.log(log_dict)

    def log_avg_meters(self, avg_meters):
        log_dict = {}
        for k, v in avg_meters.items():
            if k.split('_')[0] == 'm':
                log_dict[f'masked/{k}'] = v.val
            elif k.split('_')[0] == 'p':
                log_dict[f'precise/{k}'] = v.val
            else:
                log_dict[f'val/{k}'] = v.val

        wb.log(log_dict)

    def collate_model_kwargs(self, b_sample):
        model_kwargs = {}
        model_kwargs['noise_save_path'] = None
        # Class cond
        if hasattr(self.config, 'class_cond'):
            model_kwargs['y'] = torch.tensor(b_sample['label'], device=self.device, dtype=torch.int)
        else:
            model_kwargs['y'] = None

        # Antag conditioning experiments
        if 'antag' in b_sample:
            model_kwargs['antag'] = b_sample['antag'].unsqueeze(1).to(self.device)
        else:
            model_kwargs['antag'] = None

        if self.antag_encoder is not None:
            model_kwargs['antag_encoder'] = self.antag_encoder

        return model_kwargs

    def test(self):
        """Run inference and compute evaluation metrics."""

        self.logger.info("--- START TESTING ---")

        self.set_seed()

        max_samples = getattr(self.config.exp, "max_samples", 6000)

        avg_meters = {}
        per_fdi_meters = {}

        avg_meters = {}
        per_fdi_meters = {}
        c_iter = 0
        for idx, b_sample in enumerate(self.test_loader):
            with torch.no_grad():
                gt_sdf = b_sample["gt"].unsqueeze(1).to(self.device)
                incomplete = b_sample["incomplete"].unsqueeze(1).to(self.device)
                bs = incomplete.size(0)
                labels = b_sample["label"]

                # Prepare conditioning inputs
                model_kwargs = self.collate_model_kwargs(b_sample)
                model_kwargs["hint"] = incomplete

                generated = self.test_diffusion.p_sample_loop(
                    model=self.diff_model,
                    control_model=self.control_model,
                    shape=[bs, 1] + [self.config.exp.res] * 3,
                    device=self.device,
                    progress=True,
                    noise=None,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                ).detach()

                self.compute_metrics(
                    gt_sdf,
                    incomplete,
                    generated,
                    avg_meters,
                    per_fdi_meters,
                    bounds=b_sample["bounds"],
                    antag=model_kwargs["antag"],
                    label=labels,
                )

                if hasattr(self.config.exp, 'save_preds') and self.config.exp.save_preds:
                    if self.config.antag.is_used:
                        data_dict = {'sample': generated, 'gt': gt_sdf, 'incomplete': incomplete, 'antag': model_kwargs["antag"]}
                        self.save_samples(self.config, data_dict , idx)
                    else:
                        data_dict = {'sample': generated, 'gt': gt_sdf, 'incomplete': incomplete, 'antag': incomplete}
                        self.save_samples(self.config, data_dict, idx)

                self.log_avg_meters(avg_meters)

            if max_samples <= idx:
                break

        self.log_fdi_meters(per_fdi_meters)
        self.log_tables(avg_meters)


    def save_samples(self, config, data, curr_iter):
        samples = data['sample']
        gt = data['gt']
        incomplete = data['incomplete']
        if config.antag.is_used:
            antag = data['antag']

        bs = samples.size(0)
        for idx, samp in enumerate(samples):
            if (bs * curr_iter + idx) % config.data.save_interval == 0:
                os.makedirs(os.path.join(config.exp.log_dir, str(wb.run.name), f"batch_{curr_iter}_samp{str(idx)}"),
                            exist_ok=True)
                np.save(os.path.join(config.exp.log_dir, str(wb.run.name), f"batch_{curr_iter}_samp{str(idx)}",
                                     'sample.npy'), samp[0].cpu().numpy())
                np.save(
                    os.path.join(config.exp.log_dir, str(wb.run.name), f"batch_{curr_iter}_samp{str(idx)}", 'gt.npy'),
                    gt[idx].cpu().numpy())
                np.save(os.path.join(config.exp.log_dir, str(wb.run.name), f"batch_{curr_iter}_samp{str(idx)}",
                                     'incomplete.npy'), incomplete[idx][0].cpu().numpy())
                if config.antag.is_used:
                    np.save(os.path.join(config.exp.log_dir, str(wb.run.name), f"batch_{curr_iter}_samp{str(idx)}",
                                         'antag.npy'), antag[idx].cpu().numpy())



    def label2class(self, label):
        pos = int(label)
        if pos in [0,1,8,9]:
            return 'incisor'
        elif pos in [2,10]:
            return 'canine'
        elif pos in [3,4,11,12]:
            return 'premolar'
        elif pos in [5,6,7,13,14,15]:
            return 'molar'
        else:
            return 'unknown'

    def compute_metrics(self, gts, inputs, preds, avg_meters, per_fdi_meters=None, antag=None, bounds=None, label=None):

        gts = gts.cpu().numpy()
        inputs = inputs.cpu().numpy()
        preds = preds.cpu().numpy()
        if antag is not None:
            antag = antag.cpu().numpy()

        for i, samp in enumerate(preds):
            # Clamp the values for better computation
            samp = np.clip(samp[0], -1, 1)
            gt = gts[i][0]
            input = inputs[i][0]
            meters = {}
            fdi = label[i]
            if bounds is not None:
                bound = bounds[i].cpu().numpy()
            else:
                bound = None
            # Compute L1
            compute_L1(gt, input, samp, meters, bound)

            # Compute CD
            compute_CD(gt, input, samp, meters, bound)

            # Compute IOU
            if antag is not None:
                an = antag[i]
                compute_IoU(gt, input, samp, meters, bound, antag=an)
            else:
                compute_IoU(gt, input, samp, meters, bound)

            if label is not None and per_fdi_meters is not None:
                t_class = self.label2class(fdi)
                for k, v in meters.items():
                    if f"{t_class}_{k}" not in per_fdi_meters:
                        per_fdi_meters[f"{t_class}_{k}"] = AverageMeter()
                    per_fdi_meters[f"{t_class}_{k}"].update(v.val)


            for k, v in meters.items():
                if k not in avg_meters:
                    avg_meters[k] = AverageMeter()
                avg_meters[k].update(v.val)



@hydra.main(config_path='configs/test', config_name='test_debug')
def main(config):
    trainer = ToothCraftTester(config)
    trainer.test()

if __name__ == '__main__':
    main()