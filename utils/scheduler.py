import torch
import math

class DecayingCosineAnnealingWarmRestarts(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, decay=2/3, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.T_cur = 0  # current step in cycle
        self.cycle = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.eta_min = eta_min
        self.decay = decay
        self.cycle_steps = T_0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        factor = self.decay ** self.cycle
        return [
            self.eta_min + (base_lr * factor - self.eta_min) *
            (1 + math.cos(math.pi * self.T_cur / self.cycle_steps)) / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        if epoch is None:
            self.T_cur += 1
            if self.T_cur >= self.cycle_steps:
                self.T_cur = 0
                self.cycle += 1
                self.cycle_steps *= self.T_mult
        else:
            # optional: handle epoch-based updates
            pass
        super().step()


class CosineDecayWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, T_max, eta_min=0.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.T_max = T_max
        self.min_lr = eta_min
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1

        if step <= self.warmup_steps:
            scale = step / self.warmup_steps
        else:
            decay_step = step - self.warmup_steps
            decay_total = self.T_max - self.warmup_steps
            cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_step / decay_total))
            scale = cosine_decay

        return [
            self.min_lr + (base_lr - self.min_lr) * scale
            for base_lr in self.base_lrs
        ]