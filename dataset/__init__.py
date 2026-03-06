from torch.utils.data import DataLoader
from dataset.utils import InfSampler
from dataset.teethloader import TeethDataset
from torch.utils.data.sampler import RandomSampler


def initialize_data_loader(config, phase, repeat=False):

    dataset = TeethDataset(config, phase)
    if repeat:
        sampler = InfSampler(dataset, shuffle=True)
    else:
        sampler = RandomSampler(dataset, replacement=False)

    if phase == 'train':
        train_dl = DataLoader(dataset,
                              batch_size=config.exp.batch_size,
                              sampler=sampler,
                              num_workers=config.exp.num_workers)
        return train_dl
    elif phase == 'val':
        if config.val.batch_size and config.val.batch_size != 0:
            batch_size = config.val.batch_size
        else:
            batch_size = config.exp.batch_size
        val_dl = DataLoader(dataset,
                            batch_size=batch_size,
                            sampler=sampler,
                            num_workers=config.exp.num_workers)
        return val_dl

    elif phase == 'test':
        test_dl = DataLoader(dataset,
                            batch_size=config.exp.batch_size,
                            sampler=sampler,
                            num_workers=config.exp.num_workers)
        return test_dl