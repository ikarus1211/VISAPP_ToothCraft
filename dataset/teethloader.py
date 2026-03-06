import torch
import numpy as np
import os
import os.path as osp
from torch.utils.data import Dataset
import omegaconf
import hydra
from torch.utils.data.sampler import Sampler

class InfSampler(Sampler):
    """Samples elements randomly, without replacement.

      Arguments:
          data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, shuffle=True):
        self.data_source = data_source
        self.shuffle = shuffle
        self.reset_permutation()

    def reset_permutation(self):
        perm = len(self.data_source)
        if self.shuffle:
            perm = torch.randperm(perm)
        self._perm = perm.tolist()

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._perm) == 0:
            self.reset_permutation()

        return self._perm.pop()

    def __len__(self):
        return len(self.data_source)


class TeethDataset(Dataset):
    # Specify different subpaths if needed
    DATA_PATH_FILE = {
        'train': 'train',
        'val': 'val',
        'test': 'test'
    }

    def __init__(self, config, phase):

        self.config = config
        self.phase = phase

        # Check for multiple data loading locations
        if isinstance(config.data.data_dir, omegaconf.ListConfig):
            self.data_p = []
            for data_root in config.data.data_dir:
                data_path = osp.join(data_root, self.DATA_PATH_FILE[phase])
                data_path = hydra.utils.to_absolute_path(data_path)
                paths = self.get_paths(data_path)
                self.data_p += paths
        else:
            self.data_root = osp.join(str(config.data.data_dir), self.DATA_PATH_FILE[phase])
            self.data_root = hydra.utils.to_absolute_path(self.data_root)
            self.data_p = self.get_paths(self.data_root)


    def get_paths(self, data_root):

        data_p = []
        # Get all data paths
        incomplete_models= os.listdir(os.path.join(data_root, 'incomplete'))
        for i_mod_name in incomplete_models:
            # Path to hole
            incomplete_p = os.path.join(data_root, 'incomplete', i_mod_name)
            # Path to gt
            gt_name = i_mod_name.split('_')
            gt_name = '_'.join(gt_name[:-1]) + '_gt_' + gt_name[-1]
            gt_p = os.path.join(data_root, 'gt', gt_name)

            # Load bounds
            if self.config.data.bounds and os.path.exists(osp.join(data_root, 'bounds')):
                bound_name = i_mod_name.split('_')
                bound_name = '_'.join(bound_name[:-1]) + '_b_' + bound_name[-1]
                bound_p = osp.join(data_root, 'bounds', bound_name)
            else:
                bound_p = []

            # Path to antagonist
            if self.config.antag.is_used:
                gt_name = i_mod_name.split('_')
                gt_name = '_'.join(gt_name[:-1]) + '_antag_' + gt_name[-1]
                antag_p = os.path.join(data_root, 'antag', gt_name)
                data_p.append({'incomplete_p': incomplete_p, "gt_p": gt_p, "antag_p": antag_p,
                               'name': i_mod_name, 'bound_p': bound_p})
            else:
                data_p.append({'incomplete_p': incomplete_p, "gt_p": gt_p, 'name': i_mod_name, 'bound_p': bound_p})

        return data_p

    def __len__(self):
        return len(self.data_p)

    def torch_from_path(self, path):
        with open(path, "rb") as f:
            sdf = np.load(f, allow_pickle=True)
        sdf = self.normalize(sdf)
        sdf = torch.from_numpy(sdf).float()
        return sdf

    def get_fdi_label(self, name):

        return int(name.split('.')[0].split('_')[-1])


    def __getitem__(self, idx):

        data_dict = {}

        # Get name
        name = self.data_p[idx]['name']
        data_dict['name'] = name

        # Load incomplete sdf
        inc_path = self.data_p[idx]['incomplete_p']
        data_dict['incomplete'] = self.torch_from_path(inc_path)

        # Load gt sdf
        gt_p = self.data_p[idx]['gt_p']
        data_dict['gt'] = self.torch_from_path(gt_p)

        # Load FDI if class conditioned else returns name
        data_dict['label'] = self.mirrored_fdi_label(self.get_fdi_label(name))

        if self.config.data.bounds:
            data_dict['bounds'] = np.load(self.data_p[idx]['bound_p'])
        else:
            data_dict['bounds'] = []

        # Load antagonist
        if self.config.antag.is_used:
            antag_p = self.data_p[idx]['antag_p']
            data_dict['antag'] = self.torch_from_path(antag_p)

        return data_dict

    def mirrored_fdi_label(self, fdi: int) -> int:
        mirror_map = {
            11: 0, 21: 0,
            12: 1, 22: 1,
            13: 2, 23: 2,
            14: 3, 24: 3,
            15: 4, 25: 4,
            16: 5, 26: 5,
            17: 6, 27: 6,
            18: 7, 28: 7,
            31: 8, 41: 8,
            32: 9, 42: 9,
            33: 10, 43: 10,
            34: 11, 44: 11,
            35: 12, 45: 12,
            36: 13, 46: 13,
            37: 14, 47: 14,
            38: 15, 48: 15,
        }
        return mirror_map.get(fdi, -1)

    def normalize(self, sdf):
        sdf = sdf.clip(-1, 1)
        return sdf



@hydra.main(config_path="../configs/test", config_name="test_debug")
def main(config):
    dataset = TeethDataset(config=config, phase='test')
    print(len(dataset))
    labels = []
    for i in dataset:
        labels.append(i['label'])
    print(set(labels))

if __name__ == '__main__':
    main()