import hydra
import multiprocessing as mp
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from generator import *
import wandb as wb

class OODTeethLoader:
    def __init__(self, config):
        self.config = config
        self.set_names = self.get_set_names(config.set)
        if hasattr(config.data, "max_test"):
            self.set_names = self.set_names[:config.data.max_test]
        self.jaw_paths = self.find_model_paths()

    def __len__(self):
        return len(self.jaw_paths)

    def __iter__(self):
        return iter(self.jaw_paths)

    def get_set_names(self, set):
        resulting_lines = []
        with open(os.path.join(self.config.data.splits, f"{set}.txt"),) as f:

            for line in f:
                line = line.strip()
                numbers = line.split(",")
                resulting_lines += [x.strip() for x in numbers]
        return resulting_lines

    def find_model_paths(self):
        files = os.listdir(self.config.data.path)
        files.sort()

        lu_stl_paths = []

        for file in files:
            if file in self.set_names:
                l_path = osp.join(self.config.data.path, file, "final", "L_Final.stl")
                u_path = osp.join(self.config.data.path, file, "final", "U_Final.stl")
                lu_stl_paths.append({"lower": l_path, "upper": u_path})
        return lu_stl_paths



class Teeth3DSLoader:
    def __init__(self, config):
        self.config = config
        self.jaw_paths = self.find_model_paths(self.config.set)

    def __len__(self):
        return len(self.jaw_paths)

    def __iter__(self):
        return iter(self.jaw_paths)

    def find_model_paths(self, set):
        # Get all the file names
        lower_files = os.listdir(osp.join(self.config.data.path, f'{set}/lower'))
        lower_files.sort()
        upper_files = os.listdir(osp.join(self.config.data.path, f'{set}/upper'))
        upper_files.sort()

        # Iterate through one to  get paired lists
        lu_stl_paths = []
        for file in lower_files:
            paths_dict = {}
            if osp.exists(osp.join(self.config.data.path, f'{set}/upper', file)):
                u_path = osp.join(self.config.data.path, f'{set}/upper', file, f'{file}_upper.obj')
                paths_dict['upper'] = u_path
                upper_files.remove(file)

            if osp.exists(osp.join(self.config.data.path, f'{set}/lower', file)):
                l_path = osp.join(self.config.data.path, f'{set}/lower', file, f'{file}_lower.obj')
                paths_dict['lower'] = l_path
            if len(paths_dict) > 0:
                lu_stl_paths.append(paths_dict)
        # Add the rest
        for file in upper_files:
            paths_dict = {}
            if osp.exists(osp.join(self.config.data.path, f'{set}/upper', file)):
                u_path = osp.join(self.config.data.path, f'{set}/upper', file, f'{file}_upper.obj')
                paths_dict['upper'] = u_path
            if len(paths_dict) > 0:
                lu_stl_paths.append(paths_dict)

        return lu_stl_paths


def run_multiproces_set(config):

    # Get the data loader
    if config.data.name == "OOD":
        loader = OODTeethLoader(config)
    elif config.data.name == "3DS":
        loader = Teeth3DSLoader(config)
    else:
        raise NotImplementedError

    if config.wandb.key is not None:
        os.environ["WANDB__SERVICE_WAIT"] = "300"
        wb.login(key=config.wandb.key, relogin=True, force=True)
        wb_id = wb.util.generate_id()
        run = wb.init(project=config.wandb.project, entity=config.wandb.entity,
                           config=dict(config),
                           id=wb_id,
                           resume="allow")
    # Get the generator
    data_gen = Generator(config)
    data_gen.create_folder_structure()
    timeout = 60 * 15
    LOG_EVERY = 100
    all_ious = []
    fdi_ious = {}
    zero_ious = []
    if config.multiprocessing:

        with mp.Pool(config.cpu_count) as pool:
            # Initialize tqdm with the total number of tasks
            with tqdm(total=len(loader)) as pbar:
                # Map the tasks and update the progress bar

                for i, result in enumerate(pool.imap(data_gen.process_model, loader)):

                    ious = result[0]

                    if len(ious) == 0:
                        continue
                    fdi = result[1]

                    for j, fdi in enumerate(fdi):
                        if fdi not in fdi_ious.keys():
                            fdi_ious[fdi] = []
                        fdi_ious[fdi].append(ious[j])

                    all_ious += ious
                    zero_ious += result[2]
                    if config.wandb.key is not None:
                        if i % LOG_EVERY == 0:
                            hist, bins = np.histogram(all_ious, bins=20, range=(0, 1))
                            wb.log({
                                "iou_hist": wb.Histogram(np_histogram=(hist, bins)),
                                "zero_iou": sum(zero_ious) / len(zero_ious),
                                "iou_mean": sum(all_ious) / len(all_ious),
                            })

                            for fdi, ious in fdi_ious.items():
                                hist, bins = np.histogram(ious, bins=20, range=(0, 1))
                                mir_fdi = mirrored_fdi_label(int(fdi))
                                wb.log({
                                    f"FDI_{mir_fdi}_hist": wb.Histogram(np_histogram=(hist, bins)),
                                    f"FDI_{mir_fdi}_mean": sum(ious) / len(ious),
                                })
            
                    pbar.update(1)

                if config.wandb.key is not None:
                    hist, bins = np.histogram(all_ious, bins=20, range=(0, 1))
                    bin_centers = 0.5 * (bins[:-1] + bins[1:])
                    bin_centers = np.round(bin_centers, 3)

                    table = wb.Table(data=[[v] for v in all_ious], columns=['value'])
                    wb.log({'final_tab_hist': wb.plot.histogram(table=table, value='value', title='final_hist')})
                    table = wb.Table(
                        data=list(zip(hist,bin_centers)),
                        columns=["count", "bin_center"]
                    )
                    wb.log({"final_bar": wb.plot.bar(table, value="count", label="bin_center", title="final_bar")
                    })

                    wb.log({'iou': wb.Histogram(np_histogram=(hist, bins))})
                    wb.run.summary['final_hist'] = wb.Histogram(np_histogram=(hist, bins))

    else:
        for i, model_path in enumerate(tqdm(loader)):
            _ = data_gen.process_model(model_path)

def mirrored_fdi_label(fdi: int) -> int:
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

@hydra.main(version_base=None, config_path="configs", config_name="debug")
def main(config):
    run_multiproces_set(config)

if __name__ == "__main__":
    main()

