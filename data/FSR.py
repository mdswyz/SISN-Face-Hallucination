import os
import glob
import data

class FSR(data.BaseDataset):
    def __init__(self, phase, opt):
        root = opt.dataset_root

        self.scale = opt.scale
        dir_HQ, dir_LQ = self.get_subdir()
        self.HQ_paths = sorted(glob.glob(os.path.join(root, dir_HQ, "*.png")))
        self.LQ_paths = sorted(glob.glob(os.path.join(root, dir_LQ, "*.png")))

        # print(len(self.HQ_paths))
        # print(len(self.LQ_paths))

        split = [int(n) for n in opt.train_val_range.replace("/", "-").split("-")]
        if phase == "train":
            s = slice(split[0]-1, split[1])
            self.HQ_paths, self.LQ_paths = self.HQ_paths[s], self.LQ_paths[s]
        else:
            s = slice(split[2]-1, split[3])
            self.HQ_paths, self.LQ_paths = self.HQ_paths[s], self.LQ_paths[s]

        super().__init__(phase, opt)

    def get_subdir(self):
        dir_HQ = "HR"
        dir_LQ = "LR/X{}".format(self.scale)
        return dir_HQ, dir_LQ

