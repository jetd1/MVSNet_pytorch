from torch.utils.data import Dataset
import numpy as np
import os
import math
from PIL import Image
from datasets.data_io import *

EPS = 0.1

# the DTU dataset preprocessed by Yao Yao (only for training)
class RISDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, mask_mode='valid', subset='denoised', align=0, ndownscale=2, **kwargs):
        super().__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.subset = subset
        self.mask_mode = mask_mode
        self.align = align
        self.ndownscale = ndownscale
        # self.interval_scale = interval_scale

        assert isinstance(self.align, int)
        assert isinstance(self.ndownscale, int)
        assert self.mode in ["train", "val", "test"]
        assert self.subset in ['noisy', 'denoised', 'albedo']
        assert self.mask_mode in ['object', 'valid']

        assert self.nviews <= 5
        self.pairs = self.build_list()

    def build_list(self):
        print('Building pair list...')
        pairs = []
        with open(self.listfile) as f:
            objs = f.read().strip().splitlines()

        for obj in objs:
            obj_path = os.path.join(self.datapath, obj)
            view_samples = os.listdir(obj_path)
            for view_id in view_samples:
                for i in range(5):
                    pairs.append((obj, view_id, i))

        print("dataset", self.mode, "pairs:", len(pairs))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        intrinsics[:2, :] /= 4
        # depth_min & depth_interval: line 11
        depth_min = math.floor(float(lines[11].split()[0]))
        depth_max = math.ceil(float(lines[11].split()[1]))
        depth_interval = round(10 * (depth_max - depth_min) / (self.ndepths - 1)) / 10
        # depth_interval = 3
        return intrinsics, extrinsics, depth_min, depth_interval

    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        if self.align > 0:
            np_img = np_img[:(np_img.shape[0] // self.align) * self.align, :(np_img.shape[1] // self.align) * self.align]
        return np_img[..., :3]
    
    def read_mask(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        if self.align > 0:
            np_img = np_img[:(np_img.shape[0] // self.align) * self.align, :(np_img.shape[1] // self.align) * self.align]
        for _ in range(self.ndownscale):
            np_img = np_img[::2, ::2]
        return np_img

    def read_depth(self, filename):
        # read pfm depth file
        depth = np.load(filename) * 1000
        if self.align > 0:
            depth = depth[:(depth.shape[0] // self.align) * self.align, :(depth.shape[1] // self.align) * self.align]
        for _ in range(self.ndownscale):
            depth = depth[::2, ::2]
        return depth

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        obj, view_id, ref_view = pair
        # use only the reference view and first nviews-1 source views
        cam_ids = [(ref_view + i) % 5 for i in range(self.nviews)]

        imgs = []
        mask = None
        depth = None
        depth_values = None
        proj_matrices = []

        for i, vid in enumerate(cam_ids):
            img_filename = os.path.join(self.datapath, f'{obj}/{view_id}/{self.subset}/{vid}.png')
            depth_filename = os.path.join(self.datapath, f'{obj}/{view_id}/depth/{vid}.npy')
            mask_filename = os.path.join(self.datapath, f'{obj}/{view_id}/masks/{vid}.png')
            proj_mat_filename = os.path.join(self.datapath, f'{obj}/{view_id}/cams/{vid}.txt')
            
            imgs.append(self.read_img(img_filename))
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat = extrinsics.copy()
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                # TODO: Why distinguish train and test here?
                if self.mode == 'test':
                    depth_values = np.arange(depth_min, depth_interval * (self.ndepths - 0.5) + depth_min, depth_interval, dtype=np.float32)
                else:
                    depth_values = np.arange(depth_min, depth_interval * self.ndepths + depth_min - EPS, depth_interval, dtype=np.float32)
                
                depth = self.read_depth(depth_filename)
                if self.mask_mode == 'valid':
                    mask = (depth != 0).astype(np.float32)

                    #fix tb normalization bug
                    mask[0, 0] = 0  
                    mask[-1, -1] = 1 
                else:
                    mask = self.read_mask(mask_filename)

        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)

        return {"imgs": imgs,
                "proj_matrices": proj_matrices,
                "depth": depth,
                "depth_values": depth_values,
                "filename": f'{obj}/{view_id}' + '/{}/' + f'{cam_ids[0]}' + '{}',
                "mask": mask}

_dataset = RISDataset

if __name__ == "__main__":
    # some testing code, just IGNORE it
    a = '/wdata/indoor/realistic_indoor_dataset/filescripts/tmp_testlock_largescale_results'
    b = '/wdata/indoor/realistic_indoor_dataset/filescripts/tmp_testlock_largescale_results/test.txt'
    dataset = RISDataset(a, b, 'test', 5, 192)
    item = dataset[1]
    for key, value in item.items():
        print(key, type(value))

    import IPython
    IPython.embed()
