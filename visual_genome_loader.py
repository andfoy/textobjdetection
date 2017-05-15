
import os
import torch
import errno
import numpy as np
import os.path as osp
from PIL import Image
import torch.utils.data as data
import visual_genome.local as vg


class VisualGenomeLoader(data.Dataset):
    def __init__(self, root, offset=64, transform=None, target_transform=None,
                 train=False, test=False):
        self.root = root
        self.graph_path = osp.join(root, 'by-id')
        self.transform = transform

        self.head = 0
        self.offset = 64

        if not osp.exists(self.root):
            raise RuntimeError('Dataset not found ' +
                               'please download it from: ' +
                               'http://visualgenome.org/api/v0/api_home.html')

        if not osp.exists(self.graph_path):
            print('Processing scene graphs...')
            vg.add_attrs_to_scene_graphs(self.root)
            vg.save_scene_graphs_by_id(data_dir=self.root,
                                       image_data_dir=self.graph_path)
            print('Done!')

        self.region_descriptions = vg.get_all_region_descriptions(
            data_dir=self.root)
