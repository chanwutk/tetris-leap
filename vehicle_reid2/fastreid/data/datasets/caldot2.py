# encoding: utf-8
"""
@author:  Jinkai Zheng
@contact: 1315673509@qq.com
"""

import glob
import os.path as osp
import re

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class Caldot2(ImageDataset):
    """Caldot2.

    Reference:
        Xinchen Liu et al. A Deep Learning based Approach for Progressive Vehicle Re-Identification. ECCV 2016.
        Xinchen Liu et al. PROVID: Progressive and Multimodal Vehicle Reidentification for Large-Scale Urban Surveillance. IEEE TMM 2018.

    URL: `<https://vehiclereid.github.io/VeRi/>`_

    Dataset statistics:
        - identities: -.
        - images: -.
    """
    dataset_dir = "caldot2"
    dataset_name = "caldot2"

    def __init__(self, root='datasets', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir)
        query = self.process_dir(self.query_dir, is_train=False)
        gallery = self.process_dir(self.gallery_dir, is_train=False)

        super(Caldot2, self).__init__(train, query, gallery, **kwargs)


    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))

        data = []
        for img_path in img_paths:
            # Filename format: camid_carid_cls_xmin_ymin_xmax_ymax_frameid.jpg
            # Example: 0_22_car_0_326_33_350_695.jpg
            camid, carid, _, x_min, y_min, x_max, y_max, frame_id = img_path.split("/")[-1].rstrip(".jpg").split("_")
            
            camid = int(camid)
            carid = int(carid)
            x_min = int(x_min)
            y_min = int(y_min)
            x_max = int(x_max)
            y_max = int(y_max)
            frame_id = int(frame_id)

            if carid == -1: continue
            
            # Note: In caldot1.py there was `camid -= 1`. 
            # In generate_reid.py, we set camid to match clip id (starts at 0).
            # If fastreid expects 0-indexed camids, we might not need to subtract 1 if they are already 0-indexed.
            # However, looking at the caldot1.py code: `camid -= 1  # index starts from 0` suggests the filename camids might have been 1-based there?
            # In my generate_reid.py, cam_id comes from clip_id which could be 0.
            # If I stick to `camid` as is, it should be fine as long as it's consistent.
            # But let's check if there are any specific constraints. 
            # ReID usually treats camid as a label.
            
            # Let's keep it simple and just use the value from filename.
            
            if is_train:
                # To distinguish between different datasets if mixed, but here we train on single dataset usually.
                # However, fastreid might want string camids or remapped ones.
                # The prompt implies we are just adding Caldot2.
                # The caldot1.py does `camid = self.dataset_name + "_" + str(camid)` in is_train.
                # This seems to be for converting integer camid to a unique string identifier per dataset if combining?
                # Actually, `ImageDataset` might expect integer camids or handles them.
                # Let's look at `bases.py` if needed, but following caldot1 pattern is safest.
                
                # Wait, `caldot1.py` line 75: `camid = self.dataset_name + "_" + str(camid)`
                # Then line 77: `data.append((..., camid, ...))`
                # It seems it converts camid to string?
                # But bases.py usually expects integers for ids?
                # Let's check `bases.py` quickly or just follow pattern.
                pass

            # Actually, let's verify what `Caldot1` does exactly.
            # It seems it does `camid -= 1`. 
            # My generated data has `camid` starting from 0 (e.g. clip 0).
            # If I subtract 1, I get -1, which might be bad.
            # So I will remove `camid -= 1` for Caldot2 since my generation script produces 0-based camids.
            
            if is_train:
                 camid = self.dataset_name + "_" + str(camid)
                
            data.append((img_path, carid, camid, x_min / 720, y_min / 480, x_max / 720, y_max / 480, frame_id))

        return data
