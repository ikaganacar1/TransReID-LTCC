# encoding: utf-8
"""
LTCC (Long-Term Cloth-Changing) Person Re-Identification Dataset
"""

import glob
import re
import os.path as osp
from .bases import BaseImageDataset


class LTCC(BaseImageDataset):
    """
    LTCC Dataset for Long-Term Cloth-Changing Person Re-identification

    Dataset statistics:
    # identities: 152
    # images: 9576 (train) + varies (query/gallery based on split)
    # cameras: 12
    """
    dataset_dir = 'data'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(LTCC, self).__init__()
        self.dataset_dir = osp.join(root)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()
        self.pid_begin = pid_begin

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> LTCC loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        """
        Process directory and extract person IDs and camera IDs from filenames
        LTCC format: XXXX_cYYsZ_XX_XX.jpg
        Where XXXX is person ID, YY is camera ID, Z is sequence/session
        """
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))

        # Pattern to match LTCC naming: personID_cCameraIDsSession_frame_seq.jpg
        # Example: 0000_c10s1_03_00.jpg
        pattern = re.compile(r'([\d]+)_c([\d]+)s')

        pid_container = set()
        for img_path in sorted(img_paths):
            match = pattern.search(osp.basename(img_path))
            if match:
                pid = int(match.group(1))
                pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        dataset = []
        for img_path in sorted(img_paths):
            match = pattern.search(osp.basename(img_path))
            if match:
                pid = int(match.group(1))
                camid = int(match.group(2))
                camid -= 1  # Convert to 0-indexed (e.g., 1-12 becomes 0-11)

                # Relabel if needed (for training set)
                if relabel:
                    pid = pid2label[pid]

                # For LTCC, we use session as viewpoint (can be extracted from filename)
                # Format: personID_cCameraIDsSession_frame_seq.jpg
                viewid = 1  # Default viewid, can be extracted if needed

                dataset.append((img_path, self.pid_begin + pid, camid, viewid))

        return dataset
