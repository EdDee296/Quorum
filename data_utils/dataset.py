import os
import re
import cv2
import numpy as np
import tifffile as tiff
import albumentations as A
import torch
from torch.utils.data import Dataset

class CellDataset(Dataset):
    """PyTorch Dataset for cell segmentation.
    Returns image tensor (1,H,W) and mask tensor (H,W) with labels {0,1,2}.
    """

    def __init__(self, root_dir, preprocess_mode='basic', aug_strength='standard', target_size=(256, 256), split_ids=None):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'Microscopy_images')
        
        # Handle potential folder typos
        alt1 = os.path.join(root_dir, 'Ground_truth_masks')
        alt2 = os.path.join(root_dir, 'Groud_truth_masks')
        self.mask_dir = alt1 if os.path.exists(alt1) else alt2
        
        self.preprocess_mode = preprocess_mode
        self.target_size = target_size
        self.aug = self._make_augmentation(aug_strength, target_size=target_size)

        self.samples = self._find_samples(split_ids)
        if len(self.samples) == 0:
            raise RuntimeError(f'No samples found in {self.img_dir}. Check paths.')

    @staticmethod
    def load_split_ids(file_path):
        """Helper to read IDs from val_ids.txt."""
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found.")
            return []
        with open(file_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    def _make_augmentation(self, strength, target_size=None):
        transforms = []
        if strength == 'light':
            transforms.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ])
        elif strength == 'standard':
            transforms.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)}, 
                         scale=(0.85, 1.15), rotate=(-45, 45), 
                         border_mode=cv2.BORDER_REFLECT, p=0.5),
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
            ])
        
        if target_size is not None:
            transforms.append(A.Resize(height=target_size[0], width=target_size[1], 
                                       interpolation=cv2.INTER_LINEAR, 
                                       mask_interpolation=cv2.INTER_NEAREST))

        return A.Compose(transforms) if transforms else None

    def _find_samples(self, split_ids=None):
        pattern = re.compile(r'Microscope_image_Cell(\d+)\.tif')
        ids = []
        split_set = set(split_ids) if split_ids else None
        
        for f in sorted(os.listdir(self.img_dir)):
            m = pattern.match(f)
            if not m: continue
            cid = m.group(1)
            
            nuc = os.path.join(self.mask_dir, f'Nucleus_mask_Cell{cid}.tif')
            chrm = os.path.join(self.mask_dir, f'Chromocenter_mask_Cell{cid}.tif')
            
            if os.path.exists(nuc) and os.path.exists(chrm):
                if split_set is None or cid in split_set:
                    ids.append(cid)
        return ids

    def _preprocess_image(self, img_u16):
        img = img_u16.astype(np.float32)
        lo, hi = np.percentile(img, 1), np.percentile(img, 99.5)
        img = np.clip(img, lo, hi)
        img01 = (img - lo) / (hi - lo + 1e-8)

        if self.preprocess_mode == 'basic':
            # Just returns the clipped/normalized image
            pass 

        elif self.preprocess_mode == 'full':
            # Noise reduction + Background subtraction
            img01 = cv2.bilateralFilter(img01, d=5, sigmaColor=0.08, sigmaSpace=3)
            bg = cv2.GaussianBlur(img01, (51, 51), 0)
            corr = img01 - bg
            l2, h2 = np.percentile(corr, 1), np.percentile(corr, 99.5)
            img01 = np.clip((corr - l2) / (h2 - l2 + 1e-8), 0, 1)

        else:
            raise ValueError(f"Invalid preprocess_mode '{self.preprocess_mode}'. "
                             f"Allowed modes are: ['basic', 'full']")

        return img01.astype(np.float32)

    def _build_mask(self, nuc_raw, chrom_raw):
        mask = np.zeros_like(nuc_raw, dtype=np.uint8)
        mask[nuc_raw == 0] = 1 # Nucleus
        mask[chrom_raw == 0] = 2 # Chromocenter
        return mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cid = self.samples[idx]
        img = tiff.imread(os.path.join(self.img_dir, f'Microscope_image_Cell{cid}.tif'))
        nuc = tiff.imread(os.path.join(self.mask_dir, f'Nucleus_mask_Cell{cid}.tif'))
        chrom = tiff.imread(os.path.join(self.mask_dir, f'Chromocenter_mask_Cell{cid}.tif'))

        img_pp = self._preprocess_image(img)
        mask = self._build_mask(nuc, chrom)

        if self.aug is not None:
            augmented = self.aug(image=img_pp, mask=mask)
            img_pp, mask = augmented['image'], augmented['mask']
        elif self.target_size is not None:
            img_pp = cv2.resize(img_pp, self.target_size, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        return torch.from_numpy(img_pp).unsqueeze(0).float(), torch.from_numpy(mask.astype(np.int64))