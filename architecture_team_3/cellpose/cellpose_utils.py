import numpy as np
import cv2
from cellpose import models


class CellposeHelper:
    """
    Small helper around Cellpose so backend/train/eval code can reuse the same logic.
    """

    def __init__(
        self,
        cellpose_diameter=None,
        cellpose_model_type='nuclei',
        use_gpu=True,
        chromocenter_percentile=85,
    ):
        self.cellpose_diameter = cellpose_diameter
        self.chromocenter_percentile = chromocenter_percentile

        print(f"Initializing Cellpose model (type: {cellpose_model_type}, GPU: {use_gpu})...")
        self.cellpose_model = models.Cellpose(gpu=use_gpu, model_type=cellpose_model_type)
        print("Cellpose model loaded successfully")

    def run_cellpose(self, img_raw):
        """
        Run Cellpose on a raw microscopy image.
        Returns the nucleus instance mask from Cellpose.
        """
        if img_raw.dtype != np.uint8:
            img_min = img_raw.min()
            img_max = img_raw.max()

            if img_max > img_min:
                img_normalized = ((img_raw - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                img_normalized = np.zeros_like(img_raw, dtype=np.uint8)
        else:
            img_normalized = img_raw

        masks, flows, styles, diams = self.cellpose_model.eval(
            img_normalized,
            diameter=self.cellpose_diameter,
            channels=[[0, 0]],   # grayscale
            flow_threshold=0.4,
            cellprob_threshold=0.0,
        )

        return masks

    def detect_chromocenters_intensity(self, img_raw, nucleus_mask):
        """
        Detect chromocenters by thresholding bright pixels inside the nucleus region.
        Returns a binary mask.
        """
        chromocenter_mask = np.zeros_like(nucleus_mask, dtype=np.uint8)

        nuclei = nucleus_mask > 0
        if np.any(nuclei):
            nuclei_intensities = img_raw[nuclei]
            intensity_threshold = np.percentile(
                nuclei_intensities,
                self.chromocenter_percentile
            )

            chromocenters = nuclei & (img_raw >= intensity_threshold)
            chromocenter_mask[chromocenters] = 1

        return chromocenter_mask

    def build_mask_cellpose(self, img_raw):
        """
        Build a 3-class mask using:
        - Cellpose for nucleus detection
        - intensity thresholding for chromocenters

        Output classes:
        0 -> background
        1 -> nucleoplasm
        2 -> chromocenters
        """
        cellpose_mask = self.run_cellpose(img_raw)
        chromocenter_mask = self.detect_chromocenters_intensity(img_raw, cellpose_mask)

        mask = np.zeros_like(cellpose_mask, dtype=np.uint8)

        background = cellpose_mask == 0
        mask[background] = 0

        chromocenters = chromocenter_mask == 1
        mask[chromocenters] = 2

        nuclei = cellpose_mask > 0
        nucleoplasm = nuclei & (~chromocenters)
        mask[nucleoplasm] = 1

        return mask

    def build_mask_hybrid(self, img_raw, expert_chromocenter_mask):
        """
        Build a 3-class mask using:
        - Cellpose for nucleus detection
        - expert chromocenter mask for class 2
        """
        cellpose_mask = self.run_cellpose(img_raw)

        chrm_raw = expert_chromocenter_mask
        chrm_zeros = np.count_nonzero(chrm_raw == 0)
        chrm_255s = np.count_nonzero(chrm_raw == 255)

        if chrm_zeros < chrm_255s:
            chromocenter_mask = (chrm_raw == 0).astype(np.uint8)
        else:
            chromocenter_mask = (chrm_raw == 255).astype(np.uint8)

        mask = np.zeros_like(cellpose_mask, dtype=np.uint8)

        background = cellpose_mask == 0
        mask[background] = 0

        chromocenters = chromocenter_mask == 1
        mask[chromocenters] = 2

        nuclei = cellpose_mask > 0
        nucleoplasm = nuclei & (~chromocenters)
        mask[nucleoplasm] = 1

        return mask

    @staticmethod
    def build_ground_truth_mask(nucleus_mask_raw, chromocenter_mask_raw):
        """
        Build the normal 3-class mask from expert ground truth files.
        """
        nuc = nucleus_mask_raw
        chrom = chromocenter_mask_raw

        nuc_zeros = np.count_nonzero(nuc == 0)
        nuc_255s = np.count_nonzero(nuc == 255)
        chrom_zeros = np.count_nonzero(chrom == 0)
        chrom_255s = np.count_nonzero(chrom == 255)

        if nuc_zeros < nuc_255s:
            nuc_fg = (nuc == 0).astype(np.uint8)
        else:
            nuc_fg = (nuc == 255).astype(np.uint8)

        if chrom_zeros < chrom_255s:
            chrom_fg = (chrom == 0).astype(np.uint8)
        else:
            chrom_fg = (chrom == 255).astype(np.uint8)

        chrom_fg = (chrom_fg & nuc_fg).astype(np.uint8)

        mask = np.zeros_like(nuc_fg, dtype=np.uint8)
        mask[nuc_fg == 1] = 1
        mask[chrom_fg == 1] = 2

        return mask

    @staticmethod
    def to_single_channel_uint8(img_np):
        """
        Force image into one 2D uint8 channel.
        Handy for Cellpose train/eval.
        """
        arr = np.asarray(img_np)

        if arr.ndim == 3:
            if arr.shape[0] <= 4 and arr.shape[1] > 16 and arr.shape[2] > 16:
                arr = arr[0]
            elif arr.shape[-1] <= 4 and arr.shape[0] > 16 and arr.shape[1] > 16:
                arr = arr[..., 0]
            else:
                arr = arr[0]

        if arr.ndim != 2:
            arr = np.squeeze(arr)
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D image after squeeze, got shape {arr.shape}")

        arr = arr.astype(np.float32)
        lo, hi = np.min(arr), np.max(arr)

        if hi <= lo:
            return np.zeros_like(arr, dtype=np.uint8)

        arr = (arr - lo) / (hi - lo)
        return (arr * 255).astype(np.uint8)

    @staticmethod
    def calc_metrics(pred_instances, gt_instances):
        """
        Basic IoU and Dice from instance masks.
        Treats any non-zero pixel as foreground.
        """
        pred_bin = (pred_instances > 0).astype(np.uint8)
        gt_bin = (gt_instances > 0).astype(np.uint8)

        intersection = np.logical_and(pred_bin, gt_bin).sum()
        union = np.logical_or(pred_bin, gt_bin).sum()
        total_pixels = pred_bin.sum() + gt_bin.sum()

        iou = intersection / union if union > 0 else 0.0
        dice = (2 * intersection) / total_pixels if total_pixels > 0 else 0.0

        return iou, dice

    @staticmethod
    def extract_instances_for_cellpose(dataset):
        """
        Convert semantic chromocenter masks (class 2) into instance masks for Cellpose.
        """
        imgs = []
        instance_masks = []

        for i in range(len(dataset)):
            img_t, mask_t = dataset[i]

            img_np = img_t.squeeze().numpy()
            img_uint8 = CellposeHelper.to_single_channel_uint8(img_np)

            chromo_binary = (mask_t.squeeze().numpy() == 2).astype(np.uint8)
            _, instances = cv2.connectedComponents(chromo_binary)

            imgs.append(img_uint8)
            instance_masks.append(instances.astype(np.int32))

        return imgs, instance_masks

    @staticmethod
    def extract_nucleus_instances_for_cellpose(dataset):
        """
        Convert semantic nucleus masks (class 1 or 2) into instance masks for Cellpose.
        """
        imgs = []
        instance_masks = []

        for i in range(len(dataset)):
            img_t, mask_t = dataset[i]

            img_np = img_t.squeeze().numpy()
            img_uint8 = CellposeHelper.to_single_channel_uint8(img_np)

            nucleus_binary = (mask_t.squeeze().numpy() >= 1).astype(np.uint8)
            _, instances = cv2.connectedComponents(nucleus_binary)

            imgs.append(img_uint8)
            instance_masks.append(instances.astype(np.int32))

        return imgs, instance_masks