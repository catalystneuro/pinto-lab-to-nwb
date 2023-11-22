from pathlib import Path
from typing import Tuple

import numpy as np
from neuroconv.utils import FolderPathType
from pymatreader import read_mat
from roiextractors import SegmentationExtractor


class WidefieldProcessedSegmentationExtractor(SegmentationExtractor):
    """Custom extractor for reading segmentation data for the Widefield experiment"""

    extractor_name = "WidefieldProcessedSegmentation"
    mode = "file"

    def __init__(
        self,
        folder_path: FolderPathType,
    ):
        """
        The SegmentationExtractor for the downsampled (binned) Widefield imaging data.

        The segmentation data is stored in .mat files:
        - info.mat          : contains the general metadata of the imaging session such as frame rate etc.
        - ROIfromRef.mat     : contains the Allen area label of each pixel mapped onto the reference image of the mouse and registered to the session.
        - vasculature_mask_2.mat : contains the vasculature mask on the downsampled (binned) session image.
        - blue_pca_vasculature_mask_2.mat : contains the PCA mask for the blue channel.
        - violet_pca_vasculature_mask_2.mat.mat : contains the PCA mask for the violet channel.

         that contain the following variables:
        - The Allen area label for each binned pixel
        - The contrast based vasculature mask (corresponds to "binned_vasculature_mask_file_path")
        - The PCA masks for the blue and violet channels (corresponds to "blue_pca_mask_file_path" and "violet_pca_mask_file_path")

        Parameters
        ----------
        folder_path: FolderPathType
            The path that points to the folder that contains the .mat files.
        """
        super().__init__()

        self.folder_path = Path(folder_path)

        expected_files = [
            "info.mat",
            "ROIfromRef.mat",
            "vasculature_mask_2.mat",
            "blue_pca_vasculature_mask_2.mat",
        ]
        mat_file_paths = list(self.folder_path.glob("*.mat"))
        assert mat_file_paths, f"The .mat files are missing from {folder_path}."
        # assert all expected files are in the folder_path
        for expected_file in expected_files:
            assert (
                self.folder_path / expected_file
            ).exists(), f"The file {expected_file} is missing from {folder_path}."

        info_mat = read_mat(self.folder_path / "info.mat")
        assert "info" in info_mat, f"Could not find 'info' struct in 'info.mat'."
        self._num_frames = info_mat["info"]["numFrames"]
        self._sampling_frequency = info_mat["info"]["frameRate"]

        roi_mat = read_mat(self.folder_path / "ROIfromRef.mat")
        assert "ROIcentroids" in roi_mat, f"Could not find 'ROIcentroids' in 'ROIfromRef.mat'."
        self._roi_locations = roi_mat[
            "ROIcentroids"
        ].T  # they should be in height x width (orig they are width x height)
        # Allen area locations
        assert "ROIlbl" in roi_mat, f"Could not find 'ROIlbl' in 'ROIfromRef.mat'."
        self._roi_labels = roi_mat["ROIlbl"]

        binned_height = int(info_mat["info"]["height"] / roi_mat["dsFactor"])
        binned_width = int(info_mat["info"]["width"] / roi_mat["dsFactor"])
        self._image_size = (binned_height, binned_width)

        assert "ROI" in roi_mat, f"Could not find 'ROI' in 'ROIfromRef.mat'."
        self._image_masks = self._compute_image_masks(pixel_mask=roi_mat["ROI"])
        self._dtype = self._image_masks.dtype

        # Contrast based vasculature mask
        vasculature_mask = read_mat(self.folder_path / "vasculature_mask_2.mat")
        assert "mask_binned" in vasculature_mask, f"Could not find 'mask_binned' in 'vasculature_mask_2.mat'."
        self._image_vasculature = vasculature_mask["mask_binned"]

        # PCA mask (separate for blue and violet)
        pca_mask_blue = read_mat(self.folder_path / f"blue_pca_vasculature_mask_2.mat")
        assert "mask" in pca_mask_blue, f"Could not find 'mask' in 'blue_pca_vasculature_mask_2.mat'."
        self._image_pca_blue = pca_mask_blue["mask"]

    def _compute_image_masks(self, pixel_mask):
        """Compute the image masks from the ROI's pixel locations."""
        num_rois = self.get_num_rois()
        image_mask = np.zeros(shape=(*self._image_size, num_rois), dtype=np.uint8)
        for roi_ind, pixel_mask_roi in enumerate(pixel_mask):
            pixel_mask_roi = pixel_mask_roi[0]
            if len(pixel_mask_roi) == 0:
                # there are rois with no pixels
                continue
            x = pixel_mask_roi[:, 0] - 1
            y = pixel_mask_roi[:, 1] - 1
            image_mask[x, y, roi_ind] = 1

        return image_mask

    def get_channel_names(self):
        return ["OpticalChannelBlue"]

    def get_roi_locations(self, roi_ids=None) -> np.ndarray:
        return self._roi_locations

    def get_images_dict(self):
        """Return the images dict that contain the contrast based vasculature mask and the PCA mask for the blue channel."""
        images_dict = super().get_images_dict()
        images_dict.update(
            vasculature=self._image_vasculature,
            pca_blue=self._image_pca_blue,
        )
        return images_dict

    def get_accepted_list(self) -> list:
        return self.get_roi_ids()

    def get_rejected_list(self) -> list:
        return list()

    def get_num_frames(self) -> int:
        return self._num_frames

    def get_image_size(self) -> Tuple[int, int]:
        return self._image_size

    def get_num_rois(self) -> int:
        return self._roi_locations.shape[1]
