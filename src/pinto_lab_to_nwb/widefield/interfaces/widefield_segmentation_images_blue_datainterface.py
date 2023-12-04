from pathlib import Path

import numpy as np
from neuroconv import BaseDataInterface
from neuroconv.tools import get_module
from neuroconv.utils import FolderPathType, FilePathType
from pymatreader import read_mat
from pynwb import NWBFile
from pynwb.base import Images
from pynwb.image import GrayscaleImage


class WidefieldSegmentationImagesBlueInterface(BaseDataInterface):
    """The custom interface to add the blue channel manual and vasculature mask to the NWBFile."""

    def __init__(
        self,
        vasculature_mask_file_path: FilePathType,
        manual_mask_file_path: FilePathType,
        manual_mask_struct_name: str,
        verbose: bool = True,
    ):
        """
        The interface to add the summary images to the NWBFile.

        Parameters
        ----------
        vasculature_mask_file_path : FilePathType
            The file path to the vasculature mask file.
        manual_mask_file_path : FilePathType
            The file path to the manual mask file.
        manual_mask_struct_name: str
            The name of the struct in the manual mask file that contains the manual mask (e.g. "regMask" or "reg_manual_mask").
        verbose : bool, default: True
        """
        super().__init__(vasculature_mask_file_path=vasculature_mask_file_path)
        self.vasculature_mask_file_path = Path(vasculature_mask_file_path)
        assert (
            self.vasculature_mask_file_path.exists()
        ), f"The vasculature mask file '{vasculature_mask_file_path}' does not exist."

        self.manual_mask_file_path = Path(manual_mask_file_path)
        assert self.manual_mask_file_path.exists(), f"The manual mask file '{manual_mask_file_path}' does not exist."

        self.manual_mask_struct_name = manual_mask_struct_name

        self.verbose = verbose

        self._image_vasculature = self._load_vasculature_mask()
        self._image_manual = self._load_manual_mask()

    def _load_vasculature_mask(self) -> np.ndarray:
        vasculature_mask_mat = read_mat(str(self.vasculature_mask_file_path))
        assert (
            "mask" in vasculature_mask_mat
        ), f"The vasculature mask is missing from {self.vasculature_mask_file_path}."
        vasculature_mask = vasculature_mask_mat["mask"]

        return vasculature_mask

    def _load_manual_mask(self):
        manual_mask_mat = read_mat(self.manual_mask_file_path)
        assert (
            self.manual_mask_struct_name in manual_mask_mat
        ), f"The manual mask is missing from {self.manual_mask_file_path}."
        manual_mask = manual_mask_mat[self.manual_mask_struct_name]

        return manual_mask

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: dict):
        ophys = get_module(nwbfile=nwbfile, name="ophys")

        images_container_name = "SegmentationImagesBlue"
        if images_container_name in ophys.data_interfaces:
            raise ValueError(f"Images container {images_container_name} already exists in the NWBFile.")

        description = "Contains the manual mask and the contrast based vasculature mask for the blue channel in the full size session image."
        images_container = Images(
            name=images_container_name,
            description=description,
        )
        ophys.add(images_container)

        vasculature_image_data = self._image_vasculature.T
        images_container.add_image(
            GrayscaleImage(
                name="vasculature",
                description="The contrast based vasculature mask for the blue channel.",
                data=vasculature_image_data,
            )
        )

        manual_mask_data = self._image_manual.T
        images_container.add_image(
            GrayscaleImage(name="manual", description="The manual mask for the blue channel.", data=manual_mask_data)
        )
