from copy import deepcopy
from pathlib import Path

import numpy as np
from neuroconv import BaseDataInterface
from neuroconv.tools import get_module
from neuroconv.utils import FolderPathType, dict_deep_update
from pymatreader import read_mat
from pynwb import NWBFile
from pynwb.base import Images
from pynwb.image import GrayscaleImage


class WidefieldSummaryImagesBlueInterface(BaseDataInterface):

    def __init__(self, folder_path: FolderPathType, verbose: bool = True):
        """
        The interface to add the summary images to the NWBFile.

        Parameters
        ----------
        folder_path : FolderPathType
        verbose : bool, default: True
        """
        super().__init__(folder_path=folder_path)
        self.folder_path = Path(folder_path)
        self.verbose = verbose

        self._image_vasculature = self._load_vasculature_mask()
        self._image_manual = self._load_manual_mask()

    def get_metadata(self) -> dict:
        metadata = super().get_metadata()
        metadata["Ophys"] = dict(SegmentationImages=dict())

        summary_images_metadata = metadata["Ophys"]["SegmentationImages"]
        images_container_name = "SegmentationImagesBlue"
        summary_images_metadata.update(
            name=images_container_name,
            description="Contains the manual mask and the contrast based vasculature mask for the blue channel in the full size session image.",
            images=[dict(name="vasculature", description="The contrast based vasculature mask for the blue channel."),
                    dict(name="manual", description="The manual mask for the blue channel.")]
        )

        return metadata

    def _load_vasculature_mask(self) -> np.ndarray:
        vasculature_mask_file_path = self.folder_path / "vasculature_mask_2.mat"
        assert vasculature_mask_file_path.exists(), f"The vasculature mask file is missing from {self.folder_path}."
        vasculature_mask_mat = read_mat(str(vasculature_mask_file_path))
        assert "mask" in vasculature_mask_mat, f"The vasculature mask is missing from {vasculature_mask_file_path}."
        vasculature_mask = vasculature_mask_mat["mask"]

        return vasculature_mask

    def get_images_dict(self) -> dict:
        return dict(vasculature=self._image_vasculature, manual=self._image_manual)

    def _load_manual_mask(self):
        manual_mask_file_path = Path(self.folder_path) / "regManualMask.mat"
        assert manual_mask_file_path.exists(), f"The manual mask file is missing from {self.folder_path}."
        manual_mask_mat = read_mat(str(manual_mask_file_path))
        assert "regMask" in manual_mask_mat, f"The manual mask is missing from {manual_mask_file_path}."
        manual_mask = manual_mask_mat["regMask"]

        return manual_mask

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: dict):
        metadata_copy = deepcopy(metadata)
        metadata_copy = dict_deep_update(metadata_copy, self.get_metadata())
        ophys = get_module(nwbfile=nwbfile, name="ophys")

        images_container_name = metadata_copy["Ophys"]["SegmentationImages"]["name"]
        if images_container_name in ophys.data_interfaces:
            raise ValueError(f"Images container {images_container_name} already exists in the NWBFile.")

        images_container = Images(
            name=images_container_name,
            description=metadata_copy["Ophys"]["SegmentationImages"]["description"],
        )
        ophys.add(images_container)

        images_metadata = metadata_copy["Ophys"]["SegmentationImages"]["images"]
        for image_name, image_data in self.get_images_dict().items():
            image_kwargs = dict(data=image_data.T)
            image_metadata = next(
                (image_metadata for image_metadata in images_metadata if image_name == image_metadata["name"]), None
            )
            image_kwargs.update(image_metadata)
            images_container.add_image(GrayscaleImage(**image_kwargs))
