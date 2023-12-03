from pathlib import Path

import numpy as np
from neuroconv import BaseDataInterface
from neuroconv.tools import get_module
from neuroconv.utils import FilePathType
from pymatreader import read_mat
from pynwb import NWBFile
from pynwb.base import Images
from pynwb.image import GrayscaleImage


class WidefieldSegmentationImagesVioletInterface(BaseDataInterface):
    """The custom interface to add the violet channel PCA mask to the NWBFile."""

    def __init__(self, violet_pca_mask_file_path: FilePathType, verbose: bool = True):
        """
        The interface to add the summary images to the NWBFile.

        Parameters
        ----------
        violet_pca_mask_file_path : FilePathType
            The file path to the violet channel PCA mask file.
        verbose : bool, default: True
        """
        super().__init__(violet_pca_mask_file_path=violet_pca_mask_file_path)
        self.violet_pca_mask_file_path = Path(violet_pca_mask_file_path)
        assert (
            self.violet_pca_mask_file_path.exists()
        ), f"The violet channel PCA mask file '{violet_pca_mask_file_path}' does not exist."
        self.verbose = verbose

        self._image_pca_violet = self._load_pca_mask()

    def _load_pca_mask(self) -> np.ndarray:
        pca_mask_violet = read_mat(str(self.violet_pca_mask_file_path))
        assert "vasc_mask" in pca_mask_violet, f"Could not find 'vasc_mask' in '{self.violet_pca_mask_file_path}'."
        pca_mask = pca_mask_violet["vasc_mask"]

        return pca_mask

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: dict):
        ophys = get_module(nwbfile=nwbfile, name="ophys")

        images_container_name = "SegmentationImagesProcessedViolet"
        if images_container_name in ophys.data_interfaces:
            raise ValueError(f"Images container {images_container_name} already exists in the NWBFile.")

        description = "Contains the PCA mask for the violet channel on the binned session image."
        images_container = Images(
            name=images_container_name,
            description=description,
        )
        ophys.add(images_container)

        pca_image_data = self._image_pca_violet.T
        images_container.add_image(
            GrayscaleImage(
                name="pca_violet", description="The PCA based mask for the violet channel.", data=pca_image_data
            )
        )
