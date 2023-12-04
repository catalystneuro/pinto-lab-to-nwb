from typing import Optional

from neuroconv.datainterfaces.ophys.basesegmentationextractorinterface import BaseSegmentationExtractorInterface
from neuroconv.tools import get_module
from neuroconv.utils import FilePathType
from pynwb import NWBFile

from pinto_lab_to_nwb.widefield.extractors.widefield_processed_segmentationextractor import (
    WidefieldProcessedSegmentationExtractor,
)


class WidefieldProcessedSegmentationinterface(BaseSegmentationExtractorInterface):
    """Data interface for WidefieldProcessedSegmentationExtractor."""

    Extractor = WidefieldProcessedSegmentationExtractor

    def __init__(
        self,
        info_mat_file_path: FilePathType,
        roi_from_ref_mat_file_path: FilePathType,
        vasculature_mask_file_path: FilePathType,
        blue_pca_mask_file_path: FilePathType,
        verbose: bool = True,
    ):
        """

        Parameters
        ----------
        info_mat_file_path : FilePathType
            The file path to the 'info.mat' file.
        roi_from_ref_mat_file_path : FilePathType
            The file that contains the Allen area label of each pixel mapped onto the reference image of the mouse and registered to the session.
        vasculature_mask_file_path: FilePathType
            The file that contains the vasculature mask on the downsampled (binned) session image.
        blue_pca_mask_file_path: FilePathType
            The file that contains the PCA mask for the blue channel.
        verbose : bool, default: True
        """
        super().__init__(
            info_mat_file_path=info_mat_file_path,
            roi_from_ref_mat_file_path=roi_from_ref_mat_file_path,
            vasculature_mask_file_path=vasculature_mask_file_path,
            blue_pca_mask_file_path=blue_pca_mask_file_path,
        )
        self.verbose = verbose

    def get_metadata(self) -> dict:
        metadata = super().get_metadata()

        imaging_plane_name = "ImagingPlaneBlue"
        metadata["Ophys"]["ImagingPlane"][0].update(name=imaging_plane_name)
        plane_segmentation_metadata = metadata["Ophys"]["ImageSegmentation"]["plane_segmentations"][0]
        default_plane_segmentation_name = plane_segmentation_metadata["name"]
        plane_segmentation_name = "PlaneSegmentationProcessedBlue"
        plane_segmentation_metadata.update(
            name=plane_segmentation_name,
            imaging_plane=imaging_plane_name,
        )
        summary_images_metadata = metadata["Ophys"]["SegmentationImages"]
        _ = summary_images_metadata.pop(default_plane_segmentation_name)
        images_metadata = dict(
            vasculature=dict(
                name="vasculature", description="The contrast based vasculature mask for the blue channel."
            ),
            pca_blue=dict(name="pca_blue", description="The PCA based mask for the blue channel."),
        )

        metadata["Ophys"]["SegmentationImages"].update({"PlaneSegmentationProcessedBlue": images_metadata})

        return metadata

    def add_to_nwbfile(
        self,
        nwbfile: NWBFile,
        metadata: Optional[dict] = None,
        stub_test: bool = False,
        stub_frames: int = 100,
        include_roi_centroids: bool = True,
        include_roi_acceptance: bool = True,
        mask_type: Optional[str] = "image",
        plane_segmentation_name: Optional[str] = "PlaneSegmentationProcessedBlue",
        iterator_options: Optional[dict] = None,
        compression_options: Optional[dict] = None,
    ):
        super().add_to_nwbfile(
            nwbfile=nwbfile,
            metadata=metadata,
            stub_test=stub_test,
            stub_frames=stub_frames,
            include_roi_acceptance=include_roi_acceptance,
            mask_type=mask_type,
            plane_segmentation_name=plane_segmentation_name,
            iterator_options=iterator_options,
            compression_options=compression_options,
        )

        # Add Allen area labels as a column to the plane segmentation table
        ophys = get_module(nwbfile, "ophys")
        image_segmentation = ophys.get("ImageSegmentation")
        plane_segmentation = image_segmentation.plane_segmentations[plane_segmentation_name]
        locations = self.segmentation_extractor._roi_labels
        assert len(plane_segmentation.id) == len(
            locations
        ), "The number of ROIs does not match the number of Allen area labels."
        plane_segmentation.add_column(
            name="location",
            description="The Allen area labels for each ROI.",
            data=locations,
        )
