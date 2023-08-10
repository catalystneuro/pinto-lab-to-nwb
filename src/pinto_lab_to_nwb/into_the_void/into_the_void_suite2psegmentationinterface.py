from pathlib import Path
from typing import Optional

from neuroconv.datainterfaces import Suite2pSegmentationInterface
from neuroconv.tools.roiextractors import add_plane_segmentation, add_fluorescence_traces, add_summary_images
from neuroconv.utils import FolderPathType
from pynwb import NWBFile


class IntoTheVoidSuite2pSegmentationInterface(Suite2pSegmentationInterface):
    """Data interface for Suite2pSegmentationExtractor."""

    ExtractorName = "Suite2pSegmentationExtractor"

    @classmethod
    def get_streams(cls, folder_path: FolderPathType):
        from natsort import natsorted

        folder_path = Path(folder_path)
        plane_folders = natsorted(folder_path.glob("plane*"))
        streams = [plane.name for plane in plane_folders]
        return streams

    def __init__(self, folder_path: FolderPathType, stream_name: Optional[str] = None, verbose: bool = True):
        """

        Parameters
        ----------
        folder_path : FolderPathType
        combined : bool, default: False
        plane_no : int, default: 0
        verbose : bool, default: True
        """
        streams = self.get_streams(folder_path=folder_path)
        if stream_name is None:
            if len(streams) > 1:
                raise ValueError(
                    "More than one recording stream is detected! Please specify which stream you wish to load with the `stream_name` argument. "
                    "To see what streams are available, call `IntoTheVoidSuite2pSegmentationInterface.get_streams(folder_path=...)`."
                )
            stream_name = streams[0]

        plane_no = int("".join(filter(str.isdigit, stream_name)))
        self._stream_name = plane_no
        self.folder_path = folder_path

        super().__init__(folder_path=folder_path, combined=False, plane_no=plane_no)
        self.verbose = verbose

    def get_metadata(self) -> dict:
        metadata = super().get_metadata()

        device_metadata = metadata["Ophys"]["Device"][0]
        device_name = "BrukerFluorescenceMicroscope"
        device_metadata.update(name=device_name)

        streams = self.get_streams(folder_path=self.folder_path)
        imaging_plane_metadata = metadata["Ophys"]["ImagingPlane"][0]
        imaging_plane_metadata.update(device=device_name)
        plane_segmentation_metadata = metadata["Ophys"]["ImageSegmentation"]["plane_segmentations"][0]
        fluorescence_metadata = metadata["Ophys"]["Fluorescence"]
        roi_response_series_metadata = fluorescence_metadata["roi_response_series"]
        df_over_f_metadata = metadata["Ophys"]["DfOverF"]
        df_over_f_traces = df_over_f_metadata["roi_response_series"]
        if len(streams) > 1:
            imaging_plane_name = f"ImagingPlane{self._stream_name}"
            imaging_plane_metadata.update(name=imaging_plane_name)
            plane_segmentation_metadata.update(
                name=f"PlaneSegmentation_Plane{self._stream_name}",
            )
            if self._stream_name != 0:
                for trace in roi_response_series_metadata:
                    trace.update(name=f"{trace['name']}_Plane{self._stream_name}")
                for trace in df_over_f_traces:
                    trace.update(name=f"{trace['name']}_Plane{self._stream_name}")

        return metadata

    def add_to_nwbfile(
        self,
        nwbfile: NWBFile,
        metadata: Optional[dict] = None,
        stub_test: bool = False,
        stub_frames: int = 100,
        plane_index: int = 0,
        include_roi_centroids: bool = True,
        include_roi_acceptance: bool = True,
        mask_type: Optional[str] = "image",  # Literal["image", "pixel", "voxel"]
        iterator_options: Optional[dict] = None,
        compression_options: Optional[dict] = None,
    ):
        if stub_test:
            stub_frames = min([stub_frames, self.segmentation_extractor.get_num_frames()])
            segmentation_extractor = self.segmentation_extractor.frame_slice(start_frame=0, end_frame=stub_frames)
        else:
            segmentation_extractor = self.segmentation_extractor

        # PlaneSegmentation:
        add_plane_segmentation(
            segmentation_extractor=segmentation_extractor,
            nwbfile=nwbfile,
            metadata=metadata,
            include_roi_centroids=include_roi_centroids,
            include_roi_acceptance=include_roi_acceptance,
            plane_segmentation_index=plane_index,
            mask_type=mask_type,
            iterator_options=iterator_options,
            compression_options=compression_options,
        )

        # Add fluorescence traces:
        add_fluorescence_traces(
            segmentation_extractor=segmentation_extractor,
            nwbfile=nwbfile,
            metadata=metadata,
            plane_index=plane_index,
            iterator_options=iterator_options,
            compression_options=compression_options,
        )

        # Adding summary images (mean and correlation)
        images_set_name = "SegmentationImages" if plane_index == 0 else f"SegmentationImages_Plane{self._stream_name}"
        add_summary_images(
            nwbfile=nwbfile,
            segmentation_extractor=segmentation_extractor,
            images_set_name=images_set_name,
        )
