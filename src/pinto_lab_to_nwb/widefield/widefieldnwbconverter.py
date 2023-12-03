"""Primary NWBConverter class for this dataset."""
from pathlib import Path
from typing import Optional, Dict

import numpy as np
from natsort import natsorted
from ndx_pinto_metadata import SubjectExtension
from neuroconv import NWBConverter
from pynwb import NWBFile

from pinto_lab_to_nwb.behavior.interfaces import ViRMENBehaviorInterface, ViRMENWidefieldTimeAlignedBehaviorInterface
from pinto_lab_to_nwb.widefield.utils import load_motion_correction_data
from pinto_lab_to_nwb.widefield.utils.motion_correction import add_motion_correction
from pinto_lab_to_nwb.widefield.interfaces import (
    WidefieldImagingInterface,
    WidefieldProcessedImagingInterface,
    WidefieldProcessedSegmentationinterface,
    WidefieldSegmentationImagesBlueInterface,
    WidefieldSegmentationImagesVioletInterface,
)


class WideFieldNWBConverter(NWBConverter):
    """Primary conversion class for Widefield imaging dataset."""

    data_interface_classes = dict(
        ImagingBlue=WidefieldImagingInterface,
        ImagingViolet=WidefieldImagingInterface,
        ProcessedImagingBlue=WidefieldProcessedImagingInterface,
        ProcessedImagingViolet=WidefieldProcessedImagingInterface,
        SegmentationProcessedBlue=WidefieldProcessedSegmentationinterface,
        SummaryImagesBlue=WidefieldSegmentationImagesBlueInterface,
        SummaryImagesViolet=WidefieldSegmentationImagesVioletInterface,
        BehaviorViRMEN=ViRMENBehaviorInterface,
        BehaviorViRMENWidefieldTimeAligned=ViRMENWidefieldTimeAlignedBehaviorInterface,
    )

    def __init__(self, source_data: Dict[str, dict], verbose: bool = True):
        super().__init__(source_data, verbose)

        # Load motion correction data
        imaging_interface = self.data_interface_objects["ImagingBlue"]
        imaging_folder_path = imaging_interface.source_data["folder_path"]
        imaging_folder_name = Path(imaging_folder_path).stem
        motion_correction_mat_files = natsorted(Path(imaging_folder_path).glob(f"{imaging_folder_name}*mcorr_1.mat"))
        assert motion_correction_mat_files, f"No motion correction files found in {imaging_folder_path}."
        self._motion_correction_data = load_motion_correction_data(file_paths=motion_correction_mat_files)

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata, conversion_options: Optional[dict] = None) -> None:
        super().add_to_nwbfile(nwbfile=nwbfile, metadata=metadata, conversion_options=conversion_options)

        # Add subject (from extension)
        if metadata["SubjectExtension"] is not None:
            nwbfile.subject = SubjectExtension(**metadata["SubjectExtension"])

        # Add motion correction for blue and violet frames
        imaging_interface_names = ["ImagingBlue", "ImagingViolet"]
        for interface_name in imaging_interface_names:
            photon_series_index = conversion_options[interface_name]["photon_series_index"]
            one_photon_series_name = metadata["Ophys"]["OnePhotonSeries"][photon_series_index]["name"]

            imaging_interface = self.data_interface_objects[interface_name]
            frame_indices = imaging_interface.imaging_extractor.frame_indices
            # filter motion correction for blue/violet frames
            motion_correction = self._motion_correction_data[frame_indices, :]
            if interface_name in conversion_options:
                if "stub_test" in conversion_options[interface_name]:
                    if conversion_options[interface_name]["stub_test"]:
                        num_frames = 100
                        motion_correction = motion_correction[:num_frames, :]

            add_motion_correction(
                nwbfile=nwbfile,
                motion_correction_series=motion_correction,
                one_photon_series_name=one_photon_series_name,
                convert_to_dtype=np.uint16,
            )

    def temporally_align_data_interfaces(self):
        frame_time_aligned_behavior_interface = self.data_interface_objects["BehaviorViRMENWidefieldTimeAligned"]
        blue_frames_timestamps = frame_time_aligned_behavior_interface.get_timestamps()
        imaging_interface = self.data_interface_objects["ImagingBlue"]

        imaging_interface.set_aligned_timestamps(aligned_timestamps=blue_frames_timestamps)

        downsampled_imaging_interface = self.data_interface_objects["ProcessedImagingBlue"]
        downsampled_imaging_interface.set_aligned_timestamps(aligned_timestamps=blue_frames_timestamps)

        # For violet the interpolation doesn't work yet, the first 8 values are 0.0 after interpolation
        violet_interface = self.data_interface_objects["ImagingViolet"]
        violet_interface.align_by_interpolation(
            aligned_timestamps=blue_frames_timestamps, unaligned_timestamps=violet_interface.imaging_extractor._times
        )
