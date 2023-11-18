"""Primary NWBConverter class for this dataset."""
from pathlib import Path
from typing import Optional, Dict

from natsort import natsorted
from neuroconv import NWBConverter
from neuroconv.tools import get_module
from pynwb import NWBFile, TimeSeries

from pinto_lab_to_nwb.widefield.interfaces import WidefieldImagingInterface, WidefieldProcessedImagingInterface
from pinto_lab_to_nwb.widefield.utils import load_motion_correction_data


class WideFieldNWBConverter(NWBConverter):
    """Primary conversion class for Widefield imaging dataset."""

    data_interface_classes = dict(
        ImagingBlue=WidefieldImagingInterface,
        ImagingViolet=WidefieldImagingInterface,
        ProcessedImagingBlue=WidefieldProcessedImagingInterface,
        ProcessedImagingViolet=WidefieldProcessedImagingInterface,
    )

    def __init__(self, source_data: Dict[str, dict], verbose: bool = True):
        super().__init__(source_data, verbose)

        # Load motion correction data
        imaging_interface = self.data_interface_objects["ImagingBlue"]
        imaging_folder_path = imaging_interface.source_data["folder_path"]
        motion_correction_mat_files = natsorted(Path(imaging_folder_path).glob("*mcorr_1.mat"))
        assert motion_correction_mat_files, f"No motion correction files found in {imaging_folder_path}."
        self._motion_correction_data = load_motion_correction_data(file_paths=motion_correction_mat_files)

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata, conversion_options: Optional[dict] = None) -> None:
        super().add_to_nwbfile(nwbfile=nwbfile, metadata=metadata, conversion_options=conversion_options)

        # Add motion correction for blue and violet frames
        interface_names = ["ImagingBlue", "ImagingViolet"]
        for interface_name in interface_names:
            channel_name = interface_name.replace("Imaging", "")
            motion_correction_series_name = f"MotionCorrectionSeries{channel_name}"
            assert (
                motion_correction_series_name not in nwbfile.acquisition
            ), f"Motion correction series '{motion_correction_series_name}' already exists in NWBFile."
            imaging_interface = self.data_interface_objects[interface_name]
            frame_indices = imaging_interface.imaging_extractor.frame_indices

            motion_correction = self._motion_correction_data[frame_indices, :]
            num_frames = imaging_interface.imaging_extractor.get_num_frames()
            if interface_name in conversion_options:
                if "stub_test" in conversion_options[interface_name]:
                    if conversion_options[interface_name]["stub_test"]:
                        num_frames = 100
                        motion_correction = motion_correction[:num_frames, :]

            assert (
                motion_correction.shape[0] == num_frames
            ), f"The number of frames for motion correction ({motion_correction.shape[0]}) does not match the number of frames ({num_frames}) from the {interface_name} imaging interface."

            one_photon_series = nwbfile.acquisition[f"OnePhotonSeries{channel_name}"]
            yx_time_series = TimeSeries(
                name=motion_correction_series_name,
                description=f"The yx shifts for the {channel_name.lower()} frames.",
                data=motion_correction,
                unit="px",
                timestamps=one_photon_series.timestamps,
            )
            ophys = get_module(nwbfile, "ophys")
            ophys.add(yx_time_series)
