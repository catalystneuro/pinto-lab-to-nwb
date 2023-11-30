from typing import Optional
from warnings import warn

import numpy as np
from dateutil.parser import parse
from neuroconv.datainterfaces.ophys.baseimagingextractorinterface import BaseImagingExtractorInterface
from neuroconv.tools.roiextractors import get_nwb_imaging_metadata
from neuroconv.utils import FolderPathType, dict_deep_update, FilePathType

from pinto_lab_to_nwb.widefield.extractors.widefield_imagingextractor import WidefieldImagingExtractor


class WidefieldImagingInterface(BaseImagingExtractorInterface):
    """Data Interface for WidefieldImagingExtractor."""

    Extractor = WidefieldImagingExtractor

    def __init__(
        self,
        folder_path: FolderPathType,
        strobe_sequence_file_path: FilePathType,
        channel_name: Optional[str] = "blue",
        verbose: bool = True,
    ):
        """
        Data Interface for WidefieldImagingExtractor.

        Parameters
        ----------
        folder_path : FolderPathType
            The folder path that contains the OME-TIF image files (.ome.tif files) and
           the 'DisplaySettings' JSON file.
        strobe_sequence_file_path: FilePathType
            The file path to the strobe sequence file. This file should contain the 'strobe_session_key' key.
        channel_name: str, optional
            The name of the channel to load the frames for. The default is 'blue'.
        verbose : bool, default: True
        """
        super().__init__(
            folder_path=folder_path, strobe_sequence_file_path=strobe_sequence_file_path, channel_name=channel_name
        )
        self.channel_name = channel_name
        self.verbose = verbose

    def get_metadata_schema(self) -> dict:
        metadata_schema = super().get_metadata_schema(photon_series_type="OnePhotonSeries")
        return metadata_schema

    def get_metadata(self) -> dict:
        metadata = super().get_metadata()
        default_metadata = get_nwb_imaging_metadata(self.imaging_extractor, photon_series_type="OnePhotonSeries")
        metadata = dict_deep_update(metadata, default_metadata)
        # Remove default TwoPhotonSeries metadata to only contain metadata for OnePhotonSeries
        metadata["Ophys"].pop("TwoPhotonSeries", None)

        micromanager_metadata = self.imaging_extractor.micromanager_metadata
        session_start_time = parse(micromanager_metadata["Summary"]["StartTime"])
        metadata["NWBFile"].update(session_start_time=session_start_time)

        imaging_plane_metadata = metadata["Ophys"]["ImagingPlane"][0]
        imaging_plane_name = imaging_plane_metadata["name"] + self.channel_name.capitalize()
        imaging_plane_metadata.update(
            name=imaging_plane_name,
            imaging_rate=self.imaging_extractor.get_sampling_frequency(),
        )
        one_photon_series_metadata = metadata["Ophys"]["OnePhotonSeries"][0]
        one_photon_series_name = one_photon_series_metadata["name"] + self.channel_name.capitalize()
        one_photon_series_metadata.update(
            name=one_photon_series_name,
            imaging_plane=imaging_plane_name,
        )

        return metadata

    def align_by_interpolation(self, unaligned_timestamps: np.ndarray, aligned_timestamps: np.ndarray) -> None:
        if len(unaligned_timestamps) == len(aligned_timestamps):
            return super().align_by_interpolation(unaligned_timestamps=unaligned_timestamps, aligned_timestamps=aligned_timestamps)

        if len(unaligned_timestamps) > len(aligned_timestamps):
            # Extract the extra timestamps
            extra_timestamps = unaligned_timestamps[len(aligned_timestamps):]
            # Interpolate a single value for the extra timestamps
            interpolated_values_extra = np.interp(x=extra_timestamps, xp=aligned_timestamps, fp=unaligned_timestamps[:len(aligned_timestamps)])
            interpolated_values_matched = np.interp(
                x=self.get_timestamps()[:len(aligned_timestamps)],
                xp=aligned_timestamps,
                fp=unaligned_timestamps[:len(aligned_timestamps)],
            )
            # Combine the results
            interpolated_values = np.concatenate([interpolated_values_matched, interpolated_values_extra])

        else:
            values_to_repeat = len(unaligned_timestamps) - len(aligned_timestamps)
            # repeat the last value of unaligned_timestamps to match the length of aligned_timestamps
            unaligned_timestamps = np.append(unaligned_timestamps, unaligned_timestamps[values_to_repeat:])
            interpolated_values = np.interp(
                x=self.get_timestamps(),
                xp=aligned_timestamps,
                fp=unaligned_timestamps,
            )

        if any(np.diff(interpolated_values) == 0):
            warn("Interpolated timestamps are not unique. Falling back to using the unaligned timestamps.")
            return self.set_aligned_timestamps(aligned_timestamps=unaligned_timestamps)

        return self.set_aligned_timestamps(aligned_timestamps=interpolated_values)


