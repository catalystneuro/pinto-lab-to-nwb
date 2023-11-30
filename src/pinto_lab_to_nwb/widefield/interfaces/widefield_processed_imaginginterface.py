from typing import Optional
from warnings import warn

import numpy as np
from neuroconv.datainterfaces.ophys.baseimagingextractorinterface import BaseImagingExtractorInterface
from neuroconv.tools.roiextractors import get_nwb_imaging_metadata
from neuroconv.utils import FilePathType, dict_deep_update
from roiextractors.extraction_tools import DtypeType

from pinto_lab_to_nwb.widefield.extractors.widefield_processed_imagingextractor import (
    WidefieldProcessedImagingExtractor,
)


class WidefieldProcessedImagingInterface(BaseImagingExtractorInterface):
    """Data Interface for ProcessedMiniscopeImagingExtractor."""

    Extractor = WidefieldProcessedImagingExtractor

    def __init__(
        self,
        file_path: FilePathType,
        info_file_path: FilePathType,
        strobe_sequence_file_path: FilePathType,
        channel_name: Optional[str] = "blue",
        convert_video_dtype_to: Optional[DtypeType] = None,
        verbose: bool = True,
    ):
        """
        Initialize reading the processed Widefield (downsampled by a factor of 8) imaging data.

        Parameters
        ----------
        file_path : FilePathType
            The path that points to downsampled imaging data saved as a Matlab file.
        info_file_path: FilePathType
            The path that points to Matlab file with information about the imaging session (e.g. 'frameRate').
        strobe_sequence_file_path: PathType
            The path that points to the strobe sequence file. This file should contain the 'strobe_session_key' key.
        channel_name: str, optional
            The name of the channel to load the frames for. The default is 'blue'.
        convert_video_dtype_to: DtypeType, optional
            The dtype to convert the video to.
        """

        super().__init__(
            file_path=file_path,
            info_file_path=info_file_path,
            strobe_sequence_file_path=strobe_sequence_file_path,
            channel_name=channel_name,
            convert_video_dtype_to=convert_video_dtype_to,
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

        imaging_plane_metadata = metadata["Ophys"]["ImagingPlane"][0]
        imaging_plane_metadata.update(
            name=f"ImagingPlane{self.channel_name.capitalize()}",
            imaging_rate=self.imaging_extractor.get_sampling_frequency(),
        )
        metadata["Ophys"]["OnePhotonSeries"][0].update(
            name=f"OnePhotonSeriesProcessed{self.channel_name.capitalize()}",
            description="Processed imaging data from one-photon excitation microscopy.",
            imaging_plane=imaging_plane_metadata["name"],
            binning=64,  # downsampled by a factor of 8 (x,y)
        )

        return metadata

    def align_by_interpolation(self, unaligned_timestamps: np.ndarray, aligned_timestamps: np.ndarray) -> None:
        if len(unaligned_timestamps) == len(aligned_timestamps):
            return super().align_by_interpolation(unaligned_timestamps=unaligned_timestamps,
                                                  aligned_timestamps=aligned_timestamps)

        if len(unaligned_timestamps) > len(aligned_timestamps):
            # Extract the extra timestamps
            extra_timestamps = unaligned_timestamps[len(aligned_timestamps):]
            # Interpolate a single value for the extra timestamps
            interpolated_values_extra = np.interp(x=extra_timestamps, xp=aligned_timestamps,
                                                  fp=unaligned_timestamps[:len(aligned_timestamps)])
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