from typing import Optional

from neuroconv.datainterfaces.ophys.baseimagingextractorinterface import BaseImagingExtractorInterface
from neuroconv.tools.roiextractors import get_nwb_imaging_metadata
from neuroconv.utils import FilePathType, dict_deep_update

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
        """

        super().__init__(
            file_path=file_path,
            info_file_path=info_file_path,
            strobe_sequence_file_path=strobe_sequence_file_path,
            channel_name=channel_name,
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
