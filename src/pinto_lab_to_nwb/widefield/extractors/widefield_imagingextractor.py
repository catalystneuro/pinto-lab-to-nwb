import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from neuroconv.utils import FilePathType, FolderPathType
from pymatreader import read_mat
from roiextractors import MicroManagerTiffImagingExtractor

# todo move filtering fix to roiextractors
# Get the logger used by tifffile
tifffile_logger = logging.getLogger("tifffile")


# Define a custom filter class
class CustomWarningFilter(logging.Filter):
    def filter(self, record):
        # Filter out warnings with the specific message
        return "<tifffile.TiffTag 270 @42054>" not in record.getMessage()


# Add the custom filter to the tifffile logger
custom_filter = CustomWarningFilter()
tifffile_logger.addFilter(custom_filter)


class WidefieldImagingExtractor(MicroManagerTiffImagingExtractor):
    extractor_name = "WidefieldImaging"

    def __init__(
        self,
        folder_path: FolderPathType,
        strobe_sequence_file_path: FilePathType,
        channel_name: Optional[str] = "blue",
    ):
        """
        Specialized extractor for reading Widefield imaging experiment data.
        The extractor reads TIFF files produced via Micro-Manager and uses the strobe sequence to separate the blue and violet channels.

        Parameters
        ----------
        folder_path: FolderPathType
           The folder path that contains the multipage OME-TIF image files (.ome.tif files) and
           the 'DisplaySettings' JSON file.
        strobe_sequence_file_path: FilePathType
            The file path to the strobe sequence file. This file should contain the 'strobe_session_key' key.
        channel_name: str, optional
            The name of the channel to load the frames for. The default is 'blue'.
        """

        super().__init__(folder_path=folder_path)
        self.channel_name = channel_name

        assert Path(
            strobe_sequence_file_path
        ).exists(), f"The strobe sequence file does not exist: {strobe_sequence_file_path}"
        strobe_sequence_mat = read_mat(strobe_sequence_file_path)
        assert (
            "strobe_session_key" in strobe_sequence_mat
        ), "The strobe sequence file is missing the 'strobe_session_key' key."
        strobe_sequence_dict = strobe_sequence_mat["strobe_session_key"]
        assert (
            "strobe_sequence" in strobe_sequence_dict
        ), "The strobe sequence file is missing the 'strobe_sequence' key."
        frame_order = strobe_sequence_dict["strobe_sequence"]
        strobe_id = dict(blue=1, violet=2)[channel_name]

        self.frame_indices = np.where(frame_order == strobe_id)[0]
        self._times = strobe_sequence_dict["all_frame_times"][self.frame_indices]
        assert (
            len(frame_order) == super().get_num_frames()
        ), "The number of frames in the strobe sequence does not match the number of frames in the TIFF files."
        self._num_frames = len(self.frame_indices)

    def get_num_frames(self) -> int:
        return self._num_frames

    def get_sampling_frequency(self) -> float:
        return super().get_sampling_frequency() / 2

    def get_channel_names(self) -> List[str]:
        return [f"OpticalChannel{self.channel_name.capitalize()}"]

    def get_num_channels(self) -> int:
        return 1

    def get_video(
        self, start_frame: Optional[int] = None, end_frame: Optional[int] = None, channel: int = 0
    ) -> np.ndarray:
        start_frame = start_frame or 0
        end_frame = end_frame or self.get_num_frames()

        # Ensure the end_frame does not exceed the available frames
        last_batch = False
        if end_frame >= self.get_num_frames():
            end_frame = self.get_num_frames() - 1
            last_batch = True

        original_video_start_frame = self.frame_indices[start_frame]
        original_video_end_frame = self.frame_indices[end_frame]
        video = super().get_video(
            start_frame=original_video_start_frame,
            end_frame=original_video_end_frame + int(last_batch),
            channel=channel,
        )

        filtered_indices = (
            self.frame_indices[start_frame : end_frame + int(last_batch)] - self.frame_indices[start_frame]
        )

        return video[filtered_indices]
