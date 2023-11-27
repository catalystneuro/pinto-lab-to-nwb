from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
from lazy_ops import DatasetView
from neuroconv.utils import FilePathType
from pymatreader import read_mat
from roiextractors import ImagingExtractor
from roiextractors.extraction_tools import DtypeType


class WidefieldProcessedImagingExtractor(ImagingExtractor):
    extractor_name = "WidefieldProcessedImaging"

    def __init__(
        self,
        file_path: FilePathType,
        info_file_path: FilePathType,
        strobe_sequence_file_path: FilePathType,
        channel_name: Optional[str] = "blue",
        convert_video_dtype_to: Optional[DtypeType] = None,
    ):
        """
        The ImagingExtractor for loading the downsampled imaging data for the Widefield session.

        Parameters
        ----------
        file_path: PathType
           The path that points to downsampled imaging data saved as a Matlab file.
        info_file_path: PathType
            The path that points to Matlab file with information about the imaging session (e.g. 'frameRate').
        strobe_sequence_file_path: PathType
            The path that points to the strobe sequence file. This file should contain the 'strobe_session_key' key.
        channel_name: str, optional
            The name of the channel to load the frames for. The default is 'blue'.
        convert_video_dtype_to: DtypeType, optional
            The dtype to convert the video to.
        """
        import h5py

        super().__init__(file_path=file_path)
        self.convert_video_dtype_to = convert_video_dtype_to or np.uint16

        file = h5py.File(file_path, "r")
        expected_struct_name = "rawf"
        assert expected_struct_name in file.keys(), f"'{expected_struct_name}' is not in {file_path}."
        self._video = DatasetView(file[expected_struct_name])

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
        assert channel_name in ["blue", "violet"], f"'channel_name' must be 'blue' or 'violet'."
        self.channel_name = channel_name
        strobe_id = dict(blue=1, violet=2)[channel_name]

        self.frame_indices = np.where(frame_order == strobe_id)[0]
        self._times = strobe_sequence_dict["all_frame_times"][self.frame_indices]
        assert (
            len(frame_order) == self._video.shape[0]
        ), f"The number of frames in the strobe sequence does not match the number of frames in {file_path}."
        self._num_frames = len(self.frame_indices)

        # check image axis order is height by width
        self._width, self._height = self._video.shape[1:]
        self._dtype = self._video.dtype

        # h5py not works with this file (matlab 5.0 file)
        info_file = read_mat(info_file_path)
        assert "info" in info_file, f"'info' is not in {file_path}."
        self._sampling_frequency = float(info_file["info"]["frameRate"]) / 2  # because of blue/violet strobe

    def get_image_size(self) -> Tuple[int, int]:
        return self._height, self._width

    def get_num_frames(self) -> int:
        return self._num_frames

    def get_sampling_frequency(self) -> float:
        return self._sampling_frequency

    def get_channel_names(self) -> List[str]:
        return [f"OpticalChannel{self.channel_name.capitalize()}"]

    def get_num_channels(self) -> int:
        return 1

    def get_dtype(self) -> DtypeType:
        return self._dtype

    def get_video(
        self, start_frame: Optional[int] = None, end_frame: Optional[int] = None, channel: int = 0
    ) -> np.ndarray:
        if start_frame is not None and end_frame is not None and start_frame == end_frame:
            video_start_frame = int(self.frame_indices[start_frame])
            return self._video[video_start_frame].transpose((1, 0)).astype(dtype=self.convert_video_dtype_to)

        start_frame = start_frame or 0
        end_frame = end_frame or self.get_num_frames()

        # Ensure the end_frame does not exceed the available frames
        last_batch = False
        if end_frame >= self.get_num_frames():
            end_frame = self.get_num_frames() - 1
            last_batch = True

        original_video_start_frame = self.frame_indices[start_frame]
        original_video_end_frame = self.frame_indices[end_frame]

        video = (
            self._video.lazy_slice[original_video_start_frame : original_video_end_frame + int(last_batch), ...]
            .lazy_transpose(axis_order=(0, 2, 1))
            .dsetread()
            .astype(dtype=self.convert_video_dtype_to)
        )

        filtered_indices = (
            self.frame_indices[start_frame : end_frame + int(last_batch)] - self.frame_indices[start_frame]
        )

        return video[filtered_indices]
