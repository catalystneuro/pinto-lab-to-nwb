from typing import Optional

import numpy as np
from hdmf.backends.hdf5 import H5DataIO
from neuroconv import BaseTemporalAlignmentInterface
from neuroconv.tools import get_module
from neuroconv.utils import FilePathType, calculate_regular_series_rate
from pymatreader import read_mat
from pynwb import NWBFile, TimeSeries
from pynwb.behavior import Position, CompassDirection


class ViRMENTemporalAlignmentBehaviorInterface(BaseTemporalAlignmentInterface):
    """
    The interface for aligning the timestamps of Widefield (blue frames) and Two Photon imaging data.
    """

    def __init__(
            self,
            file_path: FilePathType,
            sync_data_struct_name: str,
            im_frame_timestamps_name: Optional[str] = None,
            verbose: bool = True,
    ):
        """
        Parameters
        ----------
        file_path: FilePathType
            The path to the .mat file containing the sync data.
        sync_data_struct_name: str
            The name of the sync data struct in the .mat file. (e.g. "wf_behav_sync_data")
        im_frame_timestamps_name: Optional[str]
            The name of the variable in the .mat file that contains the aligned timestamps for the imaging frames.
            (e.g. "im_frame_timestamps")
        """
        self.verbose = verbose
        super().__init__(file_path=file_path)
        self.im_frame_timestamps_name = im_frame_timestamps_name or "im_frame_timestamps"
        mat_dict = read_mat(filename=file_path)
        assert sync_data_struct_name in mat_dict, f"'{sync_data_struct_name}' not in file '{file_path}'."
        self._mat_dict = mat_dict[sync_data_struct_name]
        self.sync_data_struct_name = sync_data_struct_name
        self._times = None

    def get_original_timestamps(self) -> np.ndarray:
        mat_dict = read_mat(filename=self.source_data["file_path"])[self.sync_data_struct_name]
        assert self.im_frame_timestamps_name in mat_dict, (
            f"'{self.im_frame_timestamps_name}' not in in file '{self.source_data['file_path']}'."
        )
        return mat_dict[self.im_frame_timestamps_name]

    def get_timestamps(self) -> np.ndarray:
        return self._times if self._times is not None else self.get_original_timestamps()

    def set_aligned_timestamps(self, aligned_timestamps: np.ndarray):
        self._times = aligned_timestamps

    def add_position(self, nwbfile: NWBFile):
        # Processed position, velocity, viewAngle
        position_by_im_frame_name = "position_by_im_frame"
        if position_by_im_frame_name not in self._mat_dict:
            return

        behavior = get_module(nwbfile, "behavior", "contains processed behavioral data")
        if "Position" not in behavior.data_interfaces:
            behavior.add(Position(name="Position"))

        position_obj = behavior.data_interfaces["Position"]

        # reference "BehavSync2P.m" script where the position is accessed as:
        # 'position_by_im_frame' is the average [x y theta] position by imaging frame
        # 'velocity_by_im_frame' is the average [x y theta] velocity by imaging frame
        position = self._mat_dict[position_by_im_frame_name]  # x,y,z

        timestamps = self.get_timestamps()
        assert (
                len(timestamps) == position.shape[0]
        ), f"The length of timestamps ({len(timestamps)}) must match the length of position ({position.shape[0]})."

        reference_frame = "unknown"
        if "SpatialSeries" in position_obj.spatial_series:
            reference_frame = position_obj.spatial_series["SpatialSeries"].reference_frame
        spatial_series_name = "SpatialSeriesByImFrame"
        position_obj.create_spatial_series(
            name=spatial_series_name,
            data=H5DataIO(position[:, :-1], compression="gzip"),
            description="The average x, y position by imaging frame.",
            reference_frame=reference_frame,
            timestamps=H5DataIO(timestamps, compression="gzip"),
        )

        if "ViewAngle" not in behavior.data_interfaces:
            behavior.add(CompassDirection(name="ViewAngle"))
        view_angle_obj = behavior.data_interfaces["ViewAngle"]
        view_angle_data = position[:, -1]
        view_angle_obj.create_spatial_series(
            name="PositionViewAngleByImFrame",
            data=H5DataIO(view_angle_data, compression="gzip"),
            description="The average view angle of position by imaging frame.",
            reference_frame=reference_frame,
            unit="degrees",
            timestamps=position_obj[spatial_series_name],
        )

        if "velocity_by_im_frame" in self._mat_dict:
            velocity = self._mat_dict["velocity_by_im_frame"]
            velocity_data = velocity[:, :-1]
            view_angle_velocity_data = velocity_data[:, -1]

            velocity_ts = TimeSeries(
                name="VelocityByImFrame",
                data=H5DataIO(velocity_data, compression="gzip"),
                description="The average x, y velocity by imaging frame.",
                unit="m/s",
                conversion=0.01,
                timestamps=position_obj[spatial_series_name],
            )
            behavior.add(velocity_ts)

            view_angle_obj.create_spatial_series(
                name="VelocityViewAngleByImFrame",
                data=H5DataIO(view_angle_velocity_data, compression="gzip"),
                description="The average view angle of velocity by imaging frame.",
                reference_frame=reference_frame,
                unit="degrees",
                timestamps=position_obj[spatial_series_name],
            )

        if "sensordots_by_im_frame" in self._mat_dict:
            sensor_dots = self._mat_dict["sensordots_by_im_frame"]
            sensor_dots_ts = TimeSeries(
                name="SensorDotsByImFrame",
                data=H5DataIO(sensor_dots, compression="gzip"),
                description="The sensordots averaged by imaging frame.",
                unit="a.u.",
                timestamps=position_obj[spatial_series_name],
            )
            behavior.add(sensor_dots_ts)

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata) -> None:
        self.add_position(nwbfile=nwbfile)
