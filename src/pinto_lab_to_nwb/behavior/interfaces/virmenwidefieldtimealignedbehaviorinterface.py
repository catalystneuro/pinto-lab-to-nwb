import numpy as np
from hdmf.backends.hdf5 import H5DataIO
from neuroconv import BaseTemporalAlignmentInterface
from neuroconv.tools import get_module
from neuroconv.utils import FilePathType, calculate_regular_series_rate
from pymatreader import read_mat
from pynwb import NWBFile, TimeSeries
from pynwb.behavior import Position, CompassDirection


class ViRMENWidefieldTimeAlignedBehaviorInterface(BaseTemporalAlignmentInterface):
    """
    Behavior interface for Widefield conversion.
    This interface is used to align the timestamps of the Widefield imaging data (blue frames).
    """

    def __init__(self, file_path: FilePathType, verbose: bool = True):
        self.verbose = verbose
        super().__init__(file_path=file_path)
        self._mat_dict = read_mat(filename=file_path)
        assert "wf_behav_sync_data" in self._mat_dict, f"wf_behav_sync_data not in file '{file_path}'."
        self._times = None

    def get_original_timestamps(self) -> np.ndarray:
        return self._mat_dict["wf_behav_sync_data"]["im_frame_timestamps"]

    def get_timestamps(self) -> np.ndarray:
        return self._times if self._times is not None else self.get_original_timestamps()

    def set_aligned_timestamps(self, aligned_timestamps: np.ndarray):
        self._times = aligned_timestamps

    def add_position(self, nwbfile: NWBFile):
        # Processed position, velocity, viewAngle
        if "position_by_im_frame" not in self._mat_dict["wf_behav_sync_data"]:
            return

        behavior = get_module(nwbfile, "behavior", "contains processed behavioral data")
        if "Position" not in behavior.data_interfaces:
            behavior.add(Position(name="Position"))

        position_obj = behavior.data_interfaces["Position"]

        position = self._mat_dict["wf_behav_sync_data"]["position_by_im_frame"]  # x,y,z

        timestamps = self.get_timestamps()
        rate = calculate_regular_series_rate(timestamps)
        if rate:
            timing_kwargs = dict(rate=rate, starting_time=timestamps[0])
        else:
            assert (
                len(timestamps) == position.shape[0]
            ), f"The length of timestamps ({len(timestamps)}) must match the length of position ({position.shape[0]})."
            timing_kwargs = dict(timestamps=H5DataIO(timestamps, compression="gzip"))

        reference_frame = "unknown"
        if "SpatialSeries" in position_obj.spatial_series:
            reference_frame = position_obj.spatial_series["SpatialSeries"].reference_frame
        position_obj.create_spatial_series(
            name="SpatialSeriesByImFrame",
            data=H5DataIO(position[:, :-1], compression="gzip"),
            description="The position of the animal averaged over the iterations for each frame.",
            reference_frame=reference_frame,
            **timing_kwargs,
        )

        view_angle_data = position[:, -1]
        if "ViewAngle" not in behavior.data_interfaces:
            behavior.add(CompassDirection(name="ViewAngle"))
        view_angle_obj = behavior.data_interfaces["ViewAngle"]
        view_angle_obj.create_spatial_series(
            name="PositionViewAngleByImFrame",
            data=H5DataIO(view_angle_data, compression="gzip"),
            description="The view angle averaged over the iterations for each frame.",
            reference_frame=reference_frame,
            **timing_kwargs,
        )

        if "velocity_by_im_frame" in self._mat_dict["wf_behav_sync_data"]:
            velocity = self._mat_dict["wf_behav_sync_data"]["velocity_by_im_frame"]
            velocity_data = velocity[:, :-1]
            view_angle_velocity_data = velocity_data[:, -1]

            velocity_ts = TimeSeries(
                name="VelocityByImFrame",
                data=H5DataIO(velocity_data, compression="gzip"),
                description="The velocity of the animal averaged over the iterations for each frame.",
                unit="m/s",
                conversion=0.01,
                **timing_kwargs,
            )
            behavior.add(velocity_ts)

            view_angle_obj.create_spatial_series(
                name="VelocityViewAngleByImFrame",
                data=H5DataIO(view_angle_velocity_data, compression="gzip"),
                description="The view angle averaged over the iterations for each frame.",
                reference_frame=reference_frame,
                **timing_kwargs,
            )

        if "sensordots_by_im_frame" in self._mat_dict["wf_behav_sync_data"]:
            sensor_dots = self._mat_dict["wf_behav_sync_data"]["sensordots_by_im_frame"]
            sensor_dots_ts = TimeSeries(
                name="SensorDotsByImFrame",
                data=H5DataIO(sensor_dots, compression="gzip"),
                description="The sensordots averaged over the iterations for each frame.",
                unit="a.u.",
                **timing_kwargs,
            )
            behavior.add(sensor_dots_ts)

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata) -> None:
        self.add_position(nwbfile=nwbfile)
