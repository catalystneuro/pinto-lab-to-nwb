"""Primary class for converting experiment-specific behavior."""
from datetime import datetime, timedelta

import numpy as np
import scipy
from hdmf.backends.hdf5 import H5DataIO
from neuroconv.tools.nwb_helpers import get_module
from pynwb import TimeSeries
from pynwb.behavior import Position, CompassDirection, BehavioralEvents
from scipy.io import loadmat
from scipy.io.matlab import mat_struct
from pynwb.file import NWBFile

from neuroconv.utils import FilePathType, dict_deep_update
from neuroconv.basedatainterface import BaseDataInterface

from ndx_tank_metadata import RigExtension, LabMetaDataExtension, MazeExtension

from pinto_lab_to_nwb.behavior.utils import convert_mat_file_to_dict


class ViRMENBehaviorInterface(BaseDataInterface):
    """Behavior interface for into_the_void conversion"""

    def __init__(self, file_path: FilePathType, verbose: bool = True):
        self.verbose = verbose
        super().__init__(file_path=file_path)
        self._mat_dict = convert_mat_file_to_dict(mat_file_name=file_path)

    def get_metadata(self):
        # Automatically retrieve as much metadata as possible
        metadata = super().get_metadata()

        session = self._mat_dict["session"]
        experimenter = [", ".join(session["experimenter"].split(" ")[::-1])]
        format_string = "%m/%d/%Y %I:%M:%S %p"
        date_from_mat = f"{session['date']} {session['time']}"
        session_start_time = datetime.strptime(date_from_mat, format_string)

        metadata_from_mat_dict = dict(
            Subject=dict(subject_id=session["animal"]),  # , sex="U", species="Mus musculus"),
            NWBFile=dict(experimenter=experimenter)
            # session_start_time=session_start_time)
        )
        metadata = dict_deep_update(metadata, metadata_from_mat_dict)

        return metadata

    def add_lab_meta_data(self, nwbfile: NWBFile, metadata: dict):
        session = self._mat_dict["session"]

        session_start_time = metadata["NWBFile"]["session_start_time"]
        session_end_time = session_start_time + timedelta(seconds=session["sessionTime"])

        maze_extension = MazeExtension(name="mazes", description="description of the mazes")

        mazes = session["protocol"]["mazes"]
        mazes_criteria = session["protocol"]["criteria"]

        for row_ind, criteria_row in enumerate(mazes_criteria):
            for k, v in criteria_row.items():
                if isinstance(v, np.ndarray):
                    mazes_criteria[row_ind][k] = np.nan if not len(v) else v[0]  # todo fix
                else:
                    mazes_criteria[row_ind][k] = float(v)

        for maze_row_ind, (maze_row, criteria_row) in enumerate(zip(mazes, mazes_criteria)):
            maze_row = dict_deep_update(maze_row, criteria_row)
            row = dict((k, v) for k, v in maze_row.items() if k in MazeExtension.mazes_attr)
            maze_extension.add_row(**row)

        lab_metadata_kwargs = dict(
            name="LabMetaData",
            experiment_name=session["experiment"].replace(".mat", ""),
            num_trials=session["nTrials"],
            total_reward=session["totalReward"],
            num_iterations=session["nIterations"],
            advance=session["advance"],
            session_end_time=str(session_end_time),
            rig=RigExtension(name="rig", rig=session["rig"]),
            mazes=maze_extension,
        )

        # Add to file
        nwbfile.add_lab_meta_data(lab_meta_data=LabMetaDataExtension(**lab_metadata_kwargs))

    def add_trials(self, nwbfile: NWBFile):
        session = self._mat_dict["session"]

        trials = session["BehaviorTimes"]
        trial_start_times = trials["StartOfTrial"]
        trial_start_times_flatten = np.squeeze(trial_start_times.toarray())
        trial_end_times = trials["EndOfTrial"]
        trial_end_times_flatten = np.squeeze(trial_end_times.toarray())

        for trial_ind in range(session["nTrials"]):
            nwbfile.add_trial(
                start_time=trial_start_times_flatten[trial_ind],
                stop_time=trial_end_times_flatten[trial_ind],
            )

        custom_trial_columns = [k for k in trials.keys() if k not in ["StartOfTrial", "EndOfTrial"]]
        for custom_trial_column in custom_trial_columns:
            trial_column_shape = trials[custom_trial_column].shape[0]
            if trial_column_shape == 0 or trial_column_shape < (session["nTrials"]) - 1:
                continue

            custom_trial_times = np.squeeze(trials[custom_trial_column].toarray())
            if custom_trial_column == "InterTrial":
                # the last trial doesn't have a value
                custom_trial_times = np.append(custom_trial_times, np.nan)

            nwbfile.add_trial_column(
                name=custom_trial_column,
                description="A custom trial column.",  # todo get mapping
                data=custom_trial_times,
            )

        performance = session["performance"]
        # todo need descriptions for "mazeID", "mainMazeID", etc.
        # startTime is different from trial start time
        # need discription for cueParams
        performance_column_names = [column for column in performance.keys() if column not in ["cueParams"]]
        for performance_column in performance_column_names:
            data = performance[performance_column]
            if data.shape[0] == 0:
                continue

            if isinstance(data, scipy.sparse._csc.csc_matrix):
                data = np.squeeze(data.toarray())
            nwbfile.add_trial_column(
                name=performance_column,
                description="A custom performance column.",  # todo add mapping
                data=data,
            )

        return nwbfile

    def add_events(self, nwbfile: NWBFile):
        licks = self._mat_dict["session"]["licks"]
        timestamps = np.squeeze(self._mat_dict["session"]["timestamps"].toarray())
        if licks.shape[0] == 0:
            return nwbfile

        if isinstance(licks, scipy.sparse._csc.csc_matrix):
            licks = np.squeeze(licks.toarray())

        time_series = TimeSeries(
            name="licks",
            data=H5DataIO(licks, compression="gzip"),
            timestamps=H5DataIO(timestamps, compression="gzip"),
            description="no discription",
            unit="a.u.",  # TODO confirm what is unit
        )

        behavioral_events = BehavioralEvents(time_series=time_series, name="BehavioralEvents")
        behavior = get_module(nwbfile, "behavior")
        behavior.add(behavioral_events)

    def add_position(self, nwbfile: NWBFile):
        session = self._mat_dict["session"]

        # Processed position, velocity, viewAngle
        position_obj = Position(name="Position")
        view_angle_obj = CompassDirection(name="ViewAngle")

        position = session["position"]  # x,y,z,viewangle

        timestamps = np.squeeze(session["timestamps"].toarray())
        # rate = calculate_regular_series_rate(timestamps)

        position_data = position[:, :-1]
        view_angle_data = position[:, -1]

        position_obj.create_spatial_series(
            name="SpatialSeries",
            data=H5DataIO(position_data, compression="gzip"),
            reference_frame="unknown",  # todo
            conversion=0.01,
            timestamps=H5DataIO(timestamps, compression="gzip"),
        )

        view_angle_obj.create_spatial_series(
            name="PositionViewAngle",
            data=H5DataIO(view_angle_data, compression="gzip"),
            reference_frame="unknown",  # todo
            timestamps=H5DataIO(timestamps, compression="gzip"),
        )

        velocity = session["velocity"]
        velocity_data = velocity[:, :2]
        view_angle_velocity_data = velocity_data[:, -1]

        velocity_ts = TimeSeries(
            name="Velocity",
            data=H5DataIO(velocity_data, compression="gzip"),
            unit="m/s",
            conversion=0.01,
            timestamps=H5DataIO(timestamps, compression="gzip"),
        )

        view_angle_obj.create_spatial_series(
            name="VelocityViewAngle",
            data=H5DataIO(view_angle_velocity_data, compression="gzip"),
            reference_frame="unknown",  # todo
            timestamps=H5DataIO(timestamps, compression="gzip"),
        )

        sensor_dots = session["sensordots"]
        # not sure this is the right data type, will have to ask during meeting what is this
        # also there is velocitygain
        sensor_dots_ts = TimeSeries(
            name="TimeSeriesSensorDots",
            data=H5DataIO(sensor_dots, compression="gzip"),
            unit="a.u.",
            timestamps=H5DataIO(timestamps, compression="gzip"),
        )

        velocity_gain = session["velocityGain"]
        velocity_gain_ts = TimeSeries(
            name="TimeSeriesVelocityGain",
            data=H5DataIO(velocity_gain, compression="gzip"),
            unit="a.u.",
            timestamps=H5DataIO(timestamps, compression="gzip"),
        )

        behavior = get_module(nwbfile, "behavior", "contains processed behavioral data")
        behavior.add(position_obj)
        behavior.add(view_angle_obj)
        behavior.add(velocity_ts)
        behavior.add(sensor_dots_ts)
        behavior.add(velocity_gain_ts)

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: dict):
        self.add_lab_meta_data(nwbfile=nwbfile, metadata=metadata)
        self.add_trials(nwbfile=nwbfile)
        self.add_position(nwbfile=nwbfile)
        self.add_events(nwbfile=nwbfile)

        return nwbfile
