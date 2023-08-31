"""Primary class for converting experiment-specific behavior."""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from hdmf.backends.hdf5 import H5DataIO
from hdmf.common import DynamicTable
from neuroconv.tools.nwb_helpers import get_module
from pymatreader import read_mat
from pynwb import TimeSeries
from pynwb.behavior import Position, CompassDirection, BehavioralEvents
from pynwb.file import NWBFile

from neuroconv.utils import FilePathType, dict_deep_update, load_dict_from_file
from neuroconv.basedatainterface import BaseDataInterface

from ndx_pinto_metadata import LabMetaDataExtension, MazeExtension
from ..utils import create_indexed_array


class ViRMENBehaviorInterface(BaseDataInterface):
    """Behavior interface for into_the_void conversion"""

    def __init__(self, file_path: FilePathType, verbose: bool = True):
        self.verbose = verbose
        super().__init__(file_path=file_path)
        self._mat_dict = read_mat(filename=file_path)

    def get_metadata(self):
        metadata = super().get_metadata()

        session = self._mat_dict["session"]
        experimenter = [", ".join(session["experimenter"].split(" ")[::-1])]
        format_string = "%m/%d/%Y %I:%M:%S %p"
        date_from_mat = f"{session['date']} {session['time']}"
        session_start_time = datetime.strptime(date_from_mat, format_string)

        metadata_from_mat_dict = dict(
            Subject=dict(subject_id=session["animal"], sex="U", species="Mus musculus"),
            NWBFile=dict(experimenter=experimenter, session_start_time=session_start_time),
        )

        metadata = dict_deep_update(metadata, metadata_from_mat_dict)

        # load stimulus protocol name mapping from yaml
        metadata_from_yaml = load_dict_from_file(
            file_path=Path(__file__).parent.parent / "metadata" / "virmen_metadata.yaml"
        )
        metadata = dict_deep_update(metadata, metadata_from_yaml)

        return metadata

    def add_lab_meta_data(self, metadata: dict, nwbfile: NWBFile):
        session = self._mat_dict["session"]

        experiment_name = session["experiment"].replace(".mat", "")
        experiment_code = session["experimentcode"]["function_handle"]["function"]

        mazes = session["protocol"].pop("mazes")
        maze_extension = MazeExtension(
            name="mazes",
            description=f"The parameters for the mazes in {experiment_name}.",
            id=list(list(mazes.values())[0]),
        )

        for maze in mazes:
            maze_extension.add_column(
                name=maze,
                description="maze column",
                data=mazes[maze],
            )

        criteria = session["protocol"].pop("criteria")
        mazes_criteria_df = pd.DataFrame.from_records(criteria)
        for maze_column in mazes_criteria_df:
            maze_kwargs = dict(
                name=maze_column,
                description="maze information",
            )
            if mazes_criteria_df[maze_column].dtype == object:
                indexed_array, indexed_array_indices = create_indexed_array(mazes_criteria_df[maze_column].values)
                maze_kwargs.update(
                    data=list(indexed_array),
                    index=list(indexed_array_indices),
                )
            else:
                maze_kwargs.update(
                    data=mazes_criteria_df[maze_column].values.tolist(),
                )

            maze_extension.add_column(**maze_kwargs)

        global_settings = session["protocol"].pop("globalSettings")

        global_settings = dict((global_settings[i], global_settings[i + 1]) for i in range(0, len(global_settings), 2))
        session_protocol = dict_deep_update(session["protocol"], global_settings)

        stimulus_code = session_protocol.pop("stimulusGenerator")
        stimulus_code_str = stimulus_code["function_handle"]["function"]
        session_protocol.update(stimulus_code=stimulus_code_str)
        session_protocol.pop("stimulusParameters")
        # Create stimulus protocol with global settings
        stimulus_protocol = DynamicTable(
            name="stimulus_protocol",
            description="Holds information about the stimulus protocol.",
        )

        for column_name in session_protocol:
            stimulus_protocol.add_column(
                name=column_name,
                description="stimulus protocol parameter.",
            )

        stimulus_protocol.add_row(**session_protocol)
        lab_metadata_kwargs = dict(
            name="LabMetaData",
            experiment_name=session["experiment"].replace(".mat", ""),
            experiment_code=experiment_code,
            session_index=session["sessionIndex"],
            total_reward=session["totalReward"],
            squal=session["squal"],
            rig=session["rig"],
            num_trials=session["nTrials"],
            num_iterations=session["nIterations"],
            session_duration=session["sessionTime"],
            advance=session["advance"],
            mazes=maze_extension,
            stimulus_protocol=stimulus_protocol,
        )

        # Add to NWBfile
        nwbfile.add_lab_meta_data(lab_meta_data=LabMetaDataExtension(**lab_metadata_kwargs))

    def add_trials(self, nwbfile: NWBFile):
        session = self._mat_dict["session"]

        trials = session["BehaviorTimes"]
        trial_start_times = trials["StartOfTrial"]
        trial_start_times_flatten = np.squeeze(trial_start_times.toarray())
        trial_end_times = trials["EndOfTrial"]
        trial_end_times_flatten = np.squeeze(trial_end_times.toarray())
        num_trials = session["nTrials"]

        for trial_ind in range(num_trials):
            nwbfile.add_trial(
                start_time=trial_start_times_flatten[trial_ind],
                stop_time=trial_end_times_flatten[trial_ind],
            )

        custom_trial_columns = [k for k in trials.keys() if k not in ["StartOfTrial", "EndOfTrial", "EndOfExperiment"]]
        for custom_trial_column in custom_trial_columns:
            trial_column_shape = trials[custom_trial_column].shape[0]
            if trial_column_shape == 0:
                continue

            custom_trial_times = np.squeeze(trials[custom_trial_column].toarray())
            if custom_trial_column == "InterTrial":
                # the last trial doesn't have a value
                custom_trial_times = np.append(custom_trial_times, np.nan)

            if trial_column_shape < num_trials and custom_trial_column != "InterTrial":
                padding = np.full(num_trials - trial_column_shape, np.nan)
                custom_trial_times = np.concatenate([custom_trial_times, padding], axis=0)

            nwbfile.add_trial_column(
                name=custom_trial_column,
                description="A custom trial column.",
                data=custom_trial_times,
            )

        performance = session["performance"]
        cue_parameters = performance.pop("cueParams")
        cue_params_name = session["protocol"]["stimulusGenerator"]["function_handle"]["function"]

        columns_to_skip = []
        columns_to_indexed_array = []
        columns_to_add = []
        for column_name, column_value in cue_parameters[cue_params_name][0].items():
            if isinstance(column_value, dict):
                columns_to_skip.append(column_name)
            elif isinstance(column_value, np.ndarray):
                columns_to_indexed_array.append(column_name)
            else:
                columns_to_add.append(column_name)

        # Add to trials table
        for cue_column in columns_to_indexed_array:
            to_indexed_array = []
            for trial_ind in range(session["nTrials"]):
                to_indexed_array.append(cue_parameters[cue_params_name][trial_ind][cue_column])
            indexed_array, indexed_array_indices = create_indexed_array(to_indexed_array)
            nwbfile.trials.add_column(
                name=cue_column,
                description=f"{cue_params_name} indexed cue parameter.",
                data=indexed_array,
                index=indexed_array_indices,
            )

        for cue_column in columns_to_add:
            data = [cue_parameters[cue_params_name][trial_ind][cue_column] for trial_ind in range(session["nTrials"])]
            nwbfile.trials.add_column(
                name=cue_column,
                description=f"{cue_params_name} cue parameter.",
                data=data
            )

        for performance_column, data in performance.items():
            if isinstance(data, list):
                data = np.array(data)

            if data.shape[0] == 0:
                continue

            if performance_column in nwbfile.trials.colnames:
                continue

            if isinstance(data, scipy.sparse._csc.csc_matrix):
                data = np.squeeze(data.toarray())
            nwbfile.add_trial_column(
                name=performance_column,
                description="A custom performance column.",
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
