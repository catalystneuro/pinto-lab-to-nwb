"""Primary class for converting experiment-specific behavior."""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from hdmf.backends.hdf5 import H5DataIO
from hdmf.common import DynamicTable
from neuroconv.tools.nwb_helpers import get_module
from pynwb import TimeSeries
from pynwb.behavior import Position, CompassDirection, BehavioralEvents
from pynwb.file import NWBFile

from neuroconv.utils import FilePathType, dict_deep_update, load_dict_from_file
from neuroconv.basedatainterface import BaseDataInterface

from ndx_pinto_metadata import LabMetaDataExtension, MazeExtension, StimulusProtocolExtension
from ndx_pinto_metadata.utils.mat_utils import convert_mat_file_to_dict, create_indexed_array
from scipy.io.matlab import MatlabOpaque

from pinto_lab_to_nwb.behavior.utils import convert_mat_file_to_dict


class ViRMENBehaviorInterface(BaseDataInterface):
    """Behavior interface for into_the_void conversion"""

    def __init__(self, file_path: FilePathType, verbose: bool = True):
        self.verbose = verbose
        super().__init__(file_path=file_path)
        self._mat_dict = convert_mat_file_to_dict(mat_file_name=file_path)

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
        maze_extension = MazeExtension(name="mazes", description=f"The parameters for the mazes in {experiment_name}.")

        mazes = session["protocol"].pop("mazes")
        for maze in mazes:
            maze_extension.add_row(
                **maze,
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

        stimulus_protocol_metadata = metadata["StimulusProtocolExtension"]
        global_settings = session["protocol"].pop("globalSettings")

        global_settings = dict((global_settings[i], global_settings[i + 1]) for i in range(0, len(global_settings), 2))
        session_protocol = dict_deep_update(session["protocol"], global_settings)

        stimulus_protocol_kwargs = dict()
        for protocol_key, protocol_value in session_protocol.items():
            if protocol_key in StimulusProtocolExtension.__nwbfields__:
                stimulus_protocol_kwargs[protocol_key] = session_protocol[protocol_key]
            elif protocol_key in stimulus_protocol_metadata:
                protocol_key_from_metadata = stimulus_protocol_metadata[protocol_key]
                stimulus_protocol_kwargs[protocol_key_from_metadata] = session_protocol[protocol_key]

        stimulus_protocol = StimulusProtocolExtension(
            name="stimulus_protocol",
            **stimulus_protocol_kwargs,
        )

        lab_metadata_kwargs = dict(
            name="LabMetaData",
            experiment_name=session["experiment"].replace(".mat", ""),
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

        cue_parameters = performance["cueParams"]
        cue_params_name = list(cue_parameters[0].keys())[0]

        columns_to_skip = []
        columns_to_indexed_array = []
        columns_to_add = []
        for column_name, column_value in cue_parameters[0][cue_params_name].items():
            if isinstance(column_value, MatlabOpaque):
                columns_to_skip.append(column_name)
            elif isinstance(column_value, np.ndarray):
                columns_to_indexed_array.append(column_name)
            else:
                columns_to_add.append(column_name)

        cue_parameters_table = DynamicTable(name=cue_params_name, description="Holds cue parameters for each trial.")
        for cue_column in columns_to_add:
            cue_parameters_table.add_column(name=cue_column, description=f"{cue_params_name} cue parameter.")
        for trial_ind in range(session["nTrials"]):
            cue_parameters_per_trial = dict((k, v) for k, v in cue_parameters[trial_ind][cue_params_name].items() if k in columns_to_add)
            cue_parameters_table.add_row(**cue_parameters_per_trial)

        for cue_column in columns_to_indexed_array:
            to_indexed_array = []
            for trial_ind in range(session["nTrials"]):
                to_indexed_array.append(cue_parameters[trial_ind][cue_params_name][cue_column])

            indexed_array, indexed_array_indices = create_indexed_array(to_indexed_array)
            cue_parameters_table.add_column(
                name=cue_column,
                description=f"{cue_params_name} indexed cue parameter.",
                data=indexed_array,
                index=indexed_array_indices,
            )

        behavior = get_module(nwbfile, "behavior", "contains processed behavioral data")
        behavior.add(cue_parameters_table)

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

# NCCR47_TowersTaskSwitchEasy_Session_20230522_105332
# Coriander_DelayedMatchToEvidence_Session_20230615_101750
# JsCheddarGeese_TowersTaskSwitchEasy_Session_20230522_151257
fname = "JsCheddarGeese_TowersTaskSwitchEasy_Session_20230522_151257"
behavior_file_path = f"/Volumes/t7-ssd/Pinto/Behavior/{fname}.mat"
interface = ViRMENBehaviorInterface(file_path=behavior_file_path)
metadata = interface.get_metadata()
interface.run_conversion(nwbfile_path=f"{fname}.nwb", overwrite=True)
