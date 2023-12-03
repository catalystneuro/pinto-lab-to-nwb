"""Primary class for converting experiment-specific behavior."""
from datetime import datetime

import numpy as np
import pandas as pd
import scipy
from hdmf.backends.hdf5 import H5DataIO
from hdmf.common import DynamicTable
from pymatreader import read_mat
from pynwb import TimeSeries
from pynwb.behavior import Position, CompassDirection
from pynwb.file import NWBFile

from neuroconv.utils import FilePathType, dict_deep_update, calculate_regular_series_rate
from neuroconv.basetemporalalignmentinterface import BaseTemporalAlignmentInterface
from neuroconv.tools.nwb_helpers import get_module
from neuroconv.tools.signal_processing import get_rising_frames_from_ttl

from ndx_pinto_metadata import LabMetaDataExtension, MazeExtension
from ndx_events import AnnotatedEventsTable

from pinto_lab_to_nwb.behavior.utils import create_indexed_array


class ViRMENBehaviorInterface(BaseTemporalAlignmentInterface):
    """Behavior interface for into_the_void conversion"""

    def __init__(self, file_path: FilePathType, verbose: bool = True):
        self.verbose = verbose
        super().__init__(file_path=file_path)
        self._mat_dict = read_mat(filename=file_path)
        self._times = None

    def _get_session_start_time(self):
        session = self._mat_dict["session"]
        format_string = "%m/%d/%Y %I:%M:%S %p"
        date_from_mat = f"{session['date']} {session['time']}"
        session_start_time = datetime.strptime(date_from_mat, format_string)
        return session_start_time

    def get_metadata(self):
        metadata = super().get_metadata()

        session = self._mat_dict["session"]
        experimenter = [", ".join(session["experimenter"].split(" ")[::-1])]
        session_start_time = self._get_session_start_time()

        metadata_from_mat_dict = dict(
            Subject=dict(subject_id=session["animal"]),
            NWBFile=dict(experimenter=experimenter, session_start_time=session_start_time),
        )

        metadata = dict_deep_update(metadata, metadata_from_mat_dict)

        return metadata

    def get_original_timestamps(self) -> np.ndarray:
        return np.squeeze(self._mat_dict["session"]["timestamps"].toarray())

    def get_timestamps(self) -> np.ndarray:
        return self._times if self._times is not None else self.get_original_timestamps()

    def set_aligned_timestamps(self, aligned_timestamps: np.ndarray):
        self._times = aligned_timestamps

    def add_lab_meta_data(self, nwbfile: NWBFile):
        session = self._mat_dict["session"]

        experiment_name = session["experiment"].replace(".mat", "")
        experiment_code = session["experimentcode"]["function_handle"]["function"]

        mazes = session["protocol"].pop("mazes")
        maze_extension = MazeExtension(
            name="mazes",
            description=f"The parameters for the mazes in {experiment_name}.",
            id=np.arange(len(list(mazes.values())[0])),
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
            surface_quality=float(session["squal"]),
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

        behavior_times = session["BehaviorTimes"]
        trial_start_times = behavior_times["StartOfTrial"]
        trial_start_times_flatten = np.squeeze(trial_start_times.toarray())
        trial_end_times = behavior_times["EndOfTrial"]
        trial_end_times_flatten = np.squeeze(trial_end_times.toarray())
        num_trials = session["nTrials"]

        for trial_ind in range(num_trials):
            nwbfile.add_trial(
                start_time=trial_start_times_flatten[trial_ind],
                stop_time=trial_end_times_flatten[trial_ind],
            )

        custom_trial_columns = [
            k for k in behavior_times.keys() if k not in ["StartOfTrial", "EndOfTrial", "EndOfExperiment"]
        ]
        for custom_trial_column in custom_trial_columns:
            trial_column_shape = behavior_times[custom_trial_column].shape[0]
            if trial_column_shape == 0:
                continue

            custom_trial_times = np.squeeze(behavior_times[custom_trial_column].toarray())
            if custom_trial_column == "InterTrial":
                # the last 'InterTrial' doesn't have a value
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
        task_name = session["protocol"]["stimulusGenerator"]["function_handle"]["function"]

        trial_type_num_to_str = {1: "left", 2: "right"}
        trial_type_data = np.squeeze(performance["trialType"].toarray()).astype(int)
        trial_type_data = np.vectorize(trial_type_num_to_str.get)(trial_type_data)
        nwbfile.add_trial_column(
            name="trial_type",
            description="Defines the correct side (left or right) for a given trial.",
            data=trial_type_data,
        )

        trial_choice_data = np.squeeze(performance["choice"].toarray()).astype(int)
        trial_choice_data = np.vectorize(trial_type_num_to_str.get)(trial_choice_data)
        nwbfile.add_trial_column(
            name="choice",
            description="Defines which side the animal chose (left or right) for a given trial.",
            data=trial_choice_data,
        )

        stim_types_data = np.squeeze(performance["stimType"].toarray()).astype(int)
        stim_stype_num_to_str = {1: "black", 2: "white"}
        stim_types_data = np.vectorize(stim_stype_num_to_str.get)(stim_types_data)
        description = "Defines the stimulus color (black or white) for a given trial."
        if "towers" in task_name.lower():
            description = "The stimulus color defaults to black for Towers task."
        nwbfile.add_trial_column(
            name="stimulus_type",
            description=description,
            data=stim_types_data,
        )

        cue_parameters = performance["cueParams"]
        cue_parameter_names = list(cue_parameters[task_name][0].keys())
        # salientside, distractorside are in trial_type
        cue_parameter_names = [param for param in cue_parameter_names if param not in ["salientside", "distractorside"]]
        for cue_parameter_name in cue_parameter_names:
            data = [cue_parameters[task_name][trial_ind][cue_parameter_name] for trial_ind in range(num_trials)]
            is_data_ragged_array = any(isinstance(value, np.ndarray) for value in data)
            trials_column_kwargs = dict(
                name=cue_parameter_name,
                description=f"{task_name} cue parameter.",
            )
            if is_data_ragged_array:
                data, indexed_array_indices = create_indexed_array(data)
                trials_column_kwargs.update(index=indexed_array_indices)

            if all(np.isnan(data)):
                continue

            trials_column_kwargs.update(data=data)

            nwbfile.trials.add_column(**trials_column_kwargs)

        additional_columns_to_add = [
            col
            for col in performance.keys()
            if col not in ["startTime", "trialTime", "trialType", "choice", "stimType", "cueParams"]
        ]
        for column_name in additional_columns_to_add:
            data = performance[column_name]
            if isinstance(data, list):
                data = np.array(data)

            if data.shape[0] == 0:
                continue

            if column_name in nwbfile.trials.colnames:
                continue

            if isinstance(data, scipy.sparse._csc.csc_matrix):
                data = np.squeeze(data.toarray())
            nwbfile.add_trial_column(
                name=column_name,
                description="A custom performance column.",
                data=data,
            )

        return nwbfile

    def add_events(self, nwbfile: NWBFile):
        behavior = get_module(nwbfile, "behavior")
        timestamps = self.get_timestamps()

        licks = self._get_time_series(series_name="licks")
        if licks is not None:
            licks_time_series = TimeSeries(
                name="licks",
                data=H5DataIO(licks, compression="gzip"),
                timestamps=H5DataIO(timestamps, compression="gzip"),
                description="The lick response measured over time.",
                unit="a.u.",
            )
            behavior.add(licks_time_series)

        opto_voltage = self._get_time_series("optoVoltageOut")
        if opto_voltage is not None:
            opto_voltage_time_series = TimeSeries(
                name="opto_voltage",
                data=H5DataIO(opto_voltage, compression="gzip"),
                timestamps=H5DataIO(timestamps, compression="gzip"),
                description="The voltage passed to the opto LED.",
                unit="volts",
            )
            behavior.add(opto_voltage_time_series)

        ttl_events = AnnotatedEventsTable(
            name="TTLs",
            description="The times when the eye tracking and/or imaging was on determined from the ViRMEN system.",
        )

        eye_tracking_ttl = self._get_time_series("eyeCam")
        if eye_tracking_ttl is not None:
            rising_frames = get_rising_frames_from_ttl(trace=eye_tracking_ttl)
            eye_tracking_times = timestamps[rising_frames]
            ttl_events.add_event_type(
                label="EyeTracking",
                event_description="The times when the eye tracking camera was on.",
                event_times=eye_tracking_times,
            )

        widefield_ttl = self._get_time_series("widefield")
        if widefield_ttl is not None:
            rising_frames = get_rising_frames_from_ttl(trace=widefield_ttl)
            widefield_times = timestamps[rising_frames]
            ttl_events.add_event_type(
                label="Widefield",
                event_description="The times when the widefield imaging was on.",
                event_times=widefield_times,
            )

        two_photon_ttl = self._get_time_series("twop")
        if two_photon_ttl is not None:
            rising_frames = get_rising_frames_from_ttl(trace=two_photon_ttl)
            two_photon_times = timestamps[rising_frames]
            ttl_events.add_event_type(
                label="TwoPhoton",
                event_description="The times when the two photon imaging was on.",
                event_times=two_photon_times,
            )

        if ttl_events.event_times is None:
            return

        behavior.add(ttl_events)

    def _get_time_series(self, series_name: str = "licks"):
        if series_name not in self._mat_dict["session"]:
            return

        data = self._mat_dict["session"][series_name]
        if isinstance(data, scipy.sparse._csc.csc_matrix):
            data = np.squeeze(data.toarray())

        if not np.any(data):
            return

        return data

    def add_position(self, nwbfile: NWBFile):
        session = self._mat_dict["session"]

        # Processed position, velocity, viewAngle
        position_obj = Position(name="Position")
        view_angle_obj = CompassDirection(name="ViewAngle")

        position = session["position"]  # x,y,z,viewangle

        timestamps = self.get_timestamps()
        rate = calculate_regular_series_rate(timestamps)
        if rate:
            timing_kwargs = dict(rate=rate, starting_time=timestamps[0])
        else:
            assert (
                len(timestamps) == position.shape[0]
            ), f"The length of timestamps ({len(timestamps)}) must match the length of position ({position.shape[0]})."
            timing_kwargs = dict(timestamps=H5DataIO(timestamps, compression="gzip"))

        position_data = position[:, :-1]
        view_angle_data = position[:, -1]

        # For the position, our convention across tasks and mazes is that position (0,0) is the start of the "sample"
        # region (also referred to as the "cue" region) (note that this is not the position where the mouse starts
        # each trial, which varies by maze and task).
        reference_frame = "(0,0) is the start of the 'sample' region (or 'cue' region) which varies by maze and task."

        position_obj.create_spatial_series(
            name="SpatialSeries",
            data=H5DataIO(position_data, compression="gzip"),
            description="The x, y, z position of the animal by ViRMEN iteration.",
            reference_frame=reference_frame,
            conversion=0.01,
            **timing_kwargs,
        )

        view_angle_obj.create_spatial_series(
            name="PositionViewAngle",
            data=H5DataIO(view_angle_data, compression="gzip"),
            description="The view angle of the animal by ViRMEN iteration.",
            reference_frame=reference_frame,
            **timing_kwargs,
        )

        velocity = session["velocity"]
        velocity_data = velocity[:, :2]
        view_angle_velocity_data = velocity_data[:, -1]

        velocity_ts = TimeSeries(
            name="Velocity",
            data=H5DataIO(velocity_data, compression="gzip"),
            description="The velocity of the animal by ViRMEN iteration.",
            unit="m/s",
            conversion=0.01,
            **timing_kwargs,
        )

        view_angle_obj.create_spatial_series(
            name="VelocityViewAngle",
            data=H5DataIO(view_angle_velocity_data, compression="gzip"),
            description="The velocity view angle of the animal by ViRMEN iteration.",
            reference_frame=reference_frame,
            **timing_kwargs,
        )

        sensor_dots = session["sensordots"]
        sensor_dots_ts = TimeSeries(
            name="SensorDots",
            data=H5DataIO(sensor_dots, compression="gzip"),
            description="The sensordots by ViRMEN iteration.",
            unit="a.u.",
            **timing_kwargs,
        )

        velocity_gain = session["velocityGain"]
        velocity_gain_ts = TimeSeries(
            name="VelocityGain",
            data=H5DataIO(velocity_gain, compression="gzip"),
            description="The velocity gain by ViRMEN iteration.",
            unit="a.u.",
            **timing_kwargs,
        )

        behavior = get_module(nwbfile, "behavior", "contains processed behavioral data")
        behavior.add(position_obj)
        behavior.add(view_angle_obj)
        behavior.add(velocity_ts)
        behavior.add(sensor_dots_ts)
        behavior.add(velocity_gain_ts)

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: dict):
        self.add_lab_meta_data(nwbfile=nwbfile)
        self.add_trials(nwbfile=nwbfile)
        self.add_position(nwbfile=nwbfile)
        self.add_events(nwbfile=nwbfile)

        return nwbfile
