from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
from neuroconv import BaseTemporalAlignmentInterface
from neuroconv.tools import get_module
from neuroconv.tools.roiextractors import add_imaging_plane
from neuroconv.tools.roiextractors.roiextractors import get_default_ophys_metadata, add_image_segmentation
from neuroconv.utils import FolderPathType, dict_deep_update, get_base_schema, get_schema_from_hdmf_class
from pynwb import NWBFile
from pynwb.device import Device
from pynwb.ogen import OptogeneticStimulusSite
from pynwb.ophys import PlaneSegmentation, ImageSegmentation, ImagingPlane
from roiextractors.extractors.tiffimagingextractors.brukertiffimagingextractor import (
    _parse_xml,
    _determine_frame_rate,
)
from xml.etree import ElementTree
from ndx_holographic_stimulation import (
    PatternedOptogeneticSeries,
    SpiralScanning,
    PatternedOptogeneticStimulusSite,
    SpatialLightModulator,
    LightSource,
)


def parse_mark_points_xml(mark_points_path):
    assert mark_points_path.is_file(), f"The XML configuration file is not found at '{mark_points_path}'."
    tree = ElementTree.parse(mark_points_path)
    root = tree.getroot()
    return root


def get_absolute_frame_times_from_xml(xml_root):
    sequence_elements = xml_root.findall("Sequence")
    first_frame_absolute_time = float(sequence_elements[0].find("Frame").attrib["absoluteTime"])
    absolute_frame_times = []
    for sequence in sequence_elements:
        frame_elements = sequence.findall("Frame")
        absolute_times = [float(frame.attrib["absoluteTime"]) - first_frame_absolute_time for frame in frame_elements]
        absolute_frame_times.extend(absolute_times)

    return np.array(absolute_frame_times)


class HolographicStimulationInterface(BaseTemporalAlignmentInterface):
    """Data interface for adding the holographic stimulation to the NWB file."""

    def get_metadata_schema(self) -> dict:
        metadata_schema = super().get_metadata_schema()
        metadata_schema["required"] = ["Ophys"]
        metadata_schema["properties"]["Ophys"] = get_base_schema()
        metadata_schema["properties"]["Ophys"]["properties"] = dict(
            Device=dict(type="array", minItems=1, items=get_schema_from_hdmf_class(Device)),
        )
        metadata_schema["properties"]["Ophys"]["properties"].update(
            ImageSegmentation=get_schema_from_hdmf_class(ImageSegmentation),
            ImagingPlane=get_schema_from_hdmf_class(ImagingPlane),
            OptogeneticStimulusSite=get_schema_from_hdmf_class(OptogeneticStimulusSite),
        )
        metadata_schema["properties"]["Ophys"]["required"] = ["Device", "ImageSegmentation"]

        # Temporary fixes until centralized definition of metadata schemas
        metadata_schema["properties"]["Ophys"]["properties"]["ImagingPlane"].update(type="array")
        metadata_schema["properties"]["Ophys"]["properties"]["OptogeneticStimulusSite"].update(type="array")

        return metadata_schema

    def __init__(
        self,
        folder_path: FolderPathType,
        plane_segmentation_name: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Data interface for adding the holographic stimulation to the NWB file.

        Parameters
        ----------
        folder_path : FolderPathType
            The folder path that contains the holographic stimulation data.
        plane_segmentation_name: str, optional
            The name of the plane segmentation to use for the holographic stimulation.
        verbose : bool, default: True
        """

        # Load the XML file from the imaging folder
        self.folder_path = Path(folder_path)
        self.plane_segmentation_name = plane_segmentation_name or "PlaneSegmentationHolographicStimulation"
        self._times = None

        xml_root = _parse_xml(folder_path)
        sequence_elements = xml_root.findall("Sequence")
        assert (
            sequence_elements[0].find("MarkPoints") is not None
        ), "The XML file does not contain the holographic stimulation data."

        # Get the frame period from the XML file
        frame_rate = _determine_frame_rate(element=sequence_elements[0])
        frame_period = 1 / frame_rate

        first_frame_absolute_time = float(sequence_elements[0].find("Frame").attrib["absoluteTime"])
        cycle_relative_start_times = []
        cycle_relative_durations = []
        sequence_mark_points_list = []
        num_rois = 0

        rois_dict = defaultdict(list)

        for sequence in sequence_elements:
            frame_elements = sequence.findall("Frame")
            absolute_times = [
                float(frame.attrib["absoluteTime"]) - first_frame_absolute_time for frame in frame_elements
            ]
            mark_points_file_name = sequence.find("MarkPoints").get("filename")
            file_path = self.folder_path / mark_points_file_name
            mark_points = parse_mark_points_xml(file_path)
            point_elements = mark_points.findall("PVMarkPointElement")

            initial_delays = []
            durations = []
            point_elements_list = []
            for point_element in point_elements:
                markpoint_data = point_element.attrib
                galvo_point_element = point_element.find("PVGalvoPointElement")
                # Extract attributes from 'PVGalvoPointElement'
                galvo_data = galvo_point_element.attrib

                markpoint_data = dict_deep_update(markpoint_data, galvo_data)

                # Find all 'Point' elements within 'PVGalvoPointElement'
                points = galvo_point_element.findall("Point")
                for point in points:
                    point_data = point.attrib
                    point_index = int(point_data["Index"]) - 1
                    if point_data in rois_dict[point_index]:
                        continue
                    rois_dict[point_index].append(point_data)
                point_list = [point.attrib for point in points]
                num_rois = max(len(point_list), num_rois)
                markpoint_data["points"] = point_list

                # milliseconds to seconds
                relative_start_time = float(galvo_data["InitialDelay"]) / 1000
                spiral_width = float(galvo_data["Duration"]) + float(galvo_data["InterPointDelay"])

                num_repetitions = int(markpoint_data["Repetitions"])
                stimulus_duration = (spiral_width * num_repetitions) / 1000

                initial_delays.append(relative_start_time)
                durations.append(stimulus_duration)
                point_elements_list.append(markpoint_data)

            sequence_mark_points_list.extend(point_elements_list)

            cycle_relative_durations.extend(durations)
            durations[0] = 0
            cycles_initial_delays = np.cumsum(np.array(initial_delays) + np.array(durations))
            session_delays = cycles_initial_delays + absolute_times[0]
            offset = np.cumsum([frame_period] * len(point_elements))
            session_delays -= offset
            cycle_relative_start_times.extend(session_delays)

        self.num_rois = num_rois
        self.sequence_mark_points_list = sequence_mark_points_list
        self.cycle_relative_start_times = cycle_relative_start_times
        self.cycle_relative_durations = cycle_relative_durations
        self.rois_dict = rois_dict

        super().__init__(folder_path=folder_path)
        self.verbose = verbose

    def get_holographic_series(self, timestamps):
        holographic_series = np.zeros(shape=(len(timestamps), self.num_rois))
        for sequence_data, cycle_start_time, duration in zip(
            self.sequence_mark_points_list,
            self.cycle_relative_start_times,
            self.cycle_relative_durations,
        ):
            # Index for points start at 1
            point_indices = [int(point["Index"]) - 1 for point in sequence_data["points"]]
            frame_indices = np.where((cycle_start_time < timestamps) & (timestamps <= cycle_start_time + duration))[0]
            power_value = int(sequence_data["UncagingLaserPower"])
            for frame in frame_indices:
                for point in point_indices:
                    holographic_series[frame, point] = power_value

        return holographic_series

    def get_original_timestamps(self) -> np.ndarray:
        xml_root = _parse_xml(self.folder_path)
        timestamps = get_absolute_frame_times_from_xml(xml_root=xml_root)
        return timestamps

    def get_timestamps(self) -> np.ndarray:
        return self._times if self._times is not None else self.get_original_timestamps()

    def set_aligned_timestamps(self, aligned_timestamps: np.ndarray):
        self._times = aligned_timestamps

    def get_metadata(self) -> dict:
        metadata = get_default_ophys_metadata()
        device_name = "device"  # ndx-holographic-stimulation requires a device named "device"
        metadata["Ophys"]["Device"][0].update(name=device_name)
        plane_suffix = self.plane_segmentation_name.replace("PlaneSegmentation", "")
        imaging_plane_name = "ImagingPlane" + plane_suffix
        metadata["Ophys"]["ImagingPlane"][0].update(
            name=imaging_plane_name,
            description="The plane imaged by the microscope during holographic stimulation.",
            device=device_name,
        )
        default_image_segmentation = dict(
            name="ImageSegmentation",
            plane_segmentations=[
                dict(
                    name=self.plane_segmentation_name,
                    description="The stimulated ROIs",
                    imaging_plane=imaging_plane_name,
                )
            ],
        )
        metadata["Ophys"].update(dict(ImageSegmentation=default_image_segmentation))

        # add default optogenetic stimulus site metadata
        default_site_metadata = dict(
            name="site",
            description="The targeted location of the holographic stimulation.",
            device=device_name,
        )
        metadata["Ophys"].update(dict(OptogeneticStimulusSite=[default_site_metadata]))

        return metadata

    def add_to_nwbfile(
        self,
        nwbfile: NWBFile,
        metadata: Optional[dict] = None,
        stub_test: bool = False,
    ) -> None:
        metadata_copy = deepcopy(metadata)

        plane_segmentation_name = self.plane_segmentation_name
        plane_suffix = plane_segmentation_name.replace("PlaneSegmentation", "")
        imaging_plane_name = "ImagingPlane" + plane_suffix

        add_imaging_plane(nwbfile=nwbfile, metadata=metadata_copy, imaging_plane_name=imaging_plane_name)
        device_name = metadata_copy["Ophys"]["Device"][0]["name"]
        device = nwbfile.devices[device_name]

        # Add stimulus pattern to NWBFile
        num_revolutions = int(self.sequence_mark_points_list[0]["SpiralRevolutions"])
        inter_stimulus_interval = int(self.sequence_mark_points_list[0]["InterPointDelay"]) / 1000
        spiral_params = self.sequence_mark_points_list[0]["points"][0]
        diameter = float(spiral_params["SpiralWidth"])
        height = float(spiral_params["SpiralHeight"])
        spiral_scanning = SpiralScanning(
            name="stimulus_pattern",
            diameter=diameter,
            height=height,  # spiral size in microns is 15
            number_of_revolutions=num_revolutions,
            description="The spiral scanning pattern used for holographic stimulation.",
            duration=self.cycle_relative_durations[0],  # duration of each spiral
            number_of_stimulus_presentation=len(self.sequence_mark_points_list),  # or 220 (22 * 10)
            inter_stimulus_interval=inter_stimulus_interval,
        )
        nwbfile.add_lab_meta_data(spiral_scanning)

        # Add optogenetic stimulus site to NWBFile
        stimulus_site_metadata = metadata_copy["Ophys"]["OptogeneticStimulusSite"][0]
        stimulus_site_metadata.update(device=device)
        stim_site = PatternedOptogeneticStimulusSite(**stimulus_site_metadata)
        nwbfile.add_ogen_site(stim_site)

        # Add SLM device to NWBFile
        spatial_light_modulator_name = "spatial_light_modulator"
        if spatial_light_modulator_name in nwbfile.devices:
            raise ValueError(
                f"'{spatial_light_modulator_name}' already added to the NWBFile."
            )
        spatial_light_modulator_metadata = next(
            slm_metadata
            for slm_metadata in metadata["Ophys"]["Device"]
            if slm_metadata["name"] == spatial_light_modulator_name
        )
        spatial_light_modulator = SpatialLightModulator(**spatial_light_modulator_metadata)
        nwbfile.add_device(spatial_light_modulator)

        # Add light source device to NWBFile
        light_source_name = "light_source"
        if light_source_name in nwbfile.devices:
            raise ValueError(
                f"'{light_source_name}' already added to the NWBFile."
            )
        light_source_metadata = next(
            device_metadata
            for device_metadata in metadata["Ophys"]["Device"]
            if device_metadata["name"] == light_source_name
        )
        light_source = LightSource(**light_source_metadata)
        nwbfile.add_device(light_source)

        # Add plane segmentation to NWBFile
        add_image_segmentation(nwbfile=nwbfile, metadata=metadata_copy)

        ophys = get_module(nwbfile, "ophys")

        image_segmentation_metadata = metadata_copy["Ophys"]["ImageSegmentation"]
        image_segmentation_name = image_segmentation_metadata["name"]
        image_segmentation = ophys.get_data_interface(image_segmentation_name)

        plane_segmentation_metadata = next(
            (
                plane_segmentation_metadata
                for plane_segmentation_metadata in image_segmentation_metadata["plane_segmentations"]
                if plane_segmentation_metadata["name"] == plane_segmentation_name
            ),
            None,
        )
        plane_segmentation_kwargs = deepcopy(plane_segmentation_metadata)

        imaging_plane = nwbfile.imaging_planes[imaging_plane_name]
        plane_segmentation_kwargs.update(imaging_plane=imaging_plane)

        timestamps = self.get_timestamps()
        data = self.get_holographic_series(timestamps=timestamps)

        roi_ids = np.arange(0, self.num_rois)
        rois_dict = self.rois_dict
        centroids = []
        for roi_id in roi_ids:
            roi_data = rois_dict[roi_id]
            centroids.append([float(roi_data[0]["X"]), float(roi_data[0]["Y"])])

        plane_segmentation = PlaneSegmentation(id=roi_ids, **plane_segmentation_kwargs)

        plane_segmentation.add_column(
            name="ROICentroids",
            description="The x, y, centroids of each ROI.",
            data=np.array(centroids),
        )

        image_segmentation.add_plane_segmentation(plane_segmentations=[plane_segmentation])

        roi_table_region = plane_segmentation.create_roi_table_region(
            region=list(roi_ids), description="The stimulated ROIs"
        )

        holographic_series = PatternedOptogeneticSeries(
            name="HolographicStimulationSeries",
            description="The holographic stimulation for each ROI.",
            data=data,
            unit="watts",
            timestamps=timestamps,
            rois=roi_table_region,
            stimulus_pattern=spiral_scanning,
            site=stim_site,
            device=device,
            light_source=light_source,
            spatial_light_modulator=spatial_light_modulator,
        )
        nwbfile.add_stimulus(holographic_series)
