"""Primary NWBConverter class for this dataset."""
from typing import Optional

from ndx_pinto_metadata import SubjectExtension
from neuroconv import NWBConverter
from neuroconv.datainterfaces import Suite2pSegmentationInterface, BrukerTiffMultiPlaneImagingInterface
from neuroconv.converters import BrukerTiffSinglePlaneConverter, BrukerTiffMultiPlaneConverter
from neuroconv.utils import FilePathType, FolderPathType, DeepDict
from pynwb import NWBFile

from pinto_lab_to_nwb.behavior.interfaces import ViRMENBehaviorInterface, ViRMENTemporalAlignmentBehaviorInterface


def get_default_segmentation_to_imaging_name_mapping(
    imaging_folder_path: FolderPathType, segmentation_folder_path: FolderPathType
) -> dict or None:
    """
    Get the default mapping between imaging and segmentation planes.

    Parameters
    ----------
    imaging_folder_path: FolderPathType
        The folder path that contains the Bruker TIF imaging output (.ome.tif files).
    segmentation_folder_path: FolderPathType
        The folder that contains the Suite2P segmentation output. (usually named "suite2p")
    """
    streams = BrukerTiffMultiPlaneImagingInterface.get_streams(
        folder_path=imaging_folder_path,
        plane_separation_type="disjoint",
    )

    plane_streams = [
        plane_name for channel_name in streams["plane_streams"] for plane_name in streams["plane_streams"][channel_name]
    ]

    available_channels = Suite2pSegmentationInterface.get_available_channels(folder_path=segmentation_folder_path)
    available_planes = Suite2pSegmentationInterface.get_available_planes(folder_path=segmentation_folder_path)

    segmentation_channel_plane_names = [
        f"{channel_name.capitalize()}{plane_name.capitalize()}"
        for plane_name in available_planes
        for channel_name in available_channels
    ]

    num_channels = len(streams["channel_streams"])
    num_planes = 1 if not plane_streams else len(plane_streams)
    if num_channels == 1 and num_planes == 1:
        imaging_channel_plane_names = [None]
    else:
        imaging_channel_plane_names = plane_streams if num_planes != 1 else streams["channel_streams"]

    segmentation_to_imaging_name_mapping = dict(zip(segmentation_channel_plane_names, imaging_channel_plane_names))

    return segmentation_to_imaging_name_mapping


class IntoTheVoidNWBConverter(NWBConverter):
    """Primary conversion class for the Two Photon Imaging (Bruker experiment)."""

    def __init__(
        self,
        imaging_folder_path: FolderPathType,
        verbose: bool = False,
        segmentation_folder_path: Optional[FolderPathType] = None,
        segmentation_to_imaging_map: dict = None,
        virmen_file_path: Optional[FilePathType] = None,
        two_photon_time_sync_file_path: Optional[FilePathType] = None,
        two_photon_time_sync_struct_name: Optional[str] = None,
        im_frame_timestamps_name: Optional[str] = None,
    ):
        self.verbose = verbose
        self.data_interface_objects = dict()

        self.plane_map = segmentation_to_imaging_map

        streams = BrukerTiffMultiPlaneImagingInterface.get_streams(
            folder_path=imaging_folder_path,
            plane_separation_type="disjoint",
        )
        # Choose converter for Bruker depending on the number of planes
        # For multiple planes use BrukerTiffMultiPlaneConverter
        if streams["plane_streams"]:
            self.data_interface_objects.update(
                Imaging=BrukerTiffMultiPlaneConverter(
                    folder_path=imaging_folder_path,
                    plane_separation_type="disjoint",
                    verbose=verbose,
                ),
            )

        else:
            self.data_interface_objects.update(
                Imaging=BrukerTiffSinglePlaneConverter(folder_path=imaging_folder_path, verbose=verbose),
            )

        if segmentation_folder_path:
            available_planes = Suite2pSegmentationInterface.get_available_planes(folder_path=segmentation_folder_path)
            available_channels = Suite2pSegmentationInterface.get_available_channels(
                folder_path=segmentation_folder_path
            )
            # Add first channel
            for plane_name in available_planes:
                for channel_name in available_channels:
                    # check additional channel
                    if channel_name == "chan2":
                        # check we have non-empty traces
                        interface = Suite2pSegmentationInterface(
                            folder_path=segmentation_folder_path,
                            channel_name=channel_name,
                            verbose=verbose,
                        )
                        traces_to_add = interface.segmentation_extractor.get_traces_dict()
                        any_has_traces = any(
                            [bool(trace.size) for trace_name, trace in traces_to_add.items() if trace is not None]
                        )
                        if not any_has_traces:
                            continue

                    plane_name_suffix = f"{channel_name.capitalize()}{plane_name.capitalize()}"
                    segmentation_interface_name = f"Segmentation{plane_name_suffix}"
                    segmentation_source_data = dict(
                        folder_path=segmentation_folder_path,
                        channel_name=channel_name,
                        plane_name=plane_name,
                        verbose=verbose,
                    )
                    if self.plane_map:
                        mapped_plane_suffix = self.plane_map.get(plane_name_suffix, None)
                        plane_segmentation_name = "PlaneSegmentation"
                        if mapped_plane_suffix is not None:
                            plane_segmentation_name = "PlaneSegmentation" + mapped_plane_suffix.replace("_", "")
                        segmentation_source_data.update(
                            plane_segmentation_name=plane_segmentation_name,
                        )
                    Suite2pSegmentationInterface(**segmentation_source_data)
                    self.data_interface_objects.update(
                        {segmentation_interface_name: Suite2pSegmentationInterface(**segmentation_source_data)}
                    )

        if virmen_file_path:
            behavior_interface = ViRMENBehaviorInterface(file_path=virmen_file_path, verbose=verbose)
            self.data_interface_objects.update(BehaviorViRMEN=behavior_interface)

        if two_photon_time_sync_file_path:
            time_alignment_behavior_interface = ViRMENTemporalAlignmentBehaviorInterface(
                file_path=str(two_photon_time_sync_file_path),
                sync_data_struct_name=two_photon_time_sync_struct_name,
                im_frame_timestamps_name=im_frame_timestamps_name,
            )
            self.data_interface_objects.update(BehaviorViRMENTwoPhotonTimeAligned=time_alignment_behavior_interface)

    def get_metadata(self) -> DeepDict:
        imaging_metadata = self.data_interface_objects["Imaging"].get_metadata()
        metadata = super().get_metadata()

        # override device metadata
        device_metadata = imaging_metadata["Ophys"]["Device"]
        device_name = device_metadata[0]["name"]
        metadata["Ophys"]["Device"] = device_metadata

        for metadata_ind in range(len(imaging_metadata["Ophys"]["ImagingPlane"])):
            optical_channel_metadata = imaging_metadata["Ophys"]["ImagingPlane"][metadata_ind]["optical_channel"]
            # override optical channel metadata
            metadata["Ophys"]["ImagingPlane"][metadata_ind]["optical_channel"] = optical_channel_metadata

            # override device link
            metadata["Ophys"]["ImagingPlane"][metadata_ind]["device"] = device_name

        return metadata

    def temporally_align_data_interfaces(self):
        if "BehaviorViRMENTwoPhotonTimeAligned" not in self.data_interface_objects:
            return

        frame_time_aligned_behavior_interface = self.data_interface_objects["BehaviorViRMENTwoPhotonTimeAligned"]
        aligned_timestamps = frame_time_aligned_behavior_interface.get_timestamps()
        aligned_starting_time = aligned_timestamps[0]

        # set aligned starting time for segmentation interfaces
        for interface_name, interface in self.data_interface_objects.items():
            if interface_name == "Imaging":
                # set aligned starting time interfaces
                imaging_converter = self.data_interface_objects["Imaging"]
                for _, imaging_interface in imaging_converter.data_interface_objects.items():
                    imaging_interface.set_aligned_starting_time(aligned_starting_time=aligned_starting_time)
            elif interface_name.startswith("Segmentation"):
                segmentation_interface = self.data_interface_objects[interface_name]
                segmentation_interface.set_aligned_starting_time(aligned_starting_time=aligned_starting_time)

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata, conversion_options: Optional[dict] = None) -> None:
        super().add_to_nwbfile(nwbfile=nwbfile, metadata=metadata, conversion_options=conversion_options)

        # Add subject (from extension)
        if metadata["SubjectExtension"] is not None:
            nwbfile.subject = SubjectExtension(**metadata["SubjectExtension"])
