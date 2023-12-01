"""Primary NWBConverter class for this dataset."""
from pathlib import Path
from typing import Optional

from neuroconv import NWBConverter
from neuroconv.datainterfaces import Suite2pSegmentationInterface, BrukerTiffMultiPlaneImagingInterface
from neuroconv.converters import BrukerTiffSinglePlaneConverter, BrukerTiffMultiPlaneConverter

from pinto_lab_to_nwb.into_the_void.interfaces import HolographicStimulationInterface
from neuroconv.utils import FolderPathType, DeepDict, dict_deep_update


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
        imaging_channel_plane_names = plane_streams if num_planes == 1 else streams["channel_streams"]

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

        if list(Path(imaging_folder_path).glob("*MarkPoints*.xml")):
            self.data_interface_objects.update(
                HolographicStimulation=HolographicStimulationInterface(
                    folder_path=imaging_folder_path, verbose=verbose
                ),
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

        if "HolographicStimulation" in self.data_interface_objects:
            holographic_metadata = self.data_interface_objects["HolographicStimulation"].get_metadata()
            metadata = dict_deep_update(metadata, holographic_metadata)

        return metadata
