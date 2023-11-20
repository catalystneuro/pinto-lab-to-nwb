"""Primary NWBConverter class for this dataset."""
from typing import Optional

from neuroconv import NWBConverter
from neuroconv.datainterfaces import Suite2pSegmentationInterface, BrukerTiffMultiPlaneImagingInterface
from neuroconv.converters import BrukerTiffSinglePlaneConverter, BrukerTiffMultiPlaneConverter
from neuroconv.utils import FolderPathType


def get_default_imaging_to_segmentation_name_mapping(
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

    available_channels = Suite2pSegmentationInterface.get_available_channels(folder_path=segmentation_folder_path)
    available_planes = Suite2pSegmentationInterface.get_available_planes(folder_path=segmentation_folder_path)

    if len(available_planes) == 1 and len(available_channels) == 1:
        return None

    plane_default_mapping = dict()
    for channel_name, mapped_channel_name in zip(streams["channel_streams"], available_channels):
        mapped_values = [
            f"{mapped_channel_name.capitalize()}{plane_name.capitalize()}" for plane_name in available_planes
        ]
        if len(available_planes) > 1:
            plane_default_mapping.update(dict(zip(mapped_values, streams["plane_streams"][channel_name])))
        else:
            plane_default_mapping.update({mapped_values[0]: channel_name})

    return plane_default_mapping


class IntoTheVoidNWBConverter(NWBConverter):
    """Primary conversion class for the Two Photon Imaging (Bruker experiment)."""

    def __init__(
        self,
        imaging_folder_path: FolderPathType,
        verbose: bool = False,
        segmentation_folder_path: Optional[FolderPathType] = None,
        imaging_to_segmentation_plane_map: dict = None,
    ):
        self.verbose = verbose
        self.data_interface_objects = dict()

        self.imaging_to_segmentation_plane_map = imaging_to_segmentation_plane_map

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
                    plane_map = self.imaging_to_segmentation_plane_map
                    segmentation_interface_name = f"Segmentation{plane_name_suffix}"
                    segmentation_source_data = dict(
                        folder_path=segmentation_folder_path,
                        channel_name=channel_name,
                        plane_name=plane_name,
                        verbose=verbose,
                    )
                    if plane_map:
                        segmentation_source_data.update(
                            plane_segmentation_name=plane_map.get(plane_name_suffix, None).replace("_", "")
                        )
                    Suite2pSegmentationInterface(**segmentation_source_data)
                    self.data_interface_objects.update(
                        {segmentation_interface_name: Suite2pSegmentationInterface(**segmentation_source_data)}
                    )
