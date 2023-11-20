"""Primary NWBConverter class for this dataset."""
from typing import Optional
from warnings import warn

from neuroconv import NWBConverter
from neuroconv.datainterfaces import Suite2pSegmentationInterface, BrukerTiffMultiPlaneImagingInterface
from neuroconv.converters import BrukerTiffSinglePlaneConverter, BrukerTiffMultiPlaneConverter
from neuroconv.datainterfaces.ophys.suite2p.suite2pdatainterface import _update_metadata_with_new_imaging_plane_name
from neuroconv.tools.nwb_helpers import get_default_nwbfile_metadata
from neuroconv.utils import DeepDict, dict_deep_update, FolderPathType


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
            plane_default_mapping.update(dict(zip(streams["plane_streams"][channel_name], mapped_values)))
        else:
            plane_default_mapping.update({channel_name: mapped_values[0]})

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
                    segmentation_interface_name = f"Segmentation{plane_name_suffix}"
                    segmentation_source_data = dict(
                        folder_path=segmentation_folder_path,
                        channel_name=channel_name,
                        plane_name=plane_name,
                        verbose=verbose,
                    )
                    self.data_interface_objects.update(
                        {segmentation_interface_name: Suite2pSegmentationInterface(**segmentation_source_data)}
                    )

    def get_metadata(self) -> DeepDict:
        if not self.imaging_to_segmentation_plane_map:
            return super().get_metadata()

        metadata = get_default_nwbfile_metadata()
        imaging_metadata = self.data_interface_objects["Imaging"].get_metadata()
        metadata = dict_deep_update(metadata, imaging_metadata)
        for imaging_plane_suffix, mapped_plane_suffix in self.imaging_to_segmentation_plane_map.items():
            imaging_plane_suffix = imaging_plane_suffix.replace("_", "")
            imaging_plane_name = next(
                (
                    imaging_plane_metadata["name"]
                    for imaging_plane_metadata in metadata["Ophys"]["ImagingPlane"]
                    if imaging_plane_suffix in imaging_plane_metadata["name"]
                ),
                None,
            )
            if imaging_plane_name is None:
                warn(
                    "Could not determine the name of the imaging plane to automatically update the segmentation metadata with."
                    "Please manually update the links in the metadata."
                )
                return super().get_metadata()

            segmentation_interface_name = f"Segmentation{mapped_plane_suffix}"
            if segmentation_interface_name not in self.data_interface_objects:
                continue
            segmentation_metadata = self.data_interface_objects[segmentation_interface_name].get_metadata()
            new_metadata = _update_metadata_with_new_imaging_plane_name(
                metadata=segmentation_metadata,
                imaging_plane_name=imaging_plane_name,
            )
            # override device name
            new_metadata["Ophys"]["Device"][0]["name"] = imaging_metadata["Ophys"]["Device"][0]["name"]
            new_metadata["Ophys"]["ImagingPlane"][0]["device"] = imaging_metadata["Ophys"]["Device"][0]["name"]
            metadata = dict_deep_update(metadata, new_metadata)

        return metadata

    def get_conversion_options(self, stub_test: bool = False) -> dict:
        """Automatically set conversion options for each data interface."""
        conversion_options = dict(Imaging=dict(stub_test=stub_test))
        segmentation_interfaces = [
            interface_name for interface_name in self.data_interface_objects.keys() if "Segmentation" in interface_name
        ]
        for interface_ind, interface_name in enumerate(segmentation_interfaces):
            # Automatically set plane_segmentation_name for the segmentation interfaces
            metadata = self.get_metadata()
            plane_segmentation_name = metadata["Ophys"]["ImageSegmentation"]["plane_segmentations"][interface_ind][
                "name"
            ]
            conversion_options[interface_name] = dict(
                plane_segmentation_name=plane_segmentation_name, stub_test=stub_test
            )
        return conversion_options
