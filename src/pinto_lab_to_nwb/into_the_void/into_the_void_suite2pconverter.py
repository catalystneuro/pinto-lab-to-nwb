from typing import Optional

from neuroconv import NWBConverter
from neuroconv.tools.nwb_helpers import make_or_load_nwbfile
from neuroconv.utils import FolderPathType, get_schema_from_method_signature
from pynwb import NWBFile

from pinto_lab_to_nwb.into_the_void.into_the_void_suite2psegmentationinterface import (
    IntoTheVoidSuite2pSegmentationInterface,
)


class IntoTheVoidSuite2pConverter(NWBConverter):
    @classmethod
    def get_source_schema(cls):
        return get_schema_from_method_signature(cls)

    def get_conversion_options_schema(self):
        interface_name = list(self.data_interface_objects.keys())[0]
        return self.data_interface_objects[interface_name].get_conversion_options_schema()

    def __init__(
        self,
        folder_path: FolderPathType,
        verbose: bool = False,
    ):
        """
        Initializes the data interfaces for Bruker imaging data stream.
        Parameters
        ----------
        folder_path : PathType
            The path to the folder that contains the Bruker TIF image files (.ome.tif) and configuration files (.xml, .env).
        verbose : bool, default: True
            Controls verbosity.
        """
        self.verbose = verbose
        self.data_interface_objects = dict()

        streams = IntoTheVoidSuite2pSegmentationInterface.get_streams(folder_path=folder_path)

        interface_name = "Segmentation"
        for stream_ind, stream_name in enumerate(streams):
            if len(streams) > 1:
                interface_name += stream_name
            self.data_interface_objects[interface_name] = IntoTheVoidSuite2pSegmentationInterface(
                folder_path=folder_path,
                stream_name=stream_name,
            )

    def add_to_nwbfile(
        self,
        nwbfile: NWBFile,
        metadata,
        stub_test: bool = False,
        stub_frames: int = 100,
        plane_index: int = 0,
        include_roi_centroids: bool = True,
        include_roi_acceptance: bool = True,
    ):
        for plane_index, (interface_name, data_interface) in enumerate(self.data_interface_objects.items()):
            data_interface.add_to_nwbfile(
                nwbfile=nwbfile,
                metadata=metadata,
                plane_index=plane_index,
                stub_test=stub_test,
                stub_frames=stub_frames,
                include_roi_centroids=include_roi_centroids,
                include_roi_acceptance=include_roi_acceptance,
            )

    def run_conversion(
        self,
        nwbfile_path: Optional[str] = None,
        nwbfile: Optional[NWBFile] = None,
        metadata: Optional[dict] = None,
        overwrite: bool = False,
        stub_test: bool = False,
        stub_frames: int = 100,
    ) -> None:
        if metadata is None:
            metadata = self.get_metadata()

        self.validate_metadata(metadata=metadata)

        self.temporally_align_data_interfaces()

        with make_or_load_nwbfile(
            nwbfile_path=nwbfile_path,
            nwbfile=nwbfile,
            metadata=metadata,
            overwrite=overwrite,
            verbose=self.verbose,
        ) as nwbfile_out:
            self.add_to_nwbfile(nwbfile=nwbfile_out, metadata=metadata, stub_test=stub_test, stub_frames=stub_frames)


# imaging_folder_path = Path("/Volumes/t7-ssd/Pinto/NCCR32_2022_11_03_IntoTheVoid_t_series-005")
# segmentation_folder_path = imaging_folder_path / "suite2p"
#
# converter = Suite2pConverter(folder_path=segmentation_folder_path)
# metadata = converter.get_metadata()
# converter.run_conversion(nwbfile_path="test.nwb", metadata=metadata, stub_test=True)
