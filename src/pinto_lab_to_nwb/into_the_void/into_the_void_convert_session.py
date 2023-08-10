"""Primary script to run to convert an entire session for of data using the NWBConverter."""
import re
from pathlib import Path
from typing import Optional
from dateutil import tz
from neuroconv.utils import (
    load_dict_from_file,
    dict_deep_update,
    FolderPathType,
    FilePathType,
)

from pinto_lab_to_nwb.into_the_void import IntoTheVoidNWBConverter


def session_to_nwb(
    nwbfile_path: FilePathType,
    two_photon_imaging_folder_path: FolderPathType,
    segmentation_folder_path: Optional[FolderPathType] = None,
    stub_test: bool = False,
):
    """
    Converts a single session to NWB.

    Parameters
    ----------
    nwbfile_path : FilePathType
        The file path to the NWB file that will be created.
    two_photon_imaging_folder_path: FolderPathType
        The folder path that contains the Bruker TIF imaging output (.ome.tif files).
    segmentation_folder_path: FolderPathType
        The folder that contains the Suite2P segmentation output.
    stub_test: bool, optional
        For testing purposes, when stub_test=True only writes a subset of imaging and segmentation data.
    """

    source_data = dict()
    conversion_options = dict()

    # Add Imaging
    imaging_source_data = dict(folder_path=str(two_photon_imaging_folder_path))
    source_data.update(dict(Imaging=imaging_source_data))
    conversion_options.update(dict(Imaging=dict(stub_test=stub_test)))

    # Add Segmentation (optional)
    if segmentation_folder_path:
        segmentation_source_data = dict(folder_path=str(segmentation_folder_path))
        source_data.update(dict(Segmentation=segmentation_source_data))
        conversion_options.update(dict(Segmentation=dict(stub_test=False)))

    converter = IntoTheVoidNWBConverter(source_data=source_data)

    # Add datetime to conversion
    metadata = converter.get_metadata()
    # For data provenance we can add the time zone information to the conversion if missing
    session_start_time = metadata["NWBFile"]["session_start_time"]
    tzinfo = tz.gettz("US/Pacific")
    metadata["NWBFile"].update(session_start_time=session_start_time.replace(tzinfo=tzinfo))

    # Update default metadata with the editable in the corresponding yaml file
    editable_metadata_path = Path(__file__).parent / "general_metadata.yaml"
    editable_metadata = load_dict_from_file(editable_metadata_path)
    metadata = dict_deep_update(metadata, editable_metadata)

    # Update metadata with subject_id and session_id from folder_path
    # NCCR51_2023_04_07_no_task_dual_color_jrgeco_t_series-001
    file_naming_pattern = r"^(?P<subject_id>[^_]+)_(?:\d{4}_\d{2}_\d{2}_)(?P<session_id>.+)"
    match = re.match(file_naming_pattern, str(two_photon_imaging_folder_path.name))
    if match:
        groups_dict = match.groupdict()
        metadata["NWBFile"].update(session_id=groups_dict["session_id"].replace("_", "-"))
        metadata["Subject"].update(subject_id=groups_dict["subject_id"])

    # Run conversion
    converter.run_conversion(
        nwbfile_path=nwbfile_path, metadata=metadata, overwrite=True, conversion_options=conversion_options
    )


if __name__ == "__main__":
    # Parameters for conversion
    imaging_folder_path = Path("/Volumes/t7-ssd/Pinto/NCCR62_2023_07_06_IntoTheVoid_t_series_Dual_color-000")
    segmentation_folder_path = imaging_folder_path / "suite2p"
    nwbfile_path = Path("/Volumes/t7-ssd/Pinto/nwbfiles/imaging_stub2.nwb")
    stub_test = True

    session_to_nwb(
        nwbfile_path=nwbfile_path,
        two_photon_imaging_folder_path=imaging_folder_path,
        segmentation_folder_path=segmentation_folder_path,
        stub_test=stub_test,
    )
