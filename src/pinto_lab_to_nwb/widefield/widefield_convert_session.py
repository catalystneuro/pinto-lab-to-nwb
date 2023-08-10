"""Primary script to run to convert an entire session for of data using the NWBConverter."""
import re
from pathlib import Path
from dateutil import tz
from neuroconv.utils import (
    load_dict_from_file,
    dict_deep_update,
    FolderPathType,
    FilePathType,
)

from pinto_lab_to_nwb.widefield import WideFieldNWBConverter


def session_to_nwb(
    nwbfile_path: FilePathType,
    widefield_imaging_folder_path: FolderPathType,
    stub_test: bool = False,
):
    """
    Converts a single session to NWB.

    Parameters
    ----------
    nwbfile_path : FilePathType
        The file path to the NWB file that will be created.
    widefield_imaging_folder_path: FolderPathType
        The folder path that contains the Micro-Manager OME-TIF imaging output (.ome.tif files).
    stub_test: bool, optional
        For testing purposes, when stub_test=True only writes a subset of imaging and segmentation data.
    """
    widefield_imaging_folder_path = Path(widefield_imaging_folder_path)

    source_data = dict()
    conversion_options = dict()

    # Add Imaging
    imaging_source_data = dict(folder_path=str(widefield_imaging_folder_path))
    source_data.update(dict(Imaging=imaging_source_data))
    conversion_options.update(dict(Imaging=dict(stub_test=stub_test)))

    converter = WideFieldNWBConverter(source_data=source_data)

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
    # TS12_20220407_20hz_noteasy_1
    file_naming_pattern = r"^(?P<subject_id>[^_]+)_(?P<session_id>.+)"
    match = re.match(file_naming_pattern, str(widefield_imaging_folder_path.name))
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
    imaging_folder_path = Path("/Volumes/t7-ssd/Pinto/TS12_20220407_20hz_noteasy_1")
    nwbfile_path = Path("/Volumes/t7-ssd/Pinto/nwbfiles/widefield/TS12_20220407_20hz_noteasy_1.nwb")
    stub_test = False

    session_to_nwb(
        nwbfile_path=nwbfile_path,
        widefield_imaging_folder_path=imaging_folder_path,
        stub_test=stub_test,
    )
