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

from pinto_lab_to_nwb.general import make_subject_metadata
from pinto_lab_to_nwb.widefield import WideFieldNWBConverter


def session_to_nwb(
    nwbfile_path: FilePathType,
    widefield_imaging_folder_path: FolderPathType,
    strobe_sequence_file_path: FilePathType,
    processed_imaging_file_path: FilePathType,
    info_file_path: FilePathType,
    vasculature_mask_file_path: FilePathType,
    manual_mask_file_path: FilePathType,
    manual_mask_struct_name: str,
    roi_from_ref_mat_file_path: FilePathType,
    binned_vasculature_mask_file_path: FilePathType,
    binned_blue_pca_mask_file_path: FilePathType,
    binned_violet_pca_mask_file_path: FilePathType,
    subject_metadata_file_path: Optional[FilePathType] = None,
    virmen_file_path: Optional[FilePathType] = None,
    widefield_time_sync_file_path: Optional[FilePathType] = None,
    widefield_time_sync_struct_name: Optional[str] = None,
    im_frame_timestamps_name: Optional[str] = None,
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
    strobe_sequence_file_path: FilePathType
        The file path to the strobe sequence file. This file should contain the 'strobe_session_key' key.
    info_file_path: FilePathType
        The file path to the Matlab file with information about the imaging session (e.g. 'frameRate').
    vasculature_mask_file_path: FilePathType
        The file path that contains the contrast based vasculature mask on the full size session image (blue channel).
    manual_mask_file_path: FilePathType
        The file path that contains the manual mask on the full size session image (blue channel).
    manual_mask_struct_name: str
        The name of the struct in the manual mask file that contains the manual mask (e.g. "regMask" or "reg_manual_mask").
    roi_from_ref_mat_file_path: FilePathType
        The file path that contains the Allen area label of each pixel mapped onto the reference image of the mouse and registered to the session.
    binned_vasculature_mask_file_path: FilePathType
        The file path that contains the contrast based vasculature mask on the downsampled (binned) session image (blue channel).
    binned_blue_pca_mask_file_path: FilePathType
        The file path that contains the PCA mask for the blue channel.
    binned_violet_pca_mask_file_path: FilePathType
        The file path that contains the PCA mask for the violet channel.
    subject_metadata_file_path: FilePathType, optional
        The file path to the subject metadata file. This file should contain the 'metadata' key.
    virmen_file_path: FilePathType, optional
        The file path to the ViRMEN .mat file.
    widefield_time_sync_file_path: FilePathType, optional
        The file path to that points to the .mat file containing the timestamps for the imaging data.
        These timestamps are used to set the times of the widefield imaging data in the NWB file.
    widefield_time_sync_struct_name: str, optional
        The name of the sync data struct in the .mat file. (e.g. "wf_behav_sync_data")
    im_frame_timestamps_name: str, optional
        The name of the variable in the .mat file that contains the aligned timestamps for the imaging frames.
    stub_test: bool, optional
        For testing purposes, when stub_test=True only writes a subset of imaging and segmentation data.
    """
    widefield_imaging_folder_path = Path(widefield_imaging_folder_path)

    source_data = dict()
    conversion_options = dict()

    # Add Imaging
    imaging_blue_source_data = dict(
        folder_path=str(widefield_imaging_folder_path),
        strobe_sequence_file_path=str(strobe_sequence_file_path),
        channel_name="blue",
    )
    imaging_violet_source_data = dict(
        folder_path=str(widefield_imaging_folder_path),
        strobe_sequence_file_path=str(strobe_sequence_file_path),
        channel_name="violet",
    )

    source_data.update(dict(ImagingBlue=imaging_blue_source_data, ImagingViolet=imaging_violet_source_data))

    conversion_options.update(
        dict(
            ImagingBlue=dict(stub_test=stub_test, photon_series_index=0, photon_series_type="OnePhotonSeries"),
            ImagingViolet=dict(stub_test=stub_test, photon_series_index=1, photon_series_type="OnePhotonSeries"),
        ),
    )

    # Add processed imaging
    source_data.update(
        dict(
            ProcessedImagingBlue=dict(
                file_path=str(processed_imaging_file_path),
                info_file_path=str(info_file_path),
                strobe_sequence_file_path=str(strobe_sequence_file_path),
                channel_name="blue",
            )
        ),
        ProcessedImagingViolet=dict(
            file_path=str(processed_imaging_file_path),
            info_file_path=str(info_file_path),
            strobe_sequence_file_path=str(strobe_sequence_file_path),
            channel_name="violet",
        ),
    )
    conversion_options.update(
        dict(
            ProcessedImagingBlue=dict(
                stub_test=stub_test,
                parent_container="processing/ophys",
                photon_series_type="OnePhotonSeries",
                photon_series_index=2,
            ),
            ProcessedImagingViolet=dict(
                stub_test=stub_test,
                parent_container="processing/ophys",
                photon_series_type="OnePhotonSeries",
                photon_series_index=3,
            ),
        ),
    )

    # Add segmentation and summary images for the blue and violet channels
    source_data.update(
        dict(
            SegmentationProcessedBlue=dict(
                info_mat_file_path=str(info_file_path),
                roi_from_ref_mat_file_path=str(roi_from_ref_mat_file_path),
                vasculature_mask_file_path=str(binned_vasculature_mask_file_path),
                blue_pca_mask_file_path=str(binned_blue_pca_mask_file_path),
            ),
            SummaryImagesBlue=dict(
                vasculature_mask_file_path=str(vasculature_mask_file_path),
                manual_mask_file_path=str(manual_mask_file_path),
                manual_mask_struct_name=manual_mask_struct_name,
            ),
            SummaryImagesViolet=dict(
                violet_pca_mask_file_path=str(binned_violet_pca_mask_file_path),
            ),
        )
    )

    # Add behavior
    if virmen_file_path:
        source_data.update(BehaviorViRMEN=dict(file_path=str(virmen_file_path)))

    if widefield_time_sync_file_path:
        time_alignment_behavior_source_data = dict(
            file_path=str(widefield_time_sync_file_path),
            sync_data_struct_name=widefield_time_sync_struct_name,
            im_frame_timestamps_name=im_frame_timestamps_name,
        )
        source_data.update(BehaviorViRMENWidefieldTimeAligned=time_alignment_behavior_source_data)

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

        if subject_metadata_file_path:
            subject_metadata = make_subject_metadata(
                subject_id=groups_dict["subject_id"], subject_metadata_file_path=subject_metadata_file_path
            )
            metadata = dict_deep_update(metadata, subject_metadata)

    # Separate subject metadata from NWBFile metadata to add SubjectExtension
    metadata["SubjectExtension"] = metadata.pop("Subject", None)
    # Run conversion
    converter.run_conversion(
        nwbfile_path=nwbfile_path, metadata=metadata, overwrite=True, conversion_options=conversion_options
    )


if __name__ == "__main__":
    # Parameters for conversion

    # The folder path that contains the raw imaging data in Micro-Manager OME-TIF format (.ome.tif files).
    imaging_folder_path = Path("/Users/weian/data/Cherry/20230802/Cherry_20230802_20hz_1")
    # imaging_folder_path = Path("/Volumes/t7-ssd/Pinto/DrChicken_20230419_20hz")
    # The file path to the strobe sequence file.
    strobe_sequence_file_path = imaging_folder_path / "strobe_seq_1_2.mat"
    # The file path to the downsampled imaging data in Matlab format (.mat file).
    processed_imaging_path = imaging_folder_path / "rawf_full.mat"
    # The file path to the Matlab file with information about the imaging session (e.g. 'frameRate').
    info_file_path = imaging_folder_path / "info.mat"

    # Parameters for custom segmentation and summary images
    # The file path that contains the contrast based vasculature mask on the full size session image (blue channel).
    vasculature_mask_file_path = imaging_folder_path / "vasculature_mask_2.mat"

    # The file path that contains the manual mask on the full size session image (blue channel).
    manual_mask_file_path = imaging_folder_path / "reg_manual_mask_jlt6316_Cherry_20230802_1_1_1.mat"
    # manual_mask_file_path = imaging_folder_path / "regManualMask.mat"
    # The name of the struct in the manual mask file that contains the manual mask (e.g. "regMask" or "reg_manual_mask").
    manual_mask_struct_name = "reg_manual_mask"
    # manual_mask_struct_name = "regMask"

    # The file path that contains the Allen area label of each pixel mapped onto the reference image of the mouse and registered to the session.
    roi_from_ref_mat_file_path = imaging_folder_path / "ROIfromRef_1.mat"
    # roi_from_ref_mat_file_path = imaging_folder_path / "ROIfromRef.mat"

    # The file path that contains the contrast based vasculature mask on the downsampled (binned) session image (blue channel).
    binned_vasculature_mask_file_path = imaging_folder_path / "vasculature_mask_2.mat"

    # The file path that contains the PCA mask for the blue channel.
    binned_blue_pca_mask_file_path = imaging_folder_path / "blue_pca_vasculature_mask_2.mat"

    # The file path that contains the PCA mask for the violet channel.
    binned_violet_pca_mask_file_path = imaging_folder_path / "violet_pca_vasculature_mask_2.mat"

    subject_metadata_file_path = "/Volumes/t7-ssd/Pinto/Behavior/subject_metadata.mat"

    # The file path to the ViRMEN .mat file.
    virmen_file_path = "/Volumes/t7-ssd/Pinto/Behavior/DrChicken_TowersTaskSwitchEasy_Session_20230419_105733.mat"
    # The file path to that points to the .mat file containing the timestamps for the behavior data.
    behavior_timestamps_file_path = "/Volumes/t7-ssd/Pinto/eyetracking/Cherry_20230801/sync_data.csv"

    widefield_time_sync_file_path = imaging_folder_path / "wf_behav_sync.mat"

    # Parameters for the Widefield time alignment
    # Path to the aligned timestamps for the eye tracking data.
    sync_data_file_path = "/Volumes/t7-ssd/Pinto/eyetracking/Cherry_20230802/sync_data.csv"
    # The name of the struct in the .mat file that contains the timestamps for the imaging data.
    widefield_time_sync_struct_name = "wf_behav_sync_data"
    # The name of the variable in the .mat file that contains the aligned timestamps for the imaging frames.
    im_frame_timestamps_name = "im_frame_timestamps"

    # The file path to the NWB file that will be created.
    nwbfile_path = Path("/Users/weian/data/full_Cherry_20230802.nwb")

    stub_test = False

    session_to_nwb(
        nwbfile_path=nwbfile_path,
        widefield_imaging_folder_path=imaging_folder_path,
        strobe_sequence_file_path=strobe_sequence_file_path,
        processed_imaging_file_path=processed_imaging_path,
        info_file_path=info_file_path,
        vasculature_mask_file_path=vasculature_mask_file_path,
        manual_mask_file_path=manual_mask_file_path,
        manual_mask_struct_name=manual_mask_struct_name,
        roi_from_ref_mat_file_path=roi_from_ref_mat_file_path,
        binned_vasculature_mask_file_path=binned_vasculature_mask_file_path,
        binned_blue_pca_mask_file_path=binned_blue_pca_mask_file_path,
        binned_violet_pca_mask_file_path=binned_violet_pca_mask_file_path,
        subject_metadata_file_path=subject_metadata_file_path,
        virmen_file_path=virmen_file_path,
        widefield_time_sync_file_path=widefield_time_sync_file_path,
        widefield_time_sync_struct_name=widefield_time_sync_struct_name,
        im_frame_timestamps_name=im_frame_timestamps_name,
        stub_test=stub_test,
    )
