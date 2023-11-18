from datetime import datetime
from warnings import warn

from dateutil import tz
from pymatreader import read_mat


def make_subject_metadata(subject_id: str, subject_metadata_file_path: str) -> dict:
    subject_mat = read_mat(subject_metadata_file_path)

    assert "metadata" in subject_mat, f"The subject file '{subject_metadata_file_path}' should have a 'metadata' key."
    subject_metadata = subject_mat["metadata"]

    if subject_id not in subject_metadata["subject_nickname"]:
        warn(f"Subject '{subject_id}' not found in subject metadata file '{subject_metadata_file_path}'."
             f"The metadata for this subject will not be added to the NWB file.")
        return dict()

    subject_ind = subject_metadata["subject_nickname"].index(subject_id)
    date_of_birth = datetime.strptime(subject_metadata["dob"][subject_ind], '%Y-%m-%d')
    tzinfo = tz.gettz("US/Pacific")
    date_of_birth = date_of_birth.replace(tzinfo=tzinfo)

    sex_mapping = dict(Male="M", Female="F")
    subject_kwargs = dict(
        date_of_birth=date_of_birth,
        subject_id=subject_id,
        sex=sex_mapping[subject_metadata["sex"][subject_ind]],
        genotype=subject_metadata["line"][subject_ind],
        species="Mus musculus",
    )
    subject_description = subject_metadata["subject_description"][subject_ind]
    if subject_description:
        subject_kwargs["description"] = subject_description

    metadata = dict(Subject=subject_kwargs, NWBFile=dict(protocol=subject_metadata["protocol"][subject_ind]))
    return metadata
