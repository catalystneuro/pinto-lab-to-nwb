import numpy as np
import pandas as pd
from neuroconv.utils import FilePathType


def load_timestamps(timestamps_file_path: FilePathType) -> np.ndarray:
    timestamps_data = pd.read_csv(timestamps_file_path)
    assert (
        "im_frame_timestamps" in timestamps_data.columns
    ), f"im_frame_timestamps not in file '{timestamps_file_path}'."
    timestamps = timestamps_data["im_frame_timestamps"].values
    return timestamps
