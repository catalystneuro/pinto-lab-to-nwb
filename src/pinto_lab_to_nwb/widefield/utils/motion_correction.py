from pathlib import Path
from typing import List

import natsort
import numpy as np
import pymatreader


def load_motion_correction_data(file_paths: List[str]) -> np.ndarray:
    """Load motion correction data from mat files.

    Parameters
    ----------
    file_paths: List[str]
        A list of paths to the motion correction files.

    Returns
    -------
    motion_correction_data: numpy.ndarray
        The concatenated yx shifts from all the files.
    """
    motion_correction_data = np.concatenate(
        [get_yx_shifts(file_path=str(file)) for file in file_paths], axis=0)
    return motion_correction_data


def get_yx_shifts(file_path: str) -> np.ndarray:
    """Get the yx shifts from the motion correction file.

    Parameters
    ----------
    file_path: str
        The path to the motion correction file.

    Returns
    -------
    motion_correction: numpy.ndarray
        The first column is the y shifts. The second column is the x shifts.
    """
    motion_correction_data = pymatreader.read_mat(file_path)
    motion_correction = np.column_stack((motion_correction_data["yShifts"], motion_correction_data["xShifts"]))
    return motion_correction


folder_path = Path("/Users/weian/data/widefield")
files = natsort.natsorted(folder_path.glob("*mcorr_1.mat"))
print(len(files))
for file in files:
    xy = get_yx_shifts(file_path=str(file))
    if not np.any(xy):
        continue
    for t in xy:
        if np.any(t):
            print(t)
    print(file)
