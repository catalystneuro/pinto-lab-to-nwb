import numpy as np
from scipy.io import loadmat
from scipy.io.matlab import mat_struct


def convert_mat_file_to_dict(mat_file_name):
    """
    Convert mat-file to dictionary object.

    It calls a recursive function to convert all entries
    that are still matlab objects to dictionaries.
    """
    data = loadmat(mat_file_name, struct_as_record=False, squeeze_me=True)
    for key in data:
        if isinstance(data[key], mat_struct):
            data[key] = mat_obj_to_dict(data[key])
    return data


def mat_obj_to_dict(mat):
    """Recursive function to convert nested matlab struct objects to dictionaries."""
    dict_from_struct = {}
    for field_name in mat.__dict__["_fieldnames"]:
        dict_from_struct[field_name] = mat.__dict__[field_name]
        if isinstance(dict_from_struct[field_name], mat_struct):
            dict_from_struct[field_name] = mat_obj_to_dict(dict_from_struct[field_name])
        elif isinstance(dict_from_struct[field_name], np.ndarray):
            try:
                dict_from_struct[field_name] = mat_obj_to_array(dict_from_struct[field_name])
            except TypeError:
                continue
    return dict_from_struct


def mat_obj_to_array(mat_struct_array):
    """Construct array from matlab cell arrays.
    Recursively converts array elements if they contain mat objects."""
    if has_struct(mat_struct_array):
        array_from_cell = [mat_obj_to_dict(mat_struct) for mat_struct in mat_struct_array]
        array_from_cell = np.array(array_from_cell)
    else:
        array_from_cell = mat_struct_array

    return array_from_cell


def has_struct(mat_struct_array):
    """Determines if a matlab cell array contains any mat objects."""
    return any(isinstance(mat, mat_struct) for mat in mat_struct_array)
