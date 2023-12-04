# Notes concerning the `widefield` conversion

## Imaging folder structure

See the example folder structure [here](https://gin.g-node.org/CatalystNeuro/ophys_testing_data/src/main/imaging_datasets/MicroManagerTif) for the MicroManager OME-TIF format.

## Segmentation files structure

Required files for custom segmentation and summary images:

Summary images for the full size session image (blue channel):
- `"vasculature_mask_file_path"`: The file path that contains the contrast based vasculature mask on the full size session image (blue channel).
- `"manual_mask_file_path`: The file path that contains the manual mask on the full size session image (blue channel).
Summary images for the binned session image (blue channel, violet channel):
- `"binned_vasculature_mask_file_path"`: The file path that contains the contrast based vasculature mask on the downsampled (binned) session image (blue channel).
- `"binned_blue_pca_mask_file_path"`: The file path that contains the PCA mask for the blue channel.
- `"binned_violet_pca_mask_file_path"`: The file path that contains the PCA mask for the violet channel.
ROI masks and Allen area labels for the binned session image (blue channel):
- `"roi_from_ref_mat_file_path"`: The file path that contains the Allen area label of each pixel mapped onto the reference image of the mouse and registered to the session.
- `"info_file_path"`: The file path that contains the information about the imaging session (e.g. number of frames, frame rate, etc.).

## Run conversion for a single session

`widefield_convert_sesion.py`: this script defines the function to convert one full session of the conversion.
Parameters:
- "`widefield_imaging_folder_path`" : The folder path that contains the Micro-Manager OME-TIF imaging output (.ome.tif files).
- "`strobe_sequence_file_path`": The file path to the strobe sequence file. This file should contain the 'strobe_session_key' key.
- "`info_file_path`": The file path to the Matlab file with information about the imaging session (e.g. 'frameRate').
- "`vasculature_mask_file_path`": The file path that contains the contrast based vasculature mask on the full size session image (blue channel).
- "`manual_mask_file_path`": The file path that contains the manual mask on the full size session image (blue channel).
- "`manual_mask_struct_name`": The name of the struct in the manual mask file that contains the manual mask (e.g. "regMask" or "reg_manual_mask").
- "`roi_from_ref_mat_file_path`": The file path that contains the Allen area label of each pixel mapped onto the reference image of the mouse and registered to the session.
- "`binned_vasculature_mask_file_path`": The file path that contains the contrast based vasculature mask on the downsampled (binned) session image (blue channel).
- "`binned_blue_pca_mask_file_path`": The file path that contains the PCA mask for the blue channel.
- "`binned_violet_pca_mask_file_path`": The file path that contains the PCA mask for the violet channel.
- "`subject_metadata_file_path`": The file path that contains the subject metadata (e.g. subject_id, genotype, etc.).

### Example usage

To run a specific conversion, you might need to install first some conversion specific dependencies that are located in each conversion directory:
```
cd src/pinto_lab_to_nwb/widefield
pip install -r widefield_requirements.txt
```
Then you can run a specific conversion with the following command:
```
python widefield_convert_session.py
```

### Tutorials

See the [widefield tutorial](./tutorials/widefield_demo.ipynb) that demonstrates how to read and access the data in NWB.
