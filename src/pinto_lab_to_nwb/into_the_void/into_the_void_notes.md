# Notes concerning the into_the_void conversion

## Imaging folder structure

See the example folder structure [here](https://gin.g-node.org/CatalystNeuro/ophys_testing_data/src/main/imaging_datasets/BrukerTif) for the Bruker TIF format.

## Segmentation folder structure

See the example folder structure [here](https://gin.g-node.org/CatalystNeuro/ophys_testing_data/src/main/segmentation_datasets/suite2p) for the Suite2p format.

## Run conversion for a single session

`into_the_void_convert_sesion.py`: this script defines the function to convert one full session of the conversion.
Parameters:
- "`two_photon_imaging_folder_path`" : The folder path that contains the Bruker TIF imaging output (.ome.tif files).
- "`segmentation_folder_path`": The folder path that contains the Suite2p segmentation output.
- "`segmentation_to_imaging_plane_map`": A dictionary that maps each segmentation plane name to the imaging naming convention. Optional parameter.

### Imaging and Segmentation modalities naming convention

The `segmentation_to_imaging_plane_map` is a dictionary that maps each segmentation plane name to the imaging naming convention.
This is necessary when there are multiple channels or planes in the imaging data and the segmentation data, as the name
provided in the segmentation interface might not be the same as the name provided in the imaging interface.

#### Single plane, dual channel example
For example if the imaging data has a single plane with two channels, the default `segmentation_to_imaging_plane_map` will be defined as follows:

```python
from pinto_lab_to_nwb.into_the_void.into_the_voidnwbconverter import get_default_segmentation_to_imaging_name_mapping

# The folder path that contains the Bruker TIF imaging output (.ome.tif files).
imaging_folder_path = "NCCR62_2023_07_06_IntoTheVoid_t_series_Dual_color-000"
# The folder that contains the Suite2P segmentation output.
segmentation_folder_path = "NCCR62_2023_07_06_IntoTheVoid_t_series_Dual_color-000/suite2p"

# Provide a mapping between the imaging and segmentation planes
# The default mapping is to rely on the order of the planes in the imaging and segmentation folders
plane_map = get_default_segmentation_to_imaging_name_mapping(imaging_folder_path, segmentation_folder_path)
```
which will output:
```
{'Chan1Plane0': 'Ch1', 'Chan2Plane0': 'Ch2'}
```
where the keys are the segmentation plane names and the values are the imaging plane names.

This way the converter will automatically set the metadata for the segmentation interfaces to use the same naming convention as the imaging interfaces.
The default mapping can be adjusted by the user as follows:

```python
imaging_to_segmentation_plane_map = {'Chan1Plane0': 'Ch2', 'Chan2Plane0': 'Ch1'}
```
where the segmentation interface metadata for "Chan1Plane0" (the first channel from 'plane0' from the segmentation output from Suite2p) will be mapped
to use the same naming convention as the imaging interface metadata for "Ch2".

#### Dual plane, single channel example

For example if the imaging data has two planes with a single channel, the default `segmentation_to_imaging_plane_map` will be defined as follows:

```python
{'Chan1Plane0': 'Ch2_000001', 'Chan1Plane1': 'Ch2_000002'}
```
where the segmentation interface metadata for "Chan1Plane0" will be mapped to use the same naming convention as the imaging interface metadata 
for "Ch2_000001" (the first plane for the channel named 'Ch2' in the Bruker XML file).

If the default mapping has to be adjusted, the user can provide a custom mapping as follows:

```python
imaging_to_segmentation_plane_map = {'Chan2Plane0': 'Ch1', 'Chan1Plane0': 'Ch2'}
```
where the segmentation interface metadata for "Chan2Plane0" (the second channel from 'plane0' from the segmentation output from Suite2p) 
will be mapped to use the same naming convention as the imaging interface metadata  for "Ch1" (the channel named 'Ch1' in the Bruker XML file).

### Example usage

To run a specific conversion, you might need to install first some conversion specific dependencies that are located in each conversion directory:
```
cd src/pinto_lab_to_nwb/into_the_void
pip install -r into_the_void_requirements.txt
```
Then you can run a specific conversion with the following command:
```
python into_the_void_convert_session.py
```
