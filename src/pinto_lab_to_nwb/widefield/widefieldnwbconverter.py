"""Primary NWBConverter class for this dataset."""
from neuroconv import NWBConverter

from pinto_lab_to_nwb.widefield.interfaces import (
    WidefieldImagingInterface,
    WidefieldProcessedImagingInterface,
    WidefieldProcessedSegmentationinterface,
    WidefieldSegmentationImagesBlueInterface,
    WidefieldSegmentationImagesVioletInterface
)


class WideFieldNWBConverter(NWBConverter):
    """Primary conversion class for Widefield imaging dataset."""

    data_interface_classes = dict(
        ImagingBlue=WidefieldImagingInterface,
        ImagingViolet=WidefieldImagingInterface,
        ProcessedImagingBlue=WidefieldProcessedImagingInterface,
        ProcessedImagingViolet=WidefieldProcessedImagingInterface,
        SegmentationProcessedBlue=WidefieldProcessedSegmentationinterface,
        SummaryImagesBlue=WidefieldSegmentationImagesBlueInterface,
        SummaryImagesViolet=WidefieldSegmentationImagesVioletInterface,
    )
