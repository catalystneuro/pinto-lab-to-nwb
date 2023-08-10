"""Primary NWBConverter class for this dataset."""
from neuroconv import NWBConverter
from neuroconv.datainterfaces import MicroManagerTiffImagingInterface


class WideFieldNWBConverter(NWBConverter):
    """Primary conversion class for Widefield imaging dataset."""

    data_interface_classes = dict(
        Imaging=MicroManagerTiffImagingInterface,
    )
