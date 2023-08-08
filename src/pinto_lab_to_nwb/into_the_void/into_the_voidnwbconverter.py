"""Primary NWBConverter class for this dataset."""
from neuroconv import NWBConverter
from neuroconv.datainterfaces import Suite2pSegmentationInterface
from neuroconv.converters import BrukerTiffSinglePlaneConverter


class IntoTheVoidNWBConverter(NWBConverter):
    """Primary conversion class for my extracellular electrophysiology dataset."""

    data_interface_classes = dict(
        Imaging=BrukerTiffSinglePlaneConverter,
        Segmentation=Suite2pSegmentationInterface,
    )
