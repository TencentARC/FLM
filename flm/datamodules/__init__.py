from .vg_caption_datamodule import VisualGenomeCaptionDataModule
from .f30k_caption_karpathy_datamodule import F30KCaptionKarpathyDataModule
from .coco_caption_karpathy_datamodule import CocoCaptionKarpathyDataModule
from .conceptual_caption_datamodule import ConceptualCaptionDataModule
from .sbu_datamodule import SBUCaptionDataModule
from .vqav2_datamodule import VQAv2DataModule
from .nlvr2_datamodule import NLVR2DataModule
from .snli_datamodule import SNLIDataModule
from .conceptual_caption12m_datamodule import ConceptualCaption12mDataModule
# from .conceptual_caption8m_datamodule import ConceptualCaption8mDataModule
from .laion_datamodule import LaionDataModule
from .laion100m_datamodule import Laion100mDataModule
# from .wino_datamodule import WinoDataModule

_datamodules = {
    "vg": VisualGenomeCaptionDataModule,
    "f30k": F30KCaptionKarpathyDataModule,
    "coco": CocoCaptionKarpathyDataModule,
    "gcc": ConceptualCaptionDataModule,
    "sbu": SBUCaptionDataModule,
    "vqa": VQAv2DataModule,
    "nlvr2": NLVR2DataModule,
    "snli": SNLIDataModule, 
    "gcc12m": ConceptualCaption12mDataModule,
    # "gcc8m": ConceptualCaption8mDataModule,
    "laion": LaionDataModule,
    "laion100m": Laion100mDataModule,
    # "wino": WinoDataModule
}
