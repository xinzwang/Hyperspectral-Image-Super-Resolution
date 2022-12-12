# Bicubic
from .Bicubic.model import Bicubic

# RGB super resolution
from .RCAN.model import RCAN
from .SwinIR.SwinIR import SwinIR

from .SSPSR.sspsr import SSPSR
from .MCNet.mcnet import MCNet
from .BiQRNN3D.model import BiFQRNNREDC3D

# HSI super resolution
from .GDRRN.gdrrn import GDRRN
from .FCNN3D.fcnn3d import FCNN3D
from .EDSR.edsr import EDSR

# Ours
from .CANet.v1 import CANetV1
from .CANet.v2 import CANetV2
from .CANet.v3 import CANetV3

from .MemCS.model import MemCS
from .MemCS.v2 import MemCSV2

from .Ours.v0 import OursV0
from .Ours.v1 import OursV1
from .Ours.v2 import OursV2
from .Ours.v3 import OursV3
from .Ours.v4 import OursV4

from .Ours.rgb1 import OursRGB1
from .Ours.rgb2 import OursRGB2
from .Ours.rgb21 import OursRGB21
from .Ours.rgb3 import OursRGB3
from .Ours.rgb4 import OursRGB4
from .Ours.rgb5 import OursRGB5

from .Ours.PerBand import PerBand
