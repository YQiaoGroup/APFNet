# ---- voxelmorph ----
# unsupervised learning for image registration

from . import generators
from . import py
from .py.utils import default_unet_features


# import backend-dependent submodules
backend = py.utils.get_backend()


try:
    import torch
except ImportError:
    raise ImportError('Please install pytorch to use this voxelmorph backend')

from . import torch
from .torch import layers
from .torch import networks
from .torch import losses

# else:
#     # tensorflow is default backend
#     try:
#         import tensorflow
#     except ImportError:
#         raise ImportError('Please install tensorflow to use this voxelmorph backend')

#     from . import tf
#     from .tf import layers
#     from .tf import networks
#     from .tf import losses
#     from .tf import utils
