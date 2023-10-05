import os

import sybil as sb


# check if the imagebind model is available at ./ckpts/imagebind/imagebind_huge.pth if not, download it from https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth

# imagebind_ckpt = sb.load_imagebind()

# config = {
#     'imagebind_ckpt': imagebind_ckpt,
# }

config = {}
sybil = sb.engine(config)
