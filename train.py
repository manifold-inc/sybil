import os

import sybil as sb


# check if the imagebind model is available at ./ckpts/imagebind/imagebind_huge.pth if not, download it from https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth

if not os.path.exists('./ckpts/imagebind/imagebind_huge.pth'):
    os.system('mkdir -p ./ckpts/imagebind')
    os.system('wget -O ./ckpts/imagebind/imagebind_huge.pth https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth')

imagebind_ckpt = './ckpts/imagebind/imagebind_huge.pth'


config = {
    'imagebind_ckpt': imagebind_ckpt,
}

sybil = sb.sybil(config)
