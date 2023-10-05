import os

def load_imagebind():
    if not os.path.exists('./ckpts/imagebind/imagebind_huge.pth'):
        os.system('mkdir -p ./ckpts/imagebind')
        os.system('wget -O ./ckpts/imagebind/imagebind_huge.pth https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth')

    imagebind_ckpt = './ckpts/imagebind/imagebind_huge.pth'

    return imagebind_ckpt
