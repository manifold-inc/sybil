import torch
import torch.nn as nn

from imagebind.models import imagebind_model

class Engine(nn.Module):
    def __init__(self, config):
        super(Engine, self).__init__()
        self.config = config

        imagebind_ckpt = config['imagebind_ckpt'] if 'imagebind_ckpt' in config else None

        assert imagebind_ckpt is not None, 'imagebind_ckpt is not provided'

        # initalize the visual encoder ( imagebind )
        self.visual_encoder, self.visual_hidden_size = imagebind_model.imagebind_huge(pretrained=True, ckpt=imagebind_ckpt)
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder.eval()
        print('Visual encoder initialized.')

    def forward(self, batch):
        return batch