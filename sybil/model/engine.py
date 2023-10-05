import torch
import torch.nn as nn

from imagebind.models import imagebind_model

class Engine(nn.Module):
    def __init__(self, config):
        super(Engine, self).__init__()
        self.config = config

        
        # initalize the visual encoder ( imagebind )
        self.visual_hidden_size = 1024
        self.visual_encoder = imagebind_model.imagebind_huge(pretrained=True)

        # freeze the visual encoder
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder.eval()
        
        print('Visual encoder initialized.')

    def forward(self, batch):
        return batch