import torch
import torch.nn as nn
import torchvision.models as models


class VGG19(torch.nn.Module):
    def __init__(self, which_layer_extract):
        super(VGG19, self).__init__()
        self.which_layer_extract = which_layer_extract
        self.features = models.vgg19(pretrained=True).features

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        for idx in range(self.which_layer_extract):
            x = self.features[idx](x)
        return x


