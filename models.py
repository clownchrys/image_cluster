import re
import torch
from torch import nn
from torchvision.models.vgg import vgg16


class vgg16_sparse(nn.Module):
    def __init__(self, *args, **kwargs):
        super(vgg16_sparse, self).__init__()
        base = vgg16(pretrained=False, progress=False).features        
        self.conv_block = nn.ModuleList()
        for layer in base:
            layer.eval()
            self.conv_block.append(layer)
        self.descript()
        
    def forward(self, x):
        for layer in self.conv_block:
            x = layer(x)
        return x
    
    def descript(self):
        self.input_shape = (3, 224, 224)
        self.output_shape = (512, 7, 7)


def get_model(model_name, kwargs=None, state_file=None):
    """
    model_name: (str) the model names to use
        > model names are *ALL* lower cases
    kwargs: (dict) kwargs to generate architecture
        > this is hyper-parameters, not actual parameters
        > actual parameters are engaged with state_file
    state_file: (str) path to load state
        > if "default", torch uses following: model.__class__.__name__ + "_state.pt"
        > if None, it won't be loaded
    """
    assert (kwargs is None) or (type(kwargs) == dict), "kwargs must be dictionary type, or None"

    if kwargs is None:
        kwargs = "{}"
    command = "{model}(**{kwargs})".format(model=model_name, kwargs=kwargs)
    model = eval(command)
    
    if state_file == "default":
        state_file = f"{model_name}_state.pt"
    if state_file:
        model.load_state_dict(torch.load(state_file))
        
    model.eval()
    return model