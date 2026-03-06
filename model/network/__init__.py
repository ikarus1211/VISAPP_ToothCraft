from model.network.resnet3d import DiffUNet
from model.network.condition import ContextCondNet

def initialize_diff_net(config):
    diff_nets = ['DiffUNet']

    if config.net.network == 'ControlledUNet':
        model = DiffUNet(config)
    else:
        raise ValueError(f'Unknown network: {config.net.network}. Supported networks are {diff_nets}')
    return model

def initialize_control_net(config):
    cont_nets = ['ContextNet']

    if config.net.controlnet == 'ContextNet':
        model = ContextCondNet(config)
    else:
        raise ValueError(f'Unknown network: {config.net.network}. Supported networks are {cont_nets}')
    return model