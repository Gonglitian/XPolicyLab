try:
    from .deploy import *
except ImportError:
    pass

def __getattr__(name):
    if name == 'Model':
        from .model import Model
        return Model
    raise AttributeError(name)


def get_model(deploy_cfg):
    from .model import Model
    return Model(deploy_cfg)
