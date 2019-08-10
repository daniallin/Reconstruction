from models.mtan import SegMtan


def build_model(model_name):
    if model_name == 'mtan':
        return SegMtan()
    else:
        raise NotImplementedError


