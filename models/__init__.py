from models.mtan import SegMtan
from models.mtn import ReconstructMTN


def build_model(model_name, args=None):
    if model_name == "mtan":
        return SegMtan()
    elif model_name == 'reconstruct_mtn':
        return ReconstructMTN(args)
    else:
        raise NotImplementedError
