import torch
from .demucs import Demucs


def load_denoiser(ckpt_path="weights/demucs_64.ckpt"):
    state = torch.load(ckpt_path, map_location="cpu")
    model = Demucs(**state["args"])
    model.load_state_dict(state["weights"])
    return model.eval()
