"""DeepFilterNet 2/3 (Rikorose/DeepFilterNet, PyTorch). spec = DeepFilterNet3|DeepFilterNet2[:ATTEN_LIM_DB]"""

import torch
from df.enhance import enhance, init_df

import sorbio


def load(spec):
    name, _, limit = spec.partition(":")
    model, state, _ = init_df(model_base_dir=name, log_level="ERROR")
    atten = float(limit) if limit else None

    def process(x):
        with torch.no_grad():
            return enhance(model, state, torch.from_numpy(x)[None], atten_lim_db=atten).squeeze(0).numpy()

    return process, state.sr()


sorbio.run(load)
