"""ClearerVoice-Studio speech enhancement (Alibaba). spec = MossFormer2_SE_48K | FRCRN_SE_16K | MossFormerGAN_SE_16K"""

import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

import sorbio

WORK = Path(__file__).resolve().parents[2] / "models" / "clearvoice"
WORK.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)  # ClearVoice downloads into ./checkpoints

from clearvoice import ClearVoice  # noqa: E402


def load(spec):
    cv = ClearVoice(task="speech_enhancement", model_names=[spec])
    rate = 48_000 if "48K" in spec else 16_000

    def process(x):
        with tempfile.NamedTemporaryFile(suffix=".wav", dir=WORK) as tmp:
            sf.write(tmp.name, x, rate, subtype="FLOAT")
            y = cv(input_path=tmp.name, online_write=False)
        return np.asarray(y, dtype=np.float32).reshape(-1)

    return process, rate


sorbio.run(load)
