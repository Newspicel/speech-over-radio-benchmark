"""PyTorch models on CPU.

spec = demucs:dns48|dns64|master64          Meta `denoiser` (Defossez et al. 2020), 16 kHz
     | metricgan                             SpeechBrain MetricGAN+ (VoiceBank), 16 kHz
     | sepformer:wham                        SpeechBrain SepFormer WHAM! enhancement, 8 kHz
     | sepformer:dns4                        SpeechBrain SepFormer DNS4, 16 kHz
     | voicefixer:MODE                       VoiceFixer restoration (44.1 kHz), mode 0/1/2
"""

from pathlib import Path

import torch

import sorbio

CACHE = Path(__file__).resolve().parents[2] / "models" / "speechbrain"


def load(spec):
    kind, _, arg = spec.partition(":")
    if kind == "demucs":
        from denoiser import pretrained

        model = getattr(pretrained, arg)().eval()

        def process(x):
            with torch.no_grad():
                return model(torch.from_numpy(x)[None, None])[0, 0].numpy()

        return process, model.sample_rate
    if kind == "metricgan":
        from speechbrain.inference.enhancement import SpectralMaskEnhancement

        enh = SpectralMaskEnhancement.from_hparams(source="speechbrain/metricgan-plus-voicebank", savedir=str(CACHE / "metricgan"))

        def process(x):
            with torch.no_grad():
                return enh.enhance_batch(torch.from_numpy(x)[None], lengths=torch.tensor([1.0]))[0].numpy()

        return process, 16_000
    if kind == "sepformer":
        from speechbrain.inference.separation import SepformerSeparation

        source, rate = {"wham": ("speechbrain/sepformer-wham-enhancement", 8_000),
                        "dns4": ("speechbrain/sepformer-dns4-16k-enhancement", 16_000)}[arg]
        sep = SepformerSeparation.from_hparams(source=source, savedir=str(CACHE / f"sepformer-{arg}"))

        def process(x):
            with torch.no_grad():
                return sep.separate_batch(torch.from_numpy(x)[None])[0, :, 0].numpy()

        # Transformer over the whole clip: 10 s chunks keep memory bounded on 60 s inputs.
        return sorbio.chunked(process, rate), rate
    if kind == "voicefixer":
        from voicefixer import VoiceFixer

        import numpy as np

        vf = VoiceFixer()
        mode = int(arg or 0)

        def process(x):
            # VoiceFixer splits input into 30 s segments and fails on a remainder of a few
            # samples; padding to whole seconds keeps every segment at least 1 s long.
            n = len(x)
            padded = np.pad(x, (0, (-n) % 44_100))
            return np.asarray(vf.restore_inmem(padded, cuda=False, mode=mode)).reshape(-1)[:n]

        return process, 44_100
    raise ValueError(spec)


sorbio.run(load)
