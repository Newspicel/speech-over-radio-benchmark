"""Every method in the benchmark: single engines and combined chains."""

from __future__ import annotations

from dataclasses import dataclass

from . import dsp
from .engines import Stage

DSP = {
    "bandpass": lambda x, sr: dsp.bandpass(x, sr),
    "gate": lambda x, sr: dsp.gate(x, sr),
    "subtraction": lambda x, sr: dsp.spectral(x, sr, "subtraction"),
    "wiener": lambda x, sr: dsp.spectral(x, sr, "wiener"),
    "mmse-stsa": lambda x, sr: dsp.spectral(x, sr, "mmse-stsa"),
    "mmse-lsa": lambda x, sr: dsp.spectral(x, sr, "mmse-lsa"),
    "noisereduce-stationary": lambda x, sr: dsp.noisereduce(x, sr, True),
    "noisereduce-nonstationary": lambda x, sr: dsp.noisereduce(x, sr, False),
    "lms-notch": lambda x, sr: dsp.lms_notch(x, sr),
}


@dataclass(frozen=True)
class Method:
    name: str
    stages: tuple[Stage, ...]
    family: str
    description: str

    @property
    def is_chain(self) -> bool:
        return len(self.stages) > 1


def m(name: str, family: str, description: str, *stages: tuple[str, str]) -> Method:
    return Method(name, tuple(Stage(b, s) for b, s in stages), family, description)


SDRMM_MODELS = {
    "baseline": "DPDFNet baseline (16 kHz)",
    "dpdfnet2": "DPDFNet-2 (16 kHz), the SDR-- default",
    "dpdfnet4": "DPDFNet-4 (16 kHz)",
    "dpdfnet8": "DPDFNet-8 (16 kHz)",
    "dpdfnet2_8khz": "DPDFNet-2 narrowband (8 kHz)",
    "dpdfnet8_8khz": "DPDFNet-8 narrowband (8 kHz)",
    "dpdfnet2_48khz_hr": "DPDFNet-2 fullband (48 kHz HR)",
    "dpdfnet8_48khz_hr": "DPDFNet-8 fullband (48 kHz HR)",
}

METHODS: list[Method] = [
    m("input", "reference", "Received audio, untouched"),
    # ---- classic DSP (this repo, sorb/dsp.py)
    m("bandpass", "dsp", "300-3000 Hz 6th-order Butterworth band-pass", ("dsp", "bandpass")),
    m("gate", "dsp", "Downward expander / noise gate, 20 dB depth, 150 ms hold", ("dsp", "gate")),
    m("subtraction", "dsp", "Berouti power spectral subtraction, quantile noise tracker", ("dsp", "subtraction")),
    m("wiener", "dsp", "Wiener gain, decision-directed a-priori SNR", ("dsp", "wiener")),
    m("mmse-stsa", "dsp", "Ephraim-Malah MMSE short-time spectral amplitude", ("dsp", "mmse-stsa")),
    m("mmse-lsa", "dsp", "Ephraim-Malah MMSE log-spectral amplitude", ("dsp", "mmse-lsa")),
    m("noisereduce-stat", "dsp", "noisereduce spectral gating, stationary", ("dsp", "noisereduce-stationary")),
    m("lms-notch", "dsp", "Adaptive line enhancer as auto-notch: 8 kHz, 2 ms decorrelation delay, slow NLMS", ("dsp", "lms-notch")),
    m("noisereduce-nonstat", "dsp", "noisereduce spectral gating, non-stationary", ("dsp", "noisereduce-nonstationary")),
    # ---- SDR# NINR (AIRSPY, closed source): its own shark.dll under Wine, SDR# profiles
    m("ninr-lp-comm-narrow", "dsp", "SDR# NINR (current engine), 'Comm (Narrow)' profile: VAD, 2048 FFT at 7.2 kHz, 10 dB depth", ("ninr", "lp:comm-narrow")),
    m("ninr-lp-comm-narrow-30dB", "dsp", "SDR# NINR, 'Comm (Narrow)' profile at 30 dB depth", ("ninr", "lp:comm-narrow:30")),
    m("ninr-lp-comm-wide", "dsp", "SDR# NINR, 'Comm (Wide)' profile: VAD, 4096 FFT at 12 kHz, 10 dB depth", ("ninr", "lp:comm-wide")),
    m("ninr-lp-hifi-soft", "dsp", "SDR# NINR, 'Hi-Fi Soft' (default) profile, 10 dB depth", ("ninr", "lp:hifi-soft")),
    m("ninr-lp-lofi-hard", "dsp", "SDR# NINR, 'Lo-Fi Hard' profile, 10 dB depth", ("ninr", "lp:lofi-hard")),
    m("ninr-hq", "dsp", "SDR# classic NINR engine (threshold −30 dB, slope 50, 30 dB depth) at 7.2 kHz", ("ninr", "hq")),
    # ---- SDR-- (real code via tools/sdrmm-denoise)
    m("sdrmm-spectral", "sdrmm", "SDR-- Spectral: MMSE-LSA + speech presence + minimum statistics, 20 dB floor", ("sdrmm", "spectral:1.0")),
    m("sdrmm-spectral-0.5", "sdrmm", "SDR-- Spectral at strength 0.5 (10 dB floor)", ("sdrmm", "spectral:0.5")),
    m("sdrmm-rnnoise", "sdrmm", "SDR-- RNNoise (nnnoiseless), 30 dB max attenuation", ("sdrmm", "rnnoise:1.0")),
    m("sdrmm-rnnoise-0.5", "sdrmm", "SDR-- RNNoise at strength 0.5 (15 dB floor)", ("sdrmm", "rnnoise:0.5")),
    m("sdrmm-autonotch", "sdrmm", "SDR-- LMS auto-notch (64 taps)", ("sdrmm", "autonotch")),
    *[
        m(f"sdrmm-{key}", "sdrmm", f"SDR-- neural: {text}, 30 dB max attenuation", ("sdrmm", f"neural:{key}:1.0"))
        for key, text in SDRMM_MODELS.items()
    ],
    m("sdrmm-dpdfnet2-0.5", "sdrmm", "SDR-- DPDFNet-2 at strength 0.5 (15 dB floor)", ("sdrmm", "neural:dpdfnet2:0.5")),
    m("sdrmm-dpdfnet2_8khz-0.5", "sdrmm", "SDR-- DPDFNet-2 8 kHz at strength 0.5 (15 dB floor)", ("sdrmm", "neural:dpdfnet2_8khz:0.5")),
    # ---- official DPDFNet (ceva-ip, ONNX) - same weights as the SDR-- port, upstream inference
    *[
        m(f"dpdfnet-{key}", "nn", f"Official DPDFNet: {text.split(',')[0]}", ("dpdfnet", key))
        for key, text in SDRMM_MODELS.items()
    ],
    m("dpdfnet-dpdfnet2-15dB", "nn", "Official DPDFNet-2, attenuation limit 15 dB", ("dpdfnet", "dpdfnet2:15")),
    m("dpdfnet-dpdfnet2_8khz-12dB", "nn", "Official DPDFNet-2 8 kHz, attenuation limit 12 dB", ("dpdfnet", "dpdfnet2_8khz:12")),
    # ---- other speech-enhancement networks
    m("deepfilternet3", "nn", "DeepFilterNet3 (Schröter et al. 2023), 48 kHz", ("deepfilternet", "DeepFilterNet3")),
    m("deepfilternet3-12dB", "nn", "DeepFilterNet3, attenuation limit 12 dB", ("deepfilternet", "DeepFilterNet3:12")),
    m("deepfilternet2", "nn", "DeepFilterNet2, 48 kHz", ("deepfilternet", "DeepFilterNet2")),
    m("rnnoise-xiph", "nn", "Official xiph RNNoise C library (pyrnnoise)", ("rnnoise", "default")),
    m("gtcrn", "nn", "GTCRN ultralight (DNS3, 48k params), ONNX", ("onnx", "gtcrn")),
    m("dtln", "nn", "DTLN dual-signal LSTM (DNS 2020), ONNX", ("onnx", "dtln")),
    m("nsnet2", "nn", "NSNet2, Microsoft DNS-Challenge baseline, ONNX", ("onnx", "nsnet2")),
    m("demucs-dns48", "nn", "Meta Demucs denoiser dns48 (CC BY-NC)", ("torch", "demucs:dns48")),
    m("demucs-dns64", "nn", "Meta Demucs denoiser dns64 (CC BY-NC)", ("torch", "demucs:dns64")),
    m("demucs-master64", "nn", "Meta Demucs denoiser master64 (CC BY-NC)", ("torch", "demucs:master64")),
    m("metricgan+", "nn", "SpeechBrain MetricGAN+ (VoiceBank-DEMAND)", ("torch", "metricgan")),
    m("sepformer-wham", "nn", "SpeechBrain SepFormer enhancement, WHAM!, 8 kHz", ("torch", "sepformer:wham")),
    m("sepformer-dns4", "nn", "SpeechBrain SepFormer enhancement, DNS4, 16 kHz", ("torch", "sepformer:dns4")),
    m("mossformer2-48k", "nn", "ClearerVoice MossFormer2_SE_48K (runs on the Apple GPU via MPS)", ("clearvoice", "MossFormer2_SE_48K")),
    m("frcrn", "nn", "ClearerVoice FRCRN_SE_16K (Apple GPU via MPS)", ("clearvoice", "FRCRN_SE_16K")),
    # MP-SENet (audiosronnx ONNX) works but ran at RTF ~2.5 on this CPU: re-enable for a long run.
    # m("mp-senet", "nn", "MP-SENet (DNS), ONNX via audiosronnx", ("audiosronnx", "denoise:mpsenet")),
    m("cmgan", "nn", "CMGAN conformer metric-GAN, ONNX via audiosronnx", ("audiosronnx", "denoise:cmgan")),
    # ---- generative restoration
    m("voicefixer", "generative", "VoiceFixer restoration (mode 0), 44.1 kHz", ("torch", "voicefixer:0")),
    m("voicefixer-mode2", "generative", "VoiceFixer mode 2 (severe degradation)", ("torch", "voicefixer:2")),
    # ---- chains of DSP + sdrmm
    m("autonotch→sdrmm-spectral", "chain", "LMS notch, then SDR-- Spectral", ("sdrmm", "autonotch"), ("sdrmm", "spectral:1.0")),
    m("sdrmm-spectral→sdrmm-rnnoise", "chain", "SDR-- Spectral, then RNNoise", ("sdrmm", "spectral:1.0"), ("sdrmm", "rnnoise:1.0")),
    m("sdrmm-rnnoise→sdrmm-spectral", "chain", "RNNoise, then SDR-- Spectral", ("sdrmm", "rnnoise:1.0"), ("sdrmm", "spectral:1.0")),
    m("sdrmm-spectral-0.5→dpdfnet2-0.5", "chain", "Half-strength Spectral, then half-strength DPDFNet-2", ("sdrmm", "spectral:0.5"), ("sdrmm", "neural:dpdfnet2:0.5")),
    m("bandpass→sdrmm-dpdfnet2_8khz", "chain", "Band-pass, then DPDFNet-2 8 kHz", ("dsp", "bandpass"), ("sdrmm", "neural:dpdfnet2_8khz:1.0")),
    # ---- chains across engines: pre-clean, post-clean, notch, restore, bandwidth extension
    m("bandpass→deepfilternet3", "chain", "Band-pass to the radio band, then DeepFilterNet3", ("dsp", "bandpass"), ("deepfilternet", "DeepFilterNet3")),
    m("sdrmm-spectral-0.5→deepfilternet3", "chain", "Gentle SDR-- Spectral pre-clean, then DeepFilterNet3", ("sdrmm", "spectral:0.5"), ("deepfilternet", "DeepFilterNet3")),
    m("deepfilternet3→sdrmm-spectral-0.5", "chain", "DeepFilterNet3, then gentle SDR-- Spectral on the residual", ("deepfilternet", "DeepFilterNet3"), ("sdrmm", "spectral:0.5")),
    m("autonotch→deepfilternet3", "chain", "LMS auto-notch (heterodynes), then DeepFilterNet3", ("sdrmm", "autonotch"), ("deepfilternet", "DeepFilterNet3")),
    m("deepfilternet3-12dB→voicefixer", "chain", "DeepFilterNet3 (12 dB limit), then VoiceFixer restoration", ("deepfilternet", "DeepFilterNet3:12"), ("torch", "voicefixer:0")),
    m("dpdfnet2_8khz→lavasr", "chain", "Official DPDFNet-2 8 kHz, then LavaSR bandwidth extension", ("dpdfnet", "dpdfnet2_8khz"), ("audiosronnx", "bwe:lavasr")),
    m("sepformer-wham→sdrmm-spectral-0.5", "chain", "SepFormer WHAM! 8 kHz, then gentle SDR-- Spectral", ("torch", "sepformer:wham"), ("sdrmm", "spectral:0.5")),
    m("mmse-lsa→dpdfnet-dpdfnet4", "chain", "Classic MMSE-LSA, then official DPDFNet-4", ("dsp", "mmse-lsa"), ("dpdfnet", "dpdfnet4")),
    m("lms-notch→sdrmm-dpdfnet8_8khz", "chain", "2 ms-delay LMS notch, then SDR-- DPDFNet-8 8 kHz", ("dsp", "lms-notch"), ("sdrmm", "neural:dpdfnet8_8khz:1.0")),
    m("ninr-lp-comm-narrow→sdrmm-dpdfnet8_8khz", "chain", "SDR# NINR Comm (Narrow), then SDR-- DPDFNet-8 8 kHz", ("ninr", "lp:comm-narrow"), ("sdrmm", "neural:dpdfnet8_8khz:1.0")),
    # ---- does a wideband model stop deleting radio speech once the band is extended first?
    m("lavasr→dpdfnet-dpdfnet2", "chain", "LavaSR bandwidth extension, then official DPDFNet-2 (16 kHz)", ("audiosronnx", "bwe:lavasr"), ("dpdfnet", "dpdfnet2")),
    m("lavasr→sdrmm-dpdfnet2", "chain", "LavaSR bandwidth extension, then SDR-- DPDFNet-2", ("audiosronnx", "bwe:lavasr"), ("sdrmm", "neural:dpdfnet2:1.0")),
    m("lavasr→deepfilternet3", "chain", "LavaSR bandwidth extension, then DeepFilterNet3", ("audiosronnx", "bwe:lavasr"), ("deepfilternet", "DeepFilterNet3")),
]

METHODS_BY_NAME = {method.name: method for method in METHODS}
