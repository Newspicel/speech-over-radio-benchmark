"""Numpy+onnxruntime-only wrappers: GTCRN, DTLN, NSNet2 (16k). All take/return float32 mono @16 kHz."""
import numpy as np, onnxruntime as ort, librosa
from scipy.signal import get_window
M = "models"

# ---------- GTCRN (stream ONNX, 16 kHz, n_fft 512 hop 256, sqrt-hann) ----------
class GTCRN:
    def __init__(self, path=f"{M}/gtcrn/gtcrn_simple.onnx"):
        self.s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.win = np.sqrt(get_window("hann", 512, fftbins=True)).astype(np.float32)
    def __call__(self, x):
        X = librosa.stft(x, n_fft=512, hop_length=256, win_length=512, window=self.win, center=True)  # (257,T)
        inp = np.stack([X.real, X.imag], -1)[None].astype(np.float32)                                # (1,257,T,2)
        conv = np.zeros([2,1,16,16,33], np.float32); tra = np.zeros([2,3,1,1,16], np.float32); inter = np.zeros([2,1,33,16], np.float32)
        outs = []
        for t in range(inp.shape[2]):
            o, conv, tra, inter = self.s.run(None, {"mix": inp[:, :, t:t+1], "conv_cache": conv, "tra_cache": tra, "inter_cache": inter})
            outs.append(o)
        Y = np.concatenate(outs, 2)[0]
        return librosa.istft(Y[..., 0] + 1j*Y[..., 1], n_fft=512, hop_length=256, win_length=512, window=self.win, center=True, length=len(x)).astype(np.float32)

# ---------- DTLN (two ONNX stages, 16 kHz, block 512 shift 128) ----------
class DTLN:
    def __init__(self, d=f"{M}/dtln"):
        self.s1 = ort.InferenceSession(f"{d}/model_1.onnx"); self.s2 = ort.InferenceSession(f"{d}/model_2.onnx")
    def __call__(self, x):
        bl, bs = 512, 128
        def zeros(s): return {i.name: np.zeros([d if isinstance(d, int) else 1 for d in i.shape], np.float32) for i in s.get_inputs()}
        n1 = [i.name for i in self.s1.get_inputs()]; n2 = [i.name for i in self.s2.get_inputs()]
        i1, i2 = zeros(self.s1), zeros(self.s2)
        audio = np.concatenate([np.zeros(bl - bs, np.float32), x.astype(np.float32), np.zeros(bl, np.float32)])
        out = np.zeros_like(audio); inb = np.zeros(bl, np.float32); outb = np.zeros(bl, np.float32)
        for k in range((len(audio) - (bl - bs)) // bs):
            inb[:-bs] = inb[bs:]; inb[-bs:] = audio[k*bs:(k+1)*bs]
            F = np.fft.rfft(inb); i1[n1[0]] = np.abs(F).reshape(1,1,-1).astype(np.float32)
            mask, i1[n1[1]] = self.s1.run(None, i1)
            est = np.fft.irfft(np.abs(F) * mask.squeeze() * np.exp(1j*np.angle(F))).reshape(1,1,-1).astype(np.float32)
            i2[n2[0]] = est; blk, i2[n2[1]] = self.s2.run(None, i2)
            outb[:-bs] = outb[bs:]; outb[-bs:] = 0; outb += blk.squeeze()
            out[k*bs:(k+1)*bs] = outb[:bs]
        return out[bl-bs: bl-bs+len(x)]

# ---------- NSNet2 (DNS baseline, 16 kHz; 20 ms win, 10 ms hop, nfft 320, log-power feats) ----------
class NSNet2:
    def __init__(self, path=f"{M}/nsnet2/nsnet2-20ms-baseline.onnx", mingain_db=-80):
        self.s = ort.InferenceSession(path); self.mingain = 10**(mingain_db/20)
        self.win = np.sqrt(np.hanning(320)).astype(np.float32)
    def __call__(self, x):
        X = librosa.stft(x, n_fft=320, hop_length=160, win_length=320, window=self.win, center=True)  # (161,T)
        feat = np.log10(np.maximum(np.abs(X)**2, 1e-12)).T[None].astype(np.float32)                  # (1,T,161)
        G = np.clip(self.s.run(None, {self.s.get_inputs()[0].name: feat})[0][0].T, self.mingain, 1.0)
        return librosa.istft(X * G, n_fft=320, hop_length=160, win_length=320, window=self.win, center=True, length=len(x)).astype(np.float32)
