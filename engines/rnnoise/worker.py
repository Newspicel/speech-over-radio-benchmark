"""Official xiph RNNoise C library (via pyrnnoise ctypes), 48 kHz, 480-sample frames. spec = default"""

import numpy as np
from pyrnnoise.rnnoise import FRAME_SIZE, create, destroy, process_mono_frame

import sorbio


def load(spec):
    def process(x):
        n = len(x)
        pcm = np.clip(np.pad(x, (0, (-n) % FRAME_SIZE)) * 32767, -32768, 32767).astype(np.int16)
        state = create()
        out = np.empty_like(pcm)
        for i in range(0, len(pcm), FRAME_SIZE):
            frame, _ = process_mono_frame(state, pcm[i : i + FRAME_SIZE])
            out[i : i + FRAME_SIZE] = frame
        destroy(state)
        return out[:n].astype(np.float32) / 32767

    return process, 48_000


sorbio.run(load)
