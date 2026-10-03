"""Small ONNX denoisers: GTCRN (DNS3), DTLN, NSNet2 (DNS baseline). spec = gtcrn|dtln|nsnet2"""

from pathlib import Path

import requests

import models as M
import sorbio

ROOT = Path(__file__).resolve().parents[2] / "models" / "onnx"
FILES = {
    "gtcrn/gtcrn_simple.onnx": "https://github.com/Xiaobin-Rong/gtcrn/raw/main/stream/onnx_models/gtcrn_simple.onnx",
    "dtln/model_1.onnx": "https://github.com/breizhn/DTLN/raw/master/pretrained_model/model_1.onnx",
    "dtln/model_2.onnx": "https://github.com/breizhn/DTLN/raw/master/pretrained_model/model_2.onnx",
    "nsnet2/nsnet2-20ms-baseline.onnx": "https://github.com/microsoft/DNS-Challenge/raw/v4dnschallenge_ICASSP2022/NSNet2-baseline/nsnet2-20ms-baseline.onnx",
}


def fetch():
    for rel, url in FILES.items():
        path = ROOT / rel
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(requests.get(url, timeout=120).content)


def load(spec):
    fetch()
    model = {
        "gtcrn": lambda: M.GTCRN(str(ROOT / "gtcrn/gtcrn_simple.onnx")),
        "dtln": lambda: M.DTLN(str(ROOT / "dtln")),
        "nsnet2": lambda: M.NSNet2(str(ROOT / "nsnet2/nsnet2-20ms-baseline.onnx")),
    }[spec]()
    return model, 16_000


sorbio.run(load)
