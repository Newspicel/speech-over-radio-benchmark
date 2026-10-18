#!/bin/sh
# Model weights that no engine downloads by itself. Everything else under models/
# is fetched on first use (HF_HOME, TORCH_HOME and friends all point in there).
set -eu
cd "$(dirname "$0")/.."

if [ ! -f models/dnsmos/sig_bak_ovr.onnx ]; then
  mkdir -p models/dnsmos
  curl -fL -o models/dnsmos/sig_bak_ovr.onnx \
    https://github.com/microsoft/DNS-Challenge/raw/v4dnschallenge_ICASSP2022/DNSMOS/DNSMOS/sig_bak_ovr.onnx
  echo "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd  models/dnsmos/sig_bak_ovr.onnx" \
    | shasum -a 256 -c
fi

# The sdrmm neural denoisers. SDR--'s own downloads.sdrmm.com/denoise/v1 is not
# reachable, and SDR# ships shark.dll without them, so they cannot be re-fetched
# here: copy them from a machine that has them, or rebuild them from
# github.com/Newspicel/sdrmm. Expected sha256, as pinned by that project:
#   521bd03e3b3e49bb9fd5937ec1222362b31779352d74967c44018fe90f86334e  baseline
#   25567253213f524449f39636cb3c9ea1a8869b6c584c08d80ff2111ae6541bb9  dpdfnet2
#   95cea940d1449ca13441fafe2c3dd23832692bff9f0d13755db38a982d6ef470  dpdfnet4
#   b19c7edfd3c1117aaf7d28b70ec508dd2bfb573d04406f3674bfc7e018a2fdbc  dpdfnet8
#   a7863b2a439386e0aa888745b0da67c4571325346b810eb6251ae22d792b6e00  dpdfnet2_8khz
#   9a9acae4e956ecf11e08b3aa961b1adede0ab7a3abe7c2343d27e024472d96f2  dpdfnet8_8khz
#   1463d6cdc32a85ac085c99c93d088fe6f467c5ac82b467be218034a9e1341768  dpdfnet2_48khz_hr
#   4580d5e8b6a4aa09b6417b6815e97b6cf371c410dc02dee5d2934fc78e1a2968  dpdfnet8_48khz_hr
missing=
for model in baseline dpdfnet2 dpdfnet4 dpdfnet8 \
             dpdfnet2_8khz dpdfnet8_8khz dpdfnet2_48khz_hr dpdfnet8_48khz_hr; do
  [ -f "models/sdrmm/$model.sdrmmnn" ] || missing="$missing $model"
done
[ -z "$missing" ] || echo "missing models/sdrmm:$missing (sdrmm-* and sdr#-* methods need these)"

echo "models ready."
