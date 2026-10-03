//! Runs one or more sdrmm audio stages over a mono 48 kHz f32le stream.
//!
//! ```text
//! sdrmm-denoise --models DIR STAGE[,STAGE...] < in.f32 > out.f32
//! STAGE = spectral[:strength] | rnnoise[:strength] | neural:MODEL[:strength]
//!       | autonotch | clicks[:threshold]
//! ```
//!
//! Every stage is fed blocks of `BLOCK` samples, like the live audio chain, and the
//! output is trimmed by the stage latency so it lines up sample-for-sample with the input.

use std::{
    io::{Read, Write},
    path::PathBuf,
    sync::Arc,
};

use sdrmm_channels::{neural::Net, neural_denoise::NeuralDenoiser, rnnoise::RnnoiseDenoiser};
use sdrmm_dsp::{AutoNotch, ClickRemover, SpectralDenoiser};

const RATE: f64 = 48_000.0;
const BLOCK: usize = 1_024;
const CLICK_WIDTH_S: f64 = 100e-6;

enum Stage {
    Spectral(Box<SpectralDenoiser>),
    Rnnoise(Box<RnnoiseDenoiser>),
    Neural(Box<NeuralDenoiser>),
    AutoNotch(AutoNotch),
    Clicks(ClickRemover),
}

impl Stage {
    fn parse(spec: &str, models: &PathBuf) -> Result<Self, String> {
        let parts: Vec<&str> = spec.split(':').collect();
        let strength = |index: usize| -> Result<f32, String> {
            parts
                .get(index)
                .map_or(Ok(1.0), |s| s.parse().map_err(|_| format!("bad number in {spec}")))
        };
        match parts[0] {
            "spectral" => Ok(Self::Spectral(Box::new(SpectralDenoiser::new(strength(1)?)))),
            "rnnoise" => Ok(Self::Rnnoise(Box::new(RnnoiseDenoiser::new(strength(1)?)))),
            "neural" => {
                let model = parts.get(1).ok_or("neural needs a model name")?;
                let path = models.join(format!("{model}.sdrmmnn"));
                let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
                let net = Net::load(&bytes).map_err(|e| e.to_string())?;
                NeuralDenoiser::new(Arc::new(net), strength(2)?)
                    .map(|d| Self::Neural(Box::new(d)))
                    .map_err(|e| e.to_string())
            }
            "autonotch" => Ok(Self::AutoNotch(AutoNotch::new())),
            "clicks" => Ok(Self::Clicks(ClickRemover::new(
                RATE,
                CLICK_WIDTH_S,
                parts.get(1).map_or(Ok(6.0), |s| s.parse().map_err(|_| "bad threshold"))?,
            ))),
            other => Err(format!("unknown stage {other}")),
        }
    }

    fn latency(&self) -> usize {
        match self {
            Self::Spectral(d) => d.latency(),
            Self::Rnnoise(d) => d.latency(),
            Self::Neural(d) => d.latency(),
            Self::AutoNotch(_) => 0,
            Self::Clicks(d) => d.latency(),
        }
    }

    fn process(&mut self, block: &mut [f32]) -> Result<(), String> {
        match self {
            Self::Spectral(d) => d.process(block).map_err(|e| e.to_string()),
            Self::Rnnoise(d) => {
                d.process(block);
                Ok(())
            }
            Self::Neural(d) => d.process(block).map_err(|e| e.to_string()),
            Self::AutoNotch(d) => {
                d.process(block);
                Ok(())
            }
            Self::Clicks(d) => {
                d.process(block);
                Ok(())
            }
        }
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let mut models = PathBuf::from("models/sdrmm");
    let mut chain = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--models" => models = args.next().ok_or("--models needs a directory")?.into(),
            _ => chain = Some(arg),
        }
    }
    let chain = chain.ok_or("usage: sdrmm-denoise [--models DIR] STAGE[,STAGE...]")?;

    let mut raw = Vec::new();
    std::io::stdin().read_to_end(&mut raw).map_err(|e| e.to_string())?;
    let mut audio: Vec<f32> = raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let len = audio.len();

    for spec in chain.split(',').filter(|s| !s.is_empty()) {
        let mut stage = Stage::parse(spec, &models)?;
        let latency = stage.latency();
        audio.resize(len + latency, 0.0);
        for block in audio.chunks_mut(BLOCK) {
            stage.process(block)?;
        }
        audio.drain(..latency);
    }

    let bytes: Vec<u8> = audio.iter().flat_map(|s| s.to_le_bytes()).collect();
    std::io::stdout().write_all(&bytes).map_err(|e| e.to_string())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("sdrmm-denoise: {error}");
        std::process::exit(1);
    }
}
