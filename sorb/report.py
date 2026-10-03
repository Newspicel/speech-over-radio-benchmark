"""Regenerates README.md (and results/plots/*.png) from results/*.csv.

Every number and every "finding" in the README is computed here from the CSVs, so the
README can never drift from the data. Run `sorb report` after any benchmark run.
"""

from __future__ import annotations

import datetime as dt
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"
README = ROOT / "README.md"

# Reference palette (dataviz skill): three categorical slots + neutral, one sequential ramp.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
KIND_COLOR = {"DSP": "#2a78d6", "Neural": "#eb6834", "Chain": "#1baf7a", "Input": "#8a8984"}
KIND_MARKER = {"DSP": "o", "Neural": "s", "Chain": "D", "Input": "X"}
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

DELETED_DB = -20.0  # speech change below this = speech deleted
MUTE_DBFS = -60.0  # output level below this = muted
GROUP_TITLES = {
    "nfm": "NFM (repeaters, simplex, CB, scanner)",
    "am": "AM (airband, CB, shortwave/LW broadcast, HF AM)",
    "ssb": "SSB (HF amateur, aero/marine VOLMET & HF, CB, satellite)",
    "iq": "IQ recordings demodulated by this benchmark",
    "hf-real": "Real HF SSB over KiwiSDRs, receiver tuned (Zenodo 5927810)",
    "hf-real-mistuned": "Real HF SSB over KiwiSDRs, receiver 300 Hz off",
    "atc": "Real VHF airband ATC with transcripts (ATCO2, UWB-ATCC)",
    "digital": "Digital voice (decoded DMR / FreeDV) - vocoder audio, for reference",
}


# --------------------------------------------------------------------------- data


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clips = pd.read_csv(RESULTS / "clips.csv", keep_default_na=False, na_values=[""])
    corpus = pd.read_csv(RESULTS / "corpus.csv", keep_default_na=False)
    methods = pd.read_csv(RESULTS / "methods.csv", keep_default_na=False, na_values=[""])
    for col in ("chain", "description", "family"):
        methods[col] = methods[col].fillna("")
    frame = clips.merge(corpus, on="clip", how="inner")
    order = {name: i for i, name in enumerate(methods.method)}
    present = [m for m in methods.method if m in set(frame.method)]
    methods = methods[methods.method.isin(present)].reset_index(drop=True)
    frame["order"] = frame.method.map(order)
    return frame, corpus, methods


def kind(method: str, family: str) -> str:
    if family == "reference":
        return "Input"
    if family == "chain":
        return "Chain"
    if family == "dsp" or any(k in method for k in ("spectral", "autonotch")):
        return "DSP"
    return "Neural"


def wer(group: pd.DataFrame) -> float:
    """Corpus WER with each clip's errors capped at its word count, so a hallucinated
    paragraph on one hopeless clip cannot outweigh everything else."""
    words = group.get("asr_words", pd.Series(dtype=float)).fillna(0)
    if not words.sum():
        return np.nan
    errors = np.minimum(group["asr_errors"].fillna(0), words)
    return float(errors.sum() / words.sum() * 100)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, g in frame.groupby("method", sort=False):
        row = {"method": method, "n": len(g)}
        for col in ("pesq_nb", "pesq_wb", "estoi", "si_sdr", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovr"):
            if col in g and g[col].notna().any():
                row[col] = g[col].mean()
        for col in ("noise_db", "speech_db", "snr_gain_db", "output_dbfs"):
            row[col + "_med"] = g[col].median()
            row[col + "_mean"] = g[col].mean()
        row["deleted"] = float(((g.speech_db < DELETED_DB) | (g.output_dbfs < MUTE_DBFS)).mean())
        if "asr_words" in g and g.asr_words.fillna(0).sum() > 0:
            row["wer"] = wer(g)
        rows.append(row)
    return pd.DataFrame(rows).set_index("method")


# --------------------------------------------------------------------------- formatting


def f(value, digits=2, signed=False, suffix=""):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    text = f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    return text.replace("-", "−") + suffix


def table(header: list[str], rows: list[list[str]], align: str | None = None) -> str:
    align = align or "l" + "r" * (len(header) - 1)
    marks = {"l": ":--", "r": "--:", "c": ":-:"}
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(marks[a] for a in align) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def best_marks(values: pd.Series, higher: bool, exclude: set[str]) -> set[str]:
    candidates = values.drop([m for m in exclude if m in values.index]).dropna()
    if candidates.empty:
        return set()
    target = candidates.max() if higher else candidates.min()
    return set(candidates[np.isclose(candidates, target)].index)


def bold_if(text: str, cond: bool) -> str:
    return f"**{text}**" if cond and text != "—" else text


def link(method: str) -> str:
    return f"`{method}`"


# --------------------------------------------------------------------------- plots


def _style(ax):
    import matplotlib.pyplot as plt  # noqa: F401

    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_tradeoff(summary: pd.DataFrame, methods: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fam = dict(zip(methods.method, methods.family))
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=SURFACE)
    _style(ax)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.axhspan(DELETED_DB, -200, color="#f0efec", zorder=0)
    ax.text(0.99, 0.02, "speech deleted", transform=ax.transAxes, ha="right", va="bottom", color=INK_2, fontsize=8)
    points = []
    for method, row in summary.iterrows():
        k = kind(method, fam.get(method, ""))
        x, y = row.noise_db_med, row.speech_db_med
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        ax.scatter(x, y, s=46, color=KIND_COLOR[k], marker=KIND_MARKER[k], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        points.append((x, y, method))
    # Direct-label only the frontier (best speech kept for a given suppression) and the input.
    frontier, best_y = [], -np.inf
    for x, y, method in sorted(points, key=lambda p: p[0]):
        if y > best_y + 0.3 or method == "input":
            frontier.append((x, y, method))
            best_y = max(best_y, y)
    for x, y, method in frontier[:14]:
        ax.annotate(method, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=7, color=INK)
    lo = min([p[1] for p in points] + [-5])
    ax.set_ylim(max(lo - 3, -90), 3)
    ax.set_xlabel("noise change in speech pauses, dB (median) — left is more suppression", color=INK_2, fontsize=9)
    ax.set_ylabel("speech change, dB (median) — 0 is speech kept", color=INK_2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    handles = [plt.Line2D([], [], marker=KIND_MARKER[k], linestyle="", color=c, markersize=7, label=k)
               for k, c in KIND_COLOR.items()]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              labelcolor=INK_2)
    ax.set_xlim(right=3)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_heatmap(matrix: pd.DataFrame, path: Path, title: str, fmt: str, vmax: float | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq", SEQUENTIAL)
    rows, cols = matrix.shape
    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * cols, 1.2 + 0.26 * rows), facecolor=SURFACE)
    values = matrix.to_numpy(dtype=float)
    im = ax.imshow(values, cmap=cmap, aspect="auto", vmin=0, vmax=vmax or np.nanmax(values))
    ax.set_xticks(range(cols), matrix.columns, rotation=60, ha="right", fontsize=7, color=INK_2)
    ax.set_yticks(range(rows), matrix.index, fontsize=7, color=INK_2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    top = vmax or np.nanmax(values)
    for i in range(rows):
        for j in range(cols):
            v = values[i, j]
            if np.isfinite(v):
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=6,
                        color="#ffffff" if v > 0.55 * top else INK)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01).ax.tick_params(labelsize=7, colors=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------- sections


def section_leaderboard(syn: pd.DataFrame, methods: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    s = summarize(syn)
    s = s.loc[[m for m in methods.method if m in s.index]]
    # Overall = mean rank over the four views of "better": PESQ-NB (quality), ESTOI
    # (intelligibility), DNSMOS OVRL (reference-free quality) and Whisper WER (machine).
    ranks = []
    for col, higher in (("pesq_nb", True), ("estoi", True), ("dnsmos_ovr", True), ("wer", False)):
        if col in s and s[col].notna().any():
            ranks.append(s[col].rank(ascending=not higher))
    s["overall"] = pd.concat(ranks, axis=1).mean(axis=1)
    ordered = s.sort_values("overall")
    ordered = pd.concat([ordered.loc[["input"]], ordered.drop("input")]) if "input" in ordered.index else ordered
    ex = {"input"}
    best = {
        "wer": best_marks(s.get("wer", pd.Series(dtype=float)), False, ex),
        "pesq_nb": best_marks(s.pesq_nb, True, ex),
        "pesq_wb": best_marks(s.pesq_wb, True, ex),
        "estoi": best_marks(s.estoi, True, ex),
        "si_sdr": best_marks(s.si_sdr, True, ex),
        "dnsmos_ovr": best_marks(s.dnsmos_ovr, True, ex),
    }
    rtf = dict(zip(methods.method, methods.rtf))
    rows = []
    for rank, (method, r) in enumerate(ordered.iterrows()):
        rows.append([
            "" if method == "input" else str(rank),
            link(method),
            f(r.overall, 1),
            bold_if(f(r.get("wer"), 1), method in best["wer"]),
            bold_if(f(r.pesq_nb), method in best["pesq_nb"]),
            bold_if(f(r.pesq_wb), method in best["pesq_wb"]),
            bold_if(f(r.estoi, 3), method in best["estoi"]),
            bold_if(f(r.si_sdr, 1), method in best["si_sdr"]),
            bold_if(f(r.dnsmos_ovr), method in best["dnsmos_ovr"]),
            f(r.noise_db_med, 1), f(r.speech_db_med, 1),
            f(rtf.get(method), 3) if method != "input" else "",
        ])
    header = ["#", "method", "mean rank", "WER %", "PESQ-NB", "PESQ-WB", "ESTOI", "SI-SDR dB", "DNSMOS OVRL",
              "noise Δ dB", "speech Δ dB", "RTF"]
    return table(header, rows, "rl" + "r" * 10), s


def real_group_table(real: pd.DataFrame, methods: pd.DataFrame, group: str) -> str:
    g = real[real.group == group]
    s = summarize(g)
    s = s.loc[[m for m in methods.method if m in s.index]]
    ex = {"input"}
    usable = s[(s.speech_db_med > DELETED_DB) & (s.output_dbfs_med > MUTE_DBFS)]
    best_gain = best_marks(usable.snr_gain_db_med, True, ex)
    best_noise = best_marks(usable.noise_db_med, False, ex)
    best_mos = best_marks(s.dnsmos_ovr, True, ex)
    has_wer = "wer" in s and s.wer.notna().any()
    has_ref = "pesq_nb" in s and s.pesq_nb.notna().any()
    best_wer = best_marks(s.wer, False, ex) if has_wer else set()
    best_pesq = best_marks(s.pesq_nb, True, ex) if has_ref else set()
    best_estoi = best_marks(s.estoi, True, ex) if has_ref else set()
    inp = s.loc["input"] if "input" in s.index else None
    rows = []
    for method, r in s.iterrows():
        muted = r.output_dbfs_med < MUTE_DBFS
        deleted = r.speech_db_med < DELETED_DB
        if method == "input":
            row = [link("input") + " (as received)", "—", "—", "—"]
        else:
            noise = f"{f(r.noise_db_med, 1)} ({f(r.noise_db_mean, 1)}) dB"
            if muted:
                noise = f"complete mute ({f(r.noise_db_mean, 1)})"
            speech = f(r.speech_db_med, 2) + " dB" + (", speech deleted" if deleted else "")
            gain = "—" if (muted or deleted) else f(r.snr_gain_db_med, 1, signed=True) + " dB"
            row = [link(method), bold_if(noise, method in best_noise), speech, bold_if(gain, method in best_gain)]
        if has_wer:
            row.append(bold_if(f(r.get("wer"), 1), method in best_wer))
        if has_ref:
            row += [bold_if(f(r.pesq_nb), method in best_pesq), bold_if(f(r.estoi, 3), method in best_estoi)]
        row += [bold_if(f(r.dnsmos_ovr), method in best_mos), "" if method == "input" else f"{r.deleted * 100:.0f} %"]
        rows.append(row)
    header = ["engine", "noise suppression, median (mean)", "speech change (median)", "SNR gain (median)"]
    if has_wer:
        header.append("WER %")
    if has_ref:
        header += ["PESQ-NB", "ESTOI"]
    header += ["DNSMOS OVRL", "clips with speech deleted"]
    n = len(g[g.method == "input"])
    return f"{n} clips, {g[g.method == 'input'].duration_s.sum() / 60:.1f} min.\n\n" + table(header, rows)


GLANCE_GROUPS = (("nfm", "NFM"), ("am", "AM"), ("ssb", "SSB"), ("hf-real", "Real HF SSB"))


def section_glance(real: pd.DataFrame, methods: pd.DataFrame) -> str:
    """Engine rows, per radio group: noise suppression median (mean), speech change, SNR gain."""
    summaries = {g: summarize(real[real.group == g]) for g, _ in GLANCE_GROUPS if (real.group == g).any()}
    best = {}
    for g, sm in summaries.items():
        usable = sm[(sm.speech_db_med > DELETED_DB) & (sm.output_dbfs_med > MUTE_DBFS)].drop(index="input", errors="ignore")
        best[g] = (best_marks(usable.noise_db_med, False, set()), best_marks(usable.speech_db_med.abs(), False, set()),
                   best_marks(usable.snr_gain_db_med, True, set()))
    header, rows = ["engine"], []
    for g, title in GLANCE_GROUPS:
        if g in summaries:
            header += [f"{title}: noise suppression, median (mean)", f"{title}: speech change", f"{title}: SNR gain"]
    for method in methods.method:
        if method == "input":
            continue
        row = [link(method)]
        for g, _ in GLANCE_GROUPS:
            if g not in summaries:
                continue
            sm = summaries[g]
            if method not in sm.index:
                row += ["", "", ""]
                continue
            r = sm.loc[method]
            muted, deleted = r.output_dbfs_med < MUTE_DBFS, r.speech_db_med < DELETED_DB
            noise = f"complete mute ({f(r.noise_db_mean, 1)})" if muted else f"{f(r.noise_db_med, 1)} ({f(r.noise_db_mean, 1)}) dB"
            speech = f"{f(r.speech_db_med, 2)} dB" + (", speech deleted" if deleted else "")
            gain = "—" if muted or deleted else f(r.snr_gain_db_med, 1, signed=True) + " dB"
            bn, bs, bg = best[g]
            row += [bold_if(noise, method in bn), bold_if(speech, method in bs), bold_if(gain, method in bg)]
        rows.append(row)
    return table(header, rows)


def section_conditions(syn: pd.DataFrame, methods: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = {}
    for (method, cond), g in syn.groupby(["method", "condition"]):
        rows.setdefault(method, {})[cond] = wer(g) if metric == "wer" else g[metric].mean()
    matrix = pd.DataFrame(rows).T
    conds = [c for c in syn.drop_duplicates("condition").condition]
    return matrix.loc[[m for m in methods.method if m in matrix.index], conds]


def section_versus(syn_s: pd.DataFrame, real_s: pd.DataFrame | None, names: list[str]) -> str:
    rows = []
    for name in names:
        if name not in syn_s.index:
            continue
        r = syn_s.loc[name]
        q = real_s.loc[name] if real_s is not None and name in real_s.index else None
        rows.append([link(name), f(r.get("wer"), 1), f(r.pesq_nb), f(r.estoi, 3), f(r.dnsmos_ovr),
                     f(q.noise_db_med, 1) if q is not None else "—", f(q.speech_db_med, 2) if q is not None else "—",
                     f(q.snr_gain_db_med, 1, True) if q is not None else "—"])
    return table(["method", "synthetic WER %", "PESQ-NB", "ESTOI", "DNSMOS OVRL", "real noise Δ dB",
                  "real speech Δ dB", "real SNR gain dB"], rows)


def section_chains(syn_s: pd.DataFrame, real_s: pd.DataFrame | None, methods: pd.DataFrame) -> str:
    stage_to_method = {}
    for r in methods.itertuples():
        if "→" not in r.chain:
            stage_to_method[r.chain] = r.method
    rows = []
    for r in methods.itertuples():
        if r.family != "chain" or r.method not in syn_s.index:
            continue
        parts = [stage_to_method.get(p.strip()) for p in r.chain.split("→")]
        parts = [p for p in parts if p and p in syn_s.index]
        if not parts:
            continue
        key = "wer" if "wer" in syn_s and np.isfinite(syn_s.loc[r.method].get("wer", np.nan)) else "pesq_nb"
        comp = syn_s.loc[parts, key]
        best_part = comp.idxmin() if key == "wer" else comp.idxmax()
        chain_v, part_v = syn_s.loc[r.method, key], comp.loc[best_part]
        better = chain_v < part_v if key == "wer" else chain_v > part_v
        real_txt = "—"
        if real_s is not None and r.method in real_s.index and best_part in real_s.index:
            real_txt = (f"{f(real_s.loc[r.method].snr_gain_db_med, 1, True)} vs "
                        f"{f(real_s.loc[best_part].snr_gain_db_med, 1, True)} dB")
        rows.append([link(r.method), link(best_part),
                     f"{f(chain_v, 1 if key == 'wer' else 2)} vs {f(part_v, 1 if key == 'wer' else 2)}",
                     f(syn_s.loc[r.method].pesq_nb) + " vs " + f(syn_s.loc[best_part].pesq_nb),
                     real_txt, "✅ chain helps" if better else "❌ no gain"])
    if not rows:
        return "_No chains were run._"
    return table(["chain", "best single stage", "WER % (chain vs stage)", "PESQ-NB", "real SNR gain", "verdict"],
                 rows, "llrrrl")


def section_demod(syn: pd.DataFrame, syn_s: pd.DataFrame) -> str:
    pairs = [("am-cnr3", "am-sync-cnr3"), ("am-cnr8", "am-sync-cnr8"), ("am-fade-cnr14", "am-fade-sync-cnr14")]
    top = [m for m in syn_s.sort_values("wer" if "wer" in syn_s else "pesq_nb",
                                         ascending="wer" in syn_s).index if m != "input"][:3]
    rows = []
    for env, sync in pairs:
        for method in ["input", *top]:
            vals = []
            for cond in (env, sync):
                g = syn[(syn.method == method) & (syn.condition == cond)]
                if g.empty:
                    vals.append(("—", "—"))
                    continue
                vals.append((f(wer(g), 1) if "asr_words" in g and g.asr_words.sum() else "—", f(g.pesq_nb.mean())))
            rows.append([env.replace("am-", "").replace("-", " "), link(method), *vals[0], *vals[1]])
    return table(["condition", "method", "envelope WER %", "envelope PESQ-NB", "sync WER %", "sync PESQ-NB"],
                 rows, "llrrrr")


def section_port(syn_s: pd.DataFrame, real_s: pd.DataFrame | None, hf_s: pd.DataFrame | None) -> str:
    """Same DPDFNet weights, two inference stacks: SDR-- (Rust, streaming, 30 dB floor) vs upstream ONNX."""
    rows = []
    for key in ("baseline", "dpdfnet2", "dpdfnet4", "dpdfnet8", "dpdfnet2_8khz", "dpdfnet8_8khz",
                "dpdfnet2_48khz_hr", "dpdfnet8_48khz_hr"):
        ours, theirs = f"sdrmm-{key}", f"dpdfnet-{key}"
        if ours not in syn_s.index or theirs not in syn_s.index:
            continue
        def cell(frame, method, col, digits=1):
            return f(frame.loc[method].get(col), digits) if frame is not None and method in frame.index else "—"
        rows.append([
            f"`{key}`",
            f"{cell(syn_s, ours, 'wer')} / {cell(syn_s, theirs, 'wer')}",
            f"{cell(syn_s, ours, 'pesq_nb', 2)} / {cell(syn_s, theirs, 'pesq_nb', 2)}",
            f"{cell(syn_s, ours, 'speech_db_med')} / {cell(syn_s, theirs, 'speech_db_med')}",
            f"{cell(real_s, ours, 'speech_db_med')} / {cell(real_s, theirs, 'speech_db_med')}",
            f"{cell(real_s, ours, 'noise_db_med')} / {cell(real_s, theirs, 'noise_db_med')}",
            f"{cell(hf_s, ours, 'wer')} / {cell(hf_s, theirs, 'wer')}",
        ])
    if not rows:
        return "_Not run yet._"
    return table(["model", "synthetic WER %", "synthetic PESQ-NB", "synthetic speech Δ dB",
                  "real speech Δ dB", "real noise Δ dB", "real-HF WER %"], rows, "lrrrrrr")


def findings(syn_s, real_s, syn, real, methods) -> list[str]:
    out = []
    ex = ["input"]
    s = syn_s.drop(index=ex, errors="ignore")
    base = syn_s.loc["input"] if "input" in syn_s.index else None
    wers = s["wer"].dropna() if "wer" in s else pd.Series(dtype=float)
    if not wers.empty and base is not None:
        out.append(f"**Most intelligible (synthetic, Whisper WER):** {link(wers.idxmin())} at {f(wers.min(), 1)} % "
                   f"vs {f(base.wer, 1)} % for the untouched input.")
        worse = wers[wers > base.wer].sort_values(ascending=False)
        if not worse.empty:
            out.append(f"**{len(worse)} of {len(wers)} methods make speech *harder* for ASR than doing nothing** "
                       "(higher WER than the input), worst: " + ", ".join(link(m) for m in worse.index[:5]) + ".")
        for mode in ("nfm", "am", "usb"):
            sub = summarize(syn[syn["mode"] == mode]).drop(index=ex, errors="ignore")
            w = sub["wer"].dropna() if "wer" in sub else pd.Series(dtype=float)
            if not w.empty:
                out.append(f"Best on **{'SSB' if mode == 'usb' else mode.upper()}** by WER: {link(w.idxmin())} ({f(w.min(), 1)} %).")
    pesq = s["pesq_nb"].dropna() if "pesq_nb" in s else pd.Series(dtype=float)
    if not pesq.empty and base is not None:
        out.append(f"**Best quality (PESQ-NB):** {link(pesq.idxmax())} at {f(pesq.max())} (input {f(base.pesq_nb)}); "
                   f"best ESTOI: {link(s.estoi.dropna().idxmax())} ({f(s.estoi.max(), 3)}).")
    wide = [m for m in ("dpdfnet-baseline", "dpdfnet-dpdfnet2", "dpdfnet-dpdfnet4", "dpdfnet-dpdfnet8",
                        "dpdfnet-dpdfnet2_48khz_hr", "dpdfnet-dpdfnet8_48khz_hr") if m in syn_s.index]
    narrow = [m for m in ("dpdfnet-dpdfnet2_8khz", "dpdfnet-dpdfnet8_8khz") if m in syn_s.index]
    if wide and narrow:
        out.append(f"**DPDFNet's wideband models delete radio speech; its 8 kHz models do not.** Upstream DPDFNet "
                   f"(16/48 kHz) changes speech by {f(syn_s.loc[wide, 'speech_db_med'].max(), 0)} to "
                   f"{f(syn_s.loc[wide, 'speech_db_med'].min(), 0)} dB on the synthetic set, the 8 kHz variants by "
                   f"{f(syn_s.loc[narrow, 'speech_db_med'].mean(), 1)} dB. It is not the SDR-- port (same behaviour "
                   "upstream), not fixed by bandwidth extension first, and only partly by input level — see below.")
    lags = []
    for meta in (ROOT / "data" / "out").glob("*/stage.json"):
        import json

        m = json.loads(meta.read_text())
        if len(m["chain"]) == 1 and m["chain"][0].startswith("sdrmm:neural") and m.get("lag_samples"):
            lags.append(m["lag_samples"] / 48)
    if lags:
        out.append(f"**SDR-- neural denoisers are {min(lags):.0f}–{max(lags):.0f} ms later than their `latency()` "
                   "reports** (measured on every clip, constant per model). The benchmark compensates it (`sorb realign`); "
                   "SDR-- itself should add the model's look-ahead and resampler delay to `latency()`.")
    if "sdrmm-autonotch" in syn_s.index and base is not None and "wer" in syn_s:
        a = syn_s.loc["sdrmm-autonotch"]
        txt = (f"**The SDR-- auto-notch removes speech, not just tones:** speech {f(a.speech_db_med, 1)} dB, WER "
               f"{f(a.wer, 1)} % (input {f(base.wer, 1)} %). Its 4-sample (83 µs) decorrelation delay makes voiced "
               "speech predictable.")
        if "lms-notch" in syn_s.index:
            b = syn_s.loc["lms-notch"]
            txt += (f" The same LMS idea with a 2 ms delay at 8 kHz (`lms-notch`) keeps speech ({f(b.speech_db_med, 1)} dB, "
                    f"WER {f(b.wer, 1)} %).")
        out.append(txt)
    env = syn[(syn.method == "input") & (syn.condition == "am-fade-cnr14")]
    sync = syn[(syn.method == "input") & (syn.condition == "am-fade-sync-cnr14")]
    if not env.empty and not sync.empty and env.asr_words.fillna(0).sum():
        out.append(f"**On fading AM the detector matters more than any denoiser:** synchronous detection alone takes "
                   f"WER from {f(wer(env), 1)} % to {f(wer(sync), 1)} %.")
    if real_s is not None:
        r = real_s.drop(index=ex, errors="ignore")
        keep = r[(r.speech_db_med > -3) & (r.output_dbfs_med > MUTE_DBFS)].dropna(subset=["snr_gain_db_med"])
        if not keep.empty:
            b = keep.snr_gain_db_med.idxmax()
            out.append(f"**Real recordings, best SNR gain while keeping speech (speech change > −3 dB):** {link(b)}: "
                       f"{f(keep.loc[b].snr_gain_db_med, 1, True)} dB, noise {f(keep.loc[b].noise_db_med, 1)} dB, "
                       f"speech {f(keep.loc[b].speech_db_med, 2)} dB.")
        cut = r[(r.speech_db_med < DELETED_DB) | (r.output_dbfs_med < MUTE_DBFS)]
        if not cut.empty:
            out.append("**Speech deleted on real radio audio (median speech change < −20 dB or mute):** "
                       + ", ".join(link(m) for m in cut.index) + ".")
    return out


# --------------------------------------------------------------------------- README


def write() -> None:
    frame, corpus, methods = load()
    PLOTS.mkdir(parents=True, exist_ok=True)
    syn = frame[frame["set"] == "synthetic"]
    real = frame[frame["set"] == "real"]
    real_noisy = real[real.group != "digital"]

    board, syn_s = section_leaderboard(syn, methods)
    real_s = summarize(real_noisy) if not real_noisy.empty else None
    if real_s is not None:
        plot_tradeoff(real_s, methods, PLOTS / "real_tradeoff.png",
                      "Real off-air recordings: noise removed vs speech kept")
    plot_tradeoff(syn_s, methods, PLOTS / "synthetic_tradeoff.png", "Synthetic set: noise removed vs speech kept")
    has_wer = "asr_words" in syn and syn.asr_words.fillna(0).sum() > 0
    metric = "wer" if has_wer else "pesq_nb"
    matrix = section_conditions(syn, methods, metric)
    plot_heatmap(matrix, PLOTS / f"conditions_{metric}.png",
                 "Whisper WER % per condition (lower is better)" if has_wer else "PESQ-NB per condition",
                 ".0f" if has_wer else ".2f", vmax=100 if has_wer else 4.5)
    pesq_matrix = section_conditions(syn, methods, "pesq_nb")

    n_syn = syn[syn.method == "input"]
    n_real = real[real.method == "input"]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    today = dt.date.today().isoformat()

    parts = [HEADER.format(
        date=today, machine=f"{platform.machine()} / {platform.system()} {platform.release()}",
        n_methods=len(methods), n_chains=int((methods.family == "chain").sum()),
        n_syn=len(n_syn), syn_min=n_syn.duration_s.sum() / 60 if len(n_syn) else 0,
        n_cond=syn.condition.nunique(), n_real=len(n_real),
        real_min=n_real.duration_s.sum() / 60 if len(n_real) else 0, commit=commit or "uncommitted",
    )]
    parts.append("## Headline findings\n\n_Generated from the data on every run._\n\n"
                 + "\n".join(f"- {x}" for x in findings(syn_s, real_s, syn, real, methods)))
    if real_s is not None:
        parts.append(REAL_INTRO + "\n\n![trade-off on real recordings](results/plots/real_tradeoff.png)\n\n"
                     + "### At a glance\n\n" + section_glance(real, methods))
        for group in ("hf-real", "hf-real-mistuned", "atc", "nfm", "am", "ssb", "iq", "digital"):
            if (real.group == group).any():
                parts.append(f"<details><summary><b>{GROUP_TITLES[group]}</b> — all metrics</summary>\n\n"
                             + real_group_table(real, methods, group) + "\n\n</details>")
    parts.append(SYN_INTRO + "\n\n" + board + "\n\n![trade-off on synthetic set](results/plots/synthetic_tradeoff.png)")
    parts.append("### Per condition\n\n" + f"![per-condition {metric}](results/plots/conditions_{metric}.png)\n\n"
                 + "<details><summary>PESQ-NB per condition (table)</summary>\n\n"
                 + table(["method", *pesq_matrix.columns],
                         [[link(m), *[f(v) for v in row]] for m, row in pesq_matrix.iterrows()])
                 + "\n\n</details>")
    hf = real[real.group == "hf-real"]
    hf_s = summarize(hf) if not hf.empty else None
    parts.append(PORT_INTRO + "\n\n" + section_port(syn_s, real_s, hf_s))
    level_path = RESULTS / "level_sensitivity.csv"
    if level_path.exists():
        lv = pd.read_csv(level_path)
        speech = lv.pivot(index="model", columns="input_dbfs", values="speech_db")
        noise = lv.pivot(index="model", columns="input_dbfs", values="noise_db")
        rows = [[f"`{m}`", *[f"{f(speech.loc[m, c], 1)} / {f(noise.loc[m, c], 1)}" for c in speech.columns]]
                for m in speech.index]
        parts.append(LEVEL_INTRO + "\n\n" + table(["model", *[f"{c} dBFS" for c in speech.columns]], rows))
    ninr = [m for m in methods.method if m.startswith("ninr")]
    if ninr:
        others = [m for m in ("input", "sdrmm-spectral", "sdrmm-spectral-0.5", "sdrmm-rnnoise", "sdrmm-dpdfnet2",
                              "sdrmm-dpdfnet8_8khz", "mmse-lsa") if m in syn_s.index]
        parts.append(NINR_INTRO + "\n\n" + section_versus(syn_s, real_s, others + ninr))
    parts.append(CHAIN_INTRO + "\n\n" + section_chains(syn_s, real_s, methods))
    parts.append(DEMOD_INTRO + "\n\n" + section_demod(syn, syn_s))
    parts.append(methods_section(methods))
    parts.append(corpus_section(corpus))
    parts.append(FOOTER)
    README.write_text("\n\n".join(parts).rstrip() + "\n")
    print(f"wrote {README.relative_to(ROOT)}")


def methods_section(methods: pd.DataFrame) -> str:
    rows = [[link(r.method), r.family, r.description, f"`{r.chain}`" if r.chain else "", f(r.rtf, 4) if pd.notna(r.rtf) else ""]
            for r in methods.itertuples()]
    return METHODS_INTRO + "\n\n" + table(["method", "family", "what it is", "stages", "RTF"], rows, "lllll")


def corpus_section(corpus: pd.DataFrame) -> str:
    syn = corpus[corpus["set"] == "synthetic"].drop_duplicates("condition")
    syn_rows = [[f"`{r.condition}`", r.mode.upper(), r.demod, r.description] for r in syn.itertuples()]
    real = corpus[corpus["set"] == "real"].sort_values(["group", "clip"])
    real_rows = [[r.clip.split("/", 1)[1], r.group, r.mode, r.condition[:70], f"[source]({r.source})", r.license[:40]]
                 for r in real.itertuples()]
    out = CORPUS_INTRO + "\n\n### Simulated links\n\n" + table(["condition", "mode", "detector", "description"], syn_rows, "llll")
    if real_rows:
        out += ("\n\n### Real recordings\n\n<details><summary>" + f"{len(real_rows)} recordings — sources and licenses"
                + "</summary>\n\n" + table(["clip", "group", "mod", "service", "link", "license"], real_rows, "llllll")
                + "\n\n</details>")
    return out


HEADER = """# Speech-over-Radio Benchmark

How well do speech denoisers — classic DSP, small real-time networks, large
enhancement models, generative restorers and **chains of them** — clean up speech that
arrived over a radio link? Radio noise is not the café noise these models are trained
on: FM threshold clicks, AM hiss that pumps with the AGC, SSB band noise, QRN crashes,
heterodyne whistles, selective fading, all squeezed into 300–3000 Hz.

This repository measures that, end to end, on two test sets:

* **Synthetic** — {n_syn} LibriSpeech sentences sent through {n_cond} simulated links
  (NFM, AM envelope/synchronous, SSB; AWGN, Watterson fading, QRN, heterodyne, mistuning),
  {syn_min:.0f} min. Every clip has a clean-channel reference and a transcript, so PESQ,
  ESTOI, SI-SDR and **Whisper WER** are exact.
* **Real** — {n_real} real clips ({real_min:.0f} min):
  * off-air recordings — repeaters, CB, airband ATC/ATIS, shortwave broadcast, HF
    amateur/aero/maritime SSB, satellites, plus HF IQ recordings demodulated here. No
    reference exists, so they are scored the way a listener judges a denoiser: how much the
    pauses are quieted, how much the speech is changed, and DNSMOS;
  * 120 real VHF airband ATC utterances with human transcripts (real-world WER);
  * 96 LibriSpeech utterances actually transmitted over HF SSB and received by KiwiSDRs
    across Europe (Zenodo 5927810) — real channel *and* a clean reference, half of them with
    the receiver 300 Hz off.

{n_methods} methods, {n_chains} of them combined chains. Last run: **{date}** on {machine}
(Apple Silicon, 16 cores, 48 GB), code at commit `{commit}`.

> **The README is generated.** `uv run sorb all` re-runs everything that is not cached
> and rewrites this file from `results/*.csv`; `uv run sorb report` only rewrites it.
> Nothing below is typed by hand except the prose templates in `sorb/report.py`."""

REAL_INTRO = """## Real off-air recordings

Speech/non-speech frames come from Silero VAD on the received audio. Per clip:
**noise suppression** = level change in the non-speech frames, **speech change** = level
change in the speech frames (0 dB = speech kept; it also falls a little because the noise
under the speech goes too), **SNR gain** = speech change − noise change. Medians across
clips, means in brackets. A method whose median speech change is below −20 dB is marked
*speech deleted* and gets no SNR gain; one whose output sits below −60 dBFS is a *complete
mute*. **Bold** = best among methods that keep the speech."""

SYN_INTRO = """## Synthetic set: leaderboard

Against the clean-channel reference (same radio, noise switched off). Sorted by the
**mean rank** over four views of "better": PESQ-NB (perceived quality), ESTOI
(intelligibility for a listener), DNSMOS OVRL (reference-free quality) and Whisper
large-v3-turbo WER (intelligibility for a machine). The input is ranked too, so a method
below it is worse than doing nothing. **Bold** = best. RTF = inference time / audio time, each
stage timed alone on six clips with the machine otherwise idle (`sorb speed`), summed
over a chain's stages; model loading is excluded. Lower is faster; 1.0 = real time."""

PORT_INTRO = """## SDR-- DPDFNet vs upstream DPDFNet

The same eight DPDFNet weight sets run through two inference stacks: the SDR-- Rust port
(`sdrmm-*`: streaming, its own STFT/resampling, wet/dry floor at 30 dB) and Ceva's
official ONNX pipeline (`dpdfnet-*`, no attenuation limit). Each cell is
**SDR-- / upstream**. SDR-- mixes the dry signal back in at −30 dB (strength 1.0), so an
SDR-- speech change near −30 dB means the model itself removed the speech and only the
floor is left — compare with the upstream column, which has no floor. The
`lavasr→*-dpdfnet2` chains test whether the deletion is a bandwidth problem: they extend
the 300–3000 Hz radio band to wideband first, then denoise."""

NINR_INTRO = """## SDR# NINR vs SDR--

AIRSPY's NINR (SDR# "next", June 2026) is closed source; the benchmark runs it through
SDR#'s own `shark.dll` under a portable Wine (`tools/setup_ninr.sh`, then point
`SORB_SDRSHARP` at your SDR# folder — the DLL is not redistributed here). Like SDR#, the
audio is resampled to 2.4 × the profile bandwidth, fed as complex samples and resampled
back. **LP** is the current engine (`noise_filter_lp_*`) with SDR#'s built-in profiles and
its default 10 dB depth; **HQ** is the classic threshold/slope engine (`noise_filter_*`),
which this SDR# build no longer exposes, so its settings (threshold −30 dB, slope 50,
30 dB depth) are this benchmark's choice."""

LEVEL_INTRO = """### Is it the input level?

Official DPDFNet on the same synthetic clips (NFM 10/16 dB, AM 14 dB, SSB 10 dB) with the
input scaled to different RMS levels (`sorb level`). Cells are **speech change / noise
change** in dB, median over clips. The corpus itself sits at −26 dBFS."""

CHAIN_INTRO = """## Do combinations help?

Each chain against the best of its own stages run alone. A chain "helps" when it beats
that stage on synthetic WER (PESQ-NB if ASR was not run)."""

DEMOD_INTRO = """## Demodulator vs denoiser

The same weak AM signals detected two ways. A synchronous detector only sees the
in-phase noise and has no threshold effect, so it should help most where the envelope
detector breaks down. Shown for the input and the three best methods overall."""

METHODS_INTRO = """## Methods

`sdrmm-*` methods run the **actual SDR-- Rust code** (pinned commit, see
`tools/sdrmm-denoise`), not a re-implementation. Classic DSP lives in `sorb/dsp.py`.
Neural engines run in their own uv environments under `engines/` so their dependency
pins never collide. Models whose memory grows with input length (SepFormer, CMGAN) run
on 10 s / 4 s chunks with a short cross-fade.

Installed but not in this run: **MP-SENet** (works via audiosronnx, but RTF ≈ 2.5 on this
CPU), **MossFormerGAN** (RTF ≈ 1.3), and the multi-GB generative restorers (UniPASE,
Resemble-Enhance, AnyEnhance, …) — see the `engines/` notes to add them."""

CORPUS_INTRO = """## Corpus

Synthetic clips: 12 LibriSpeech test-clean speakers (6 F / 6 M), one 8–15 s sentence each,
with 1 s of keyed carrier before and after (`sorb/corpus.py`). The radio links are
simulated at 48 kHz complex baseband in `sorb/channel.py`: transmitter speech processing
(band-limiting, pre-emphasis/compression, deviation limiter), the channel (AWGN at a set
CNR/SNR, ITU-R F.1487 Watterson fading, Poisson QRN bursts, a drifting heterodyne), and a
receiver (channel filter, discriminator / envelope / synchronous / product detector,
de-emphasis, AGC). The reference is the same receiver with the noise switched off.
Real recordings are downloaded from their original hosts and never redistributed here."""

FOOTER = """## Metrics

| metric | set | meaning |
| :-- | :-- | :-- |
| WER % | synthetic | Whisper large-v3-turbo (MLX), greedy, English; Whisper text normaliser; corpus-level |
| PESQ-NB / PESQ-WB | synthetic | ITU-T P.862 at 8 kHz / P.862.2 at 16 kHz vs. the clean-channel reference |
| ESTOI | synthetic | extended STOI, intelligibility proxy |
| SI-SDR | synthetic | scale-invariant SDR, dB |
| DNSMOS OVRL | both | Microsoft DNSMOS P.835 overall, reference-free (trained on wideband speech, so radio scores low in absolute terms — compare differences) |
| noise / speech Δ, SNR gain | both | VAD-gated level changes (oracle labels on synthetic, Silero VAD on real) |

## Limitations

* **Whisper is not a person.** It is trained on noisy audio and is hurt by enhancement
  artifacts, so WER punishes aggressive denoisers harder than a listener would. Read it next
  to ESTOI/PESQ, not instead of them.
* **WER is capped per clip** at the clip's word count (Whisper can hallucinate paragraphs on
  hopeless audio), and number styles are normalised ("3.1", "31", "three one" → "3 1").
  ATC transcripts follow their own conventions, so ATC WER is only comparable between methods.
* **DNSMOS** was trained on wideband speech; radio audio scores low in absolute terms.
* **Real-recording speech/noise frames** come from Silero VAD on the received audio — it can
  mislabel very noisy stretches. SI-SDR is not reported for real HF (the link's phase response
  makes sample-exact comparison meaningless).
* **12 speakers, 19 links** on the synthetic side; the real side is whatever was freely available.
* SepFormer and CMGAN run on 10 s / 4 s chunks to bound memory; outputs are delay-compensated
  by cross-correlation (`sorb realign`).

## Reproduce

```sh
uv sync                                            # harness + metrics (Python 3.12)
(cd tools/sdrmm-denoise && cargo build --release)  # SDR-- engines (Rust 1.99, pinned sdrmm commit)
uv run sorb all                                    # corpus, process, score, asr, speed, level, report
```

SDR# NINR needs your own SDR# "next" build and a portable Wine (nothing system-wide):

```sh
tools/setup_ninr.sh
export SORB_SDRSHARP=/path/to/sdrsharp-x86-next   # folder with shark.dll
uv run sorb process -m 'ninr*'
```

Step by step (everything is cached; re-running only does what is missing):

```sh
uv run sorb corpus                    # LibriSpeech + simulated links, real recordings, ATC, real HF
uv run sorb process -m 'sdrmm-*'      # run methods (globs); each engine in its own uv env
uv run sorb realign                   # re-measure stage delays, fix shifted outputs
uv run sorb score                     # PESQ / ESTOI / SI-SDR / DNSMOS / level changes
uv run sorb asr                       # Whisper (MLX) WER, with a stall watchdog
uv run sorb speed                     # RTF per stage, run alone
uv run sorb level                     # DPDFNet speech retention vs input level
uv run sorb report                    # rewrite this README from results/
```

Environment knobs: `SORB_LANES` (parallel engine processes, default 6), `SORB_SCORE_JOBS`
(default 4), `SORB_ASR_JOBS` (default 3). Peak memory with the defaults stays under ~12 GB.

Downloads: LibriSpeech test-clean (346 MB), the real recordings (~1 GB), the Zenodo HF set
(1.3 GB), ATC parquet files (245 MB), SDR-- models from `downloads.sdrmm.com`, DNSMOS ONNX,
Whisper weights, and each engine's weights on first use (all under `models/`).
Audio is cached as 24-bit FLAC at 16 kHz (radio speech has nothing above 4 kHz)."""
