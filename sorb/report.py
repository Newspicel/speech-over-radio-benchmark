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
    "nfm": "NFM: repeaters, simplex, CB, scanner",
    "am": "AM: airband, CB, shortwave, LW, HF AM",
    "ssb": "SSB: HF amateur, aero, marine, CB, satellite",
    "iq": "HF IQ recordings, demodulated here",
    "hf-real": "HF SSB over KiwiSDRs, tuned (Zenodo 5927810)",
    "hf-real-mistuned": "HF SSB over KiwiSDRs, 300 Hz off",
    "atc": "VHF airband ATC with transcripts",
    "digital": "Decoded DMR and FreeDV (vocoder audio)",
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
        return "-"
    return (f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}") + suffix


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
    return f"**{text}**" if cond and text != "-" else text


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
    ax.set_xlabel("noise change in pauses, dB (median)", color=INK_2, fontsize=9)
    ax.set_ylabel("speech change, dB (median)", color=INK_2, fontsize=9)
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
    header = ["#", "method", "rank", "WER %", "PESQ-NB", "PESQ-WB", "ESTOI", "SI-SDR", "DNSMOS",
              "noise dB", "speech dB", "RTF"]
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
            row = [link("input"), "", "", ""]
        else:
            noise = f"mute ({f(r.noise_db_mean, 1)})" if muted else f"{f(r.noise_db_med, 1)} ({f(r.noise_db_mean, 1)})"
            speech = f(r.speech_db_med, 1) + (" deleted" if deleted else "")
            gain = "-" if (muted or deleted) else f(r.snr_gain_db_med, 1, signed=True)
            row = [link(method), bold_if(noise, method in best_noise), speech, bold_if(gain, method in best_gain)]
        if has_wer:
            row.append(bold_if(f(r.get("wer"), 1), method in best_wer))
        if has_ref:
            row += [bold_if(f(r.pesq_nb), method in best_pesq), bold_if(f(r.estoi, 3), method in best_estoi)]
        row += [bold_if(f(r.dnsmos_ovr), method in best_mos), "" if method == "input" else f"{r.deleted * 100:.0f} %"]
        rows.append(row)
    header = ["method", "noise, median (mean)", "speech", "SNR gain"]
    if has_wer:
        header.append("WER %")
    if has_ref:
        header += ["PESQ-NB", "ESTOI"]
    header += ["DNSMOS", "speech deleted"]
    n = len(g[g.method == "input"])
    return f"{n} clips, {g[g.method == 'input'].duration_s.sum() / 60:.1f} min.\n\n" + table(header, rows)


GLANCE_GROUPS = (("nfm", "NFM"), ("am", "AM"), ("ssb", "SSB"), ("hf-real", "HF link"))


def section_glance(real: pd.DataFrame, methods: pd.DataFrame) -> str:
    """Engine rows, per radio group: noise suppression median (mean), speech change, SNR gain."""
    summaries = {g: summarize(real[real.group == g]) for g, _ in GLANCE_GROUPS if (real.group == g).any()}
    best = {}
    for g, sm in summaries.items():
        usable = sm[(sm.speech_db_med > DELETED_DB) & (sm.output_dbfs_med > MUTE_DBFS)].drop(index="input", errors="ignore")
        best[g] = (best_marks(usable.noise_db_med, False, set()), best_marks(usable.speech_db_med.abs(), False, set()),
                   best_marks(usable.snr_gain_db_med, True, set()))
    header, rows = ["method"], []
    for g, title in GLANCE_GROUPS:
        if g in summaries:
            header += [f"{title} noise", f"{title} speech", f"{title} SNR gain"]
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
            noise = f"mute ({f(r.noise_db_mean, 1)})" if muted else f"{f(r.noise_db_med, 1)} ({f(r.noise_db_mean, 1)})"
            speech = f(r.speech_db_med, 1) + (" deleted" if deleted else "")
            gain = "-" if muted or deleted else f(r.snr_gain_db_med, 1, signed=True)
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
                     f(q.noise_db_med, 1) if q is not None else "-", f(q.speech_db_med, 2) if q is not None else "-",
                     f(q.snr_gain_db_med, 1, True) if q is not None else "-"])
    return table(["method", "WER %", "PESQ-NB", "ESTOI", "DNSMOS", "real noise dB", "real speech dB",
                  "real SNR gain"], rows)


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
        real_txt = "-"
        if real_s is not None and r.method in real_s.index and best_part in real_s.index:
            real_txt = (f"{f(real_s.loc[r.method].snr_gain_db_med, 1, True)} vs "
                        f"{f(real_s.loc[best_part].snr_gain_db_med, 1, True)}")
        rows.append([link(r.method), link(best_part),
                     f"{f(chain_v, 1 if key == 'wer' else 2)} vs {f(part_v, 1 if key == 'wer' else 2)}",
                     f(syn_s.loc[r.method].pesq_nb) + " vs " + f(syn_s.loc[best_part].pesq_nb),
                     real_txt, "better" if better else "worse"])
    if not rows:
        return "No chains run."
    return table(["chain", "best stage alone", "WER %", "PESQ-NB", "real SNR gain", ""], rows, "llrrrl")


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
                    vals.append(("-", "-"))
                    continue
                vals.append((f(wer(g), 1) if "asr_words" in g and g.asr_words.sum() else "-", f(g.pesq_nb.mean())))
            rows.append([env.replace("am-", "").replace("-", " "), link(method), *vals[0], *vals[1]])
    return table(["condition", "method", "envelope WER %", "envelope PESQ", "sync WER %", "sync PESQ"],
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
            return f(frame.loc[method].get(col), digits) if frame is not None and method in frame.index else "-"
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
        return "Not run yet."
    return table(["model", "WER %", "PESQ-NB", "speech dB", "real speech dB", "real noise dB",
                  "HF link WER %"], rows, "lrrrrrr")


def findings(syn_s, real_s, syn, real, methods) -> list[str]:
    out = []
    s = syn_s.drop(index=["input"], errors="ignore")
    base = syn_s.loc["input"] if "input" in syn_s.index else None
    wers = s["wer"].dropna() if "wer" in s else pd.Series(dtype=float)
    if not wers.empty and base is not None:
        worse = wers[wers > base.wer]
        out.append(f"Best Whisper WER: {link(wers.idxmin())}, {f(wers.min(), 1)} % (input {f(base.wer, 1)} %). "
                   f"{len(worse)} of {len(wers)} methods make WER worse than doing nothing.")
        best = []
        for mode, name in (("nfm", "NFM"), ("am", "AM"), ("usb", "SSB")):
            sub = summarize(syn[syn["mode"] == mode]).drop(index=["input"], errors="ignore")
            w = sub["wer"].dropna() if "wer" in sub else pd.Series(dtype=float)
            if not w.empty:
                best.append(f"{name} {link(w.idxmin())} ({f(w.min(), 1)} %)")
        if best:
            out.append("Best WER per mode: " + ", ".join(best) + ".")
    if "pesq_nb" in s and base is not None:
        out.append(f"Best PESQ-NB: {link(s.pesq_nb.idxmax())}, {f(s.pesq_nb.max())} (input {f(base.pesq_nb)}). "
                   f"Best ESTOI: {link(s.estoi.idxmax())}, {f(s.estoi.max(), 3)} (input {f(base.estoi, 3)}).")
    wide = [m for m in ("dpdfnet-baseline", "dpdfnet-dpdfnet2", "dpdfnet-dpdfnet4", "dpdfnet-dpdfnet8",
                        "dpdfnet-dpdfnet2_48khz_hr", "dpdfnet-dpdfnet8_48khz_hr") if m in syn_s.index]
    narrow = [m for m in ("dpdfnet-dpdfnet2_8khz", "dpdfnet-dpdfnet8_8khz") if m in syn_s.index]
    if wide and narrow:
        out.append(f"DPDFNet 16 and 48 kHz models delete radio speech ({f(syn_s.loc[wide, 'speech_db_med'].max(), 0)} "
                   f"to {f(syn_s.loc[wide, 'speech_db_med'].min(), 0)} dB). The 8 kHz models keep it "
                   f"({f(syn_s.loc[narrow, 'speech_db_med'].mean(), 1)} dB). Upstream does the same, so it is the "
                   "model, not the SDR-- port.")
    lags = []
    for meta in (ROOT / "data" / "out").glob("*/stage.json"):
        import json

        m = json.loads(meta.read_text())
        if len(m["chain"]) == 1 and m["chain"][0].startswith("sdrmm:neural") and m.get("lag_samples"):
            lags.append(m["lag_samples"] / 48)
    if lags:
        out.append(f"SDR-- neural output is {min(lags):.0f} to {max(lags):.0f} ms later than `latency()` reports.")
    if "sdrmm-autonotch" in syn_s.index and base is not None and "wer" in syn_s:
        a = syn_s.loc["sdrmm-autonotch"]
        txt = (f"The SDR-- auto-notch removes speech ({f(a.speech_db_med, 1)} dB, WER {f(a.wer, 1)} %). "
               "Its 4-sample delay makes voiced speech predictable.")
        if "lms-notch" in syn_s.index:
            txt += f" With a 2 ms delay (`lms-notch`) speech stays ({f(syn_s.loc['lms-notch'].speech_db_med, 1)} dB)."
        out.append(txt)
    env = syn[(syn.method == "input") & (syn.condition == "am-fade-cnr14")]
    sync = syn[(syn.method == "input") & (syn.condition == "am-fade-sync-cnr14")]
    if not env.empty and not sync.empty and env.asr_words.fillna(0).sum():
        out.append(f"Fading AM: a sync detector alone takes WER from {f(wer(env), 1)} % to {f(wer(sync), 1)} %, "
                   "more than any denoiser.")
    if real_s is not None:
        r = real_s.drop(index=["input"], errors="ignore")
        keep = r[(r.speech_db_med > -3) & (r.output_dbfs_med > MUTE_DBFS)].dropna(subset=["snr_gain_db_med"])
        if not keep.empty:
            b = keep.snr_gain_db_med.idxmax()
            out.append(f"Real recordings, best SNR gain with speech kept: {link(b)}, "
                       f"{f(keep.loc[b].snr_gain_db_med, 1, True)} dB.")
        cut = r[(r.speech_db_med < DELETED_DB) | (r.output_dbfs_med < MUTE_DBFS)]
        if not cut.empty:
            out.append("Speech deleted on real recordings: " + ", ".join(link(m) for m in cut.index) + ".")
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
                      "Real recordings: noise removed vs speech kept")
    plot_tradeoff(syn_s, methods, PLOTS / "synthetic_tradeoff.png", "Synthetic: noise removed vs speech kept")
    has_wer = "asr_words" in syn and syn.asr_words.fillna(0).sum() > 0
    metric = "wer" if has_wer else "pesq_nb"
    matrix = section_conditions(syn, methods, metric)
    plot_heatmap(matrix, PLOTS / f"conditions_{metric}.png",
                 "Whisper WER % per condition" if has_wer else "PESQ-NB per condition",
                 ".0f" if has_wer else ".2f", vmax=100 if has_wer else 4.5)
    pesq_matrix = section_conditions(syn, methods, "pesq_nb")

    n_syn = syn[syn.method == "input"]
    n_real = real[real.method == "input"]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    today = dt.date.today().isoformat()

    parts = [HEADER.format(
        date=today, machine="Apple Silicon Mac" if platform.machine() == "arm64" and platform.system() == "Darwin" else platform.platform(),
        n_methods=len(methods), n_chains=int((methods.family == "chain").sum()),
        n_syn=len(n_syn), syn_min=n_syn.duration_s.sum() / 60 if len(n_syn) else 0,
        n_cond=syn.condition.nunique(), n_real=len(n_real),
        real_min=n_real.duration_s.sum() / 60 if len(n_real) else 0, commit=commit or "uncommitted",
    )]
    parts.append("## Findings\n\n"
                 + "\n".join(f"- {x}" for x in findings(syn_s, real_s, syn, real, methods)))
    if real_s is not None:
        parts.append(REAL_INTRO + "\n\n![trade-off on real recordings](results/plots/real_tradeoff.png)\n\n"
                     + "### Summary\n\n" + section_glance(real, methods))
        for group in ("hf-real", "hf-real-mistuned", "atc", "nfm", "am", "ssb", "iq", "digital"):
            if (real.group == group).any():
                parts.append(f"<details><summary><b>{GROUP_TITLES[group]}</b></summary>\n\n"
                             + real_group_table(real, methods, group) + "\n\n</details>")
    parts.append(SYN_INTRO + "\n\n" + board + "\n\n![trade-off on synthetic set](results/plots/synthetic_tradeoff.png)")
    parts.append("### Per condition\n\n" + f"![per-condition {metric}](results/plots/conditions_{metric}.png)\n\n"
                 + "<details><summary>PESQ-NB per condition</summary>\n\n"
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
    ninr = [m for m in methods.method if m.startswith("sdr#-ninr")]
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
    return METHODS_INTRO + "\n\n" + table(["method", "family", "description", "stages", "RTF"], rows, "lllll")


def corpus_section(corpus: pd.DataFrame) -> str:
    syn = corpus[corpus["set"] == "synthetic"].drop_duplicates("condition")
    syn_rows = [[f"`{r.condition}`", r.mode.upper(), r.demod, r.description] for r in syn.itertuples()]
    real = corpus[corpus["set"] == "real"].sort_values(["group", "clip"])
    real_rows = [[r.clip.split("/", 1)[1], r.group, r.mode, r.condition[:70], f"[source]({r.source})", r.license[:40]]
                 for r in real.itertuples()]
    out = CORPUS_INTRO + "\n\n### Simulated links\n\n" + table(["condition", "mode", "detector", "description"], syn_rows, "llll")
    if real_rows:
        out += ("\n\n### Real recordings\n\n<details><summary>" + f"{len(real_rows)} recordings, sources and licenses"
                + "</summary>\n\n" + table(["clip", "group", "mod", "service", "link", "license"], real_rows, "llllll")
                + "\n\n</details>")
    return out


HEADER = """# Speech over Radio Benchmark

Compares speech denoisers on audio received over radio: NFM, AM and SSB, with hiss,
threshold clicks, fading, static crashes and heterodynes.

- **Synthetic:** {n_syn} LibriSpeech sentences through {n_cond} simulated links ({syn_min:.0f} min).
  Clean reference and transcript for every clip.
- **Real:** {n_real} clips ({real_min:.0f} min). Off-air recordings, airband ATC with transcripts,
  and LibriSpeech sent over real HF links (clean reference available).

{n_methods} methods, {n_chains} of them chains. Run on {date}, {machine}, commit `{commit}`.
This file is generated by `uv run sorb report`."""

REAL_INTRO = """## Real recordings

Speech and pauses are found with Silero VAD. **noise** is the level change in pauses,
**speech** the level change during speech, **SNR gain** the difference, all in dB. Medians,
means in brackets. Below -20 dB of speech change counts as deleted. Bold is best."""

SYN_INTRO = """## Synthetic leaderboard

Scored against the same radio with the noise switched off. **rank** is the mean rank over
PESQ-NB, ESTOI, DNSMOS and WER. RTF is processing time per second of audio, one stage at a time."""

PORT_INTRO = """## SDR-- vs upstream DPDFNet

Same weights, two runtimes. Cells are **SDR-- / upstream**. SDR-- mixes the input back in
at -30 dB, so -30 dB of speech change means the model removed the speech."""

NINR_INTRO = """## SDR# NINR vs SDR--

NINR runs from SDR#'s own `shark.dll` under Wine, with SDR#'s resampling and profile values.
**LP** is the current engine at the default 10 dB depth. **HQ** is the older threshold/slope
engine; SDR# no longer exposes it, so its settings here (threshold -30 dB, slope 50,
depth 30 dB) are a guess."""

LEVEL_INTRO = """### Input level

Upstream DPDFNet at different input levels (`sorb level`). Cells are speech / noise change
in dB. The corpus sits at -26 dBFS."""

CHAIN_INTRO = """## Chains

Each chain against the better of its stages run alone."""

DEMOD_INTRO = """## Demodulator

Envelope vs synchronous AM detection, for the input and the three best methods."""

METHODS_INTRO = """## Methods

`sdrmm-*` runs the SDR-- Rust code at a pinned commit. `sdr#-ninr-*` runs SDR#'s NINR.
Neural engines each have their own uv environment in `engines/`. SepFormer and CMGAN run
in chunks to bound memory. Not included: MP-SENet (too slow on CPU), MossFormerGAN and
the large generative models."""

CORPUS_INTRO = """## Corpus

Synthetic: 12 LibriSpeech speakers (6 F, 6 M), 8 to 15 s each, with 1 s of carrier before
and after. `sorb/channel.py` simulates the transmitter, the channel and the receiver. Real
audio is downloaded from the original hosts and not redistributed."""

FOOTER = """## Metrics

| metric | meaning |
| :-- | :-- |
| WER | Whisper large-v3-turbo, errors capped at the clip's word count |
| PESQ-NB / WB | ITU-T P.862 at 8 kHz / P.862.2 at 16 kHz |
| ESTOI | intelligibility estimate |
| SI-SDR | synthetic only |
| DNSMOS | Microsoft DNSMOS P.835 overall, no reference needed |
| noise, speech, SNR gain | level changes in pauses and in speech |

## Limitations

- Whisper is trained on noisy audio and dislikes denoiser artifacts. Read WER next to ESTOI.
- DNSMOS expects wideband speech, so radio scores low overall.
- VAD can mislabel very noisy audio.
- ATC transcripts use their own conventions; compare ATC WER between methods only.

## Run

```sh
uv sync
(cd tools/sdrmm-denoise && cargo build --release)
uv run sorb all
```

Steps, all cached:

```sh
uv run sorb corpus      # download and build the test sets
uv run sorb process     # run methods, -m takes names or globs
uv run sorb score       # PESQ, ESTOI, SI-SDR, DNSMOS, levels
uv run sorb asr         # Whisper WER
uv run sorb speed       # RTF per stage
uv run sorb level       # DPDFNet input level test
uv run sorb report      # rewrite this README
```

SDR# NINR needs your own SDR# copy:

```sh
tools/setup_ninr.sh
export SORB_SDRSHARP=/path/to/sdrsharp-x86-next
uv run sorb process -m 'sdr#*'
```

Audio is cached as 16 kHz FLAC in `data/`, models in `models/`.

## License

ISC. `tools/sdrmm-denoise` links SDR-- and is AGPL-3.0-or-later.
"""
