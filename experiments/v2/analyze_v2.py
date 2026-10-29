"""Aggregate v2 results into paper tables, tests and figures.

Reads experiments/v2/results/results_seed*.json and writes
experiments/v2/analysis/ (summary.md, *.csv, *.png).

Usage
    python -m experiments.v2.analyze_v2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RES_DIR = ROOT / "experiments" / "v2" / "results"
OUT = ROOT / "experiments" / "v2" / "analysis"

LABEL_FREE = ("no_adapt", "src_lgbm", "coral_rf", "coral_deep", "mmd", "dann")
SUPERVISED = ("finetune", "contrastive", "tradaboost", "pooled_lgbm")
TARGET_ONLY = ("tgt_rf", "tgt_lgbm", "tgt_mlp")
CONTROLS = ("finetune_zf", "contrastive_zf")
ORDER = LABEL_FREE + SUPERVISED + TARGET_ONLY + CONTROLS
HONEST = "budget_plugin"   # primary deployable policy when labels exist


def load() -> pd.DataFrame:
    rows = []
    for f in sorted(RES_DIR.glob("results_seed*.json")):
        for r in json.load(open(f)):
            if "error" in r:
                continue
            base = {k: r[k] for k in ("seed", "src", "tgt", "method", "frac", "pr_auc", "roc_auc")}
            h = r.get("hold") or {}
            base.update(hold_n_pos=h.get("n_pos", np.nan),
                        hold_f1=np.nan if h.get("f1_opt") is None else h["f1_opt"],
                        hold_pr_auc=np.nan if h.get("pr_auc") is None else h["pr_auc"])
            for pol, m in r["policies"].items():
                rows.append({**base, "policy": pol, "f1": m["f1"],
                             "precision": m["precision"], "recall": m["recall"],
                             "thr": m["thr"], "prior_est": m.get("prior_est", np.nan),
                             "f1_est": m.get("f1_est", np.nan)})
    return pd.DataFrame(rows)


def expand_target_only(df: pd.DataFrame) -> pd.DataFrame:
    """Copy target-only rows onto every (src, tgt) pair so pairs line up."""
    pairs = df[df.src != "none"][["src", "tgt"]].drop_duplicates()
    t = df[df.src == "none"].drop(columns="src").merge(pairs, on="tgt")
    return pd.concat([df[df.src != "none"], t], ignore_index=True)


def ci95(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.nan
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def fmt(m, c):
    return f"{m:.3f}" if not np.isfinite(c) else f"{m:.3f} ± {c:.3f}"


def method_table(df, policy, metric="f1"):
    """Mean over pairs per seed, then mean ± CI over seeds; method x frac."""
    sub = df[df.policy == policy]
    per_seed = sub.groupby(["method", "frac", "seed"])[metric].mean().reset_index()
    agg = per_seed.groupby(["method", "frac"])[metric].agg(["mean", ci95]).reset_index()
    agg["cell"] = [fmt(m, c) for m, c in zip(agg["mean"], agg["ci95"])]
    tab = agg.pivot(index="method", columns="frac", values="cell")
    return tab.reindex([m for m in ORDER if m in tab.index])


def paired_test(df, a_methods, b_methods, policy, frac):
    """Best-of-a vs best-of-b per (seed, pair); Wilcoxon over pairs (seed-mean)."""
    sub = df[(df.policy == policy) & (df.frac == frac)]
    a = sub[sub.method.isin(a_methods)].groupby(["seed", "src", "tgt"]).f1.max()
    b = sub[sub.method.isin(b_methods)].groupby(["seed", "src", "tgt"]).f1.max()
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if j.empty:
        return None
    j = j.groupby(["src", "tgt"]).mean()
    d = j.a - j.b
    p = stats.wilcoxon(d).pvalue if (d != 0).sum() >= 1 else np.nan
    return {"frac": frac, "mean_a": j.a.mean(), "mean_b": j.b.mean(),
            "diff": d.mean(), "wins_a": int((d > 0).sum()), "n": len(d), "p": p}


CANDIDATES = SUPERVISED + TARGET_ONLY
DEV_SEED = 42   # the selector rule was fixed after inspecting this seed only


def selection(df: pd.DataFrame) -> pd.DataFrame:
    """Budget-only model selection per (seed, pair, frac).

    Every candidate is thresholded by budget_plugin.  A selector picks one
    candidate using budget information only; we record the test F1 it gets.
      plugin_est   argmax of the plug-in F1 estimate (hold-out Platt + pool)
      hold_f1      argmax of the empirical best F1 on the hold-out labels
      hold_pr_auc  argmax of hold-out PR-AUC
    Fixed choices and the per-cell test oracle are reported alongside.
    """
    sub = df[(df.policy == HONEST) & df.method.isin(CANDIDATES) & (df.frac > 0)]
    out = []
    for (seed, s, t, f), g in sub.groupby(["seed", "src", "tgt", "frac"]):
        g = g.set_index("method")
        rec = {"seed": seed, "src": s, "tgt": t, "frac": f,
               "n_pos": g.hold_n_pos.max(),
               "oracle": g.f1.max(),
               "always_tgt_lgbm": g.f1.get("tgt_lgbm", np.nan),
               "always_pooled_lgbm": g.f1.get("pooled_lgbm", np.nan)}
        for name, col in (("plugin_est", "f1_est"), ("hold_f1", "hold_f1"),
                          ("hold_pr_auc", "hold_pr_auc")):
            c = g[col].dropna()
            rec[name] = g.f1[c.idxmax()] if len(c) else np.nan
            if name == "plugin_est" and len(c):
                rec["pick"] = c.idxmax()
        out.append(rec)
    return pd.DataFrame(out)


SELECTORS = ("plugin_est", "hold_f1", "hold_pr_auc", "always_tgt_lgbm",
             "always_pooled_lgbm", "oracle")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = expand_target_only(load())
    seeds = sorted(df.seed.unique())
    md = [f"# v2 results summary\n\nSeeds: {seeds}; rows: {len(df):,}\n"]

    # 1. label-free methods at zero labels, every label-free policy
    lf = df[(df.frac == 0.0) & df.method.isin(LABEL_FREE)]
    t = (lf.groupby(["method", "policy", "seed"]).f1.mean().groupby(["method", "policy"])
         .agg(["mean", ci95]).reset_index())
    t["cell"] = [fmt(m, c) for m, c in zip(t["mean"], t["ci95"])]
    t = t.pivot(index="method", columns="policy", values="cell")
    t = t.reindex([m for m in LABEL_FREE if m in t.index])
    t.to_csv(OUT / "t1_zero_label_policies.csv")
    md += ["## T1. Zero target labels: test F1 by threshold policy\n", t.to_markdown(), "\n"]
    pr0 = lf.groupby(["method", "seed"]).pr_auc.mean().groupby("method").agg(["mean", ci95])
    md += ["PR-AUC at zero labels:\n", pr0.reindex([m for m in LABEL_FREE if m in pr0.index]).round(4).to_markdown(), "\n"]

    # 2. labelled budgets: honest policy, PR-AUC, oracle
    lab = df[df.frac > 0]
    for name, pol, metric in (("T2_f1_budget_plugin", HONEST, "f1"),
                              ("T3_f1_budget_f1", "budget_f1", "f1"),
                              ("T4_f1_fixed", "fixed", "f1"),
                              ("T5_f1_oracle_val", "oracle_val", "f1"),
                              ("T6_pr_auc", "fixed", "pr_auc")):
        tab = method_table(lab, pol, metric)
        tab.to_csv(OUT / f"{name}.csv")
        md += [f"## {name}\n", tab.to_markdown(), "\n"]

    # 3. transfer vs target-only
    md.append("## T7. Best transfer vs best target-only (budget_plugin, Wilcoxon over 12 pairs)\n")
    rows = [paired_test(lab, SUPERVISED, TARGET_ONLY, HONEST, f) for f in sorted(lab.frac.unique())]
    t7 = pd.DataFrame([r for r in rows if r])
    t7.to_csv(OUT / "t7_transfer_vs_target_only.csv", index=False)
    md += [t7.round(4).to_markdown(index=False), "\n"]

    # per-pair view at each budget
    sub = lab[lab.policy == HONEST]
    best = lambda ms: sub[sub.method.isin(ms)].groupby(["src", "tgt", "frac"]).f1.mean()
    pp = pd.concat([best(SUPERVISED).rename("transfer_best_mean"),
                    best(TARGET_ONLY).rename("target_only_best_mean")], axis=1)
    pp.to_csv(OUT / "t8_per_pair.csv")

    # 4. threshold policies: honest vs oracle gap, direct-F1 vs plug-in
    md.append("## T9. Threshold policy comparison (labelled budgets, all supervised + target-only)\n")
    pol = (lab[lab.method.isin(SUPERVISED + TARGET_ONLY)]
           .groupby(["frac", "policy", "seed"]).f1.mean().groupby(["frac", "policy"]).mean()
           .unstack("policy"))
    pol.to_csv(OUT / "t9_policy_by_frac.csv")
    md += [pol.round(3).to_markdown(), "\n"]

    # 5. prior estimation quality (label-free methods)
    pe = lf[lf.policy.isin(["em_plugin", "bbse_plugin"])].copy()
    if not pe.empty:
        truth = {}
        for f in sorted(RES_DIR.glob("results_seed*.json")):
            for r in json.load(open(f)):
                if "error" not in r and "prior_plugin" in r.get("policies", {}):
                    truth[(r["seed"], r["tgt"])] = r["policies"]["prior_plugin"]["prior_est"]
        pe["true_prior"] = [truth.get((s, t), np.nan) for s, t in zip(pe.seed, pe.tgt)]
        pe["ratio"] = pe.prior_est / pe.true_prior
        t10 = pe.groupby(["policy", "tgt"]).ratio.median().unstack("tgt")
        t10.to_csv(OUT / "t10_prior_ratio.csv")
        md += ["## T10. Estimated / true target prior (median over methods, pairs, seeds)\n",
               t10.round(2).to_markdown(), "\n"]

    # 6. v1-design control: zero_fill vs union
    ctl = lab[(lab.policy == HONEST) & lab.method.isin(CONTROLS + ("finetune", "contrastive"))]
    if ctl.method.isin(CONTROLS).any():
        t11 = ctl.groupby(["method", "frac"]).f1.mean().unstack("frac")
        t11.to_csv(OUT / "t11_alignment_control.csv")
        md += ["## T11. Union vs v1 zero_fill alignment (seed 42)\n", t11.round(3).to_markdown(), "\n"]

    # 7. budget-only model selection (development seed vs confirmation seeds)
    sel = selection(df)
    sel.to_csv(OUT / "t12_selection_cells.csv", index=False)
    sel["split"] = np.where(sel.seed == DEV_SEED, "development", "confirmation")
    cols = [c for c in SELECTORS if sel[c].notna().any()]
    t12 = sel.groupby(["split", "frac"])[cols].mean()
    t12.to_csv(OUT / "t12_selection.csv")
    md += ["## T12. Budget-only model selection: test F1 of the selected model\n",
           t12.round(3).to_markdown(), "\n"]
    conf = sel[sel.split == "confirmation"]
    if not conf.empty:
        tests = []
        for base in ("hold_f1", "always_tgt_lgbm", "always_pooled_lgbm"):
            for f in sorted(conf.frac.unique()):
                c = conf[conf.frac == f].groupby(["src", "tgt"])[["plugin_est", base]].mean().dropna()
                d = c.plugin_est - c[base]
                p = stats.wilcoxon(d).pvalue if (d != 0).any() else np.nan
                tests.append({"vs": base, "frac": f, "diff": d.mean(),
                              "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "p": p})
        t13 = pd.DataFrame(tests)
        t13.to_csv(OUT / "t13_selection_tests.csv", index=False)
        md += ["## T13. plugin_est selector vs baselines (confirmation seeds, Wilcoxon over pairs)\n",
               t13.round(4).to_markdown(index=False), "\n"]
    est = df[(df.policy == HONEST) & df.method.isin(CANDIDATES) & (df.frac > 0)][["f1_est", "f1"]].dropna()
    md += [f"Plug-in F1 estimate vs test F1: Pearson r = {est.corr().iloc[0, 1]:.3f}, "
           f"mean abs error = {(est.f1_est - est.f1).abs().mean():.3f} (n = {len(est)})\n"]

    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
