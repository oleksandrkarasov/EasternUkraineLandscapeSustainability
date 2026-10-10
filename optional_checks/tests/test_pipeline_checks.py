#!/usr/bin/env python3
"""

    python tests/test_pipeline_checks.py

Each test reproduces a failure case named in the audit or checks a safeguard
added for it. All data are small SYNTHETIC tables written to a temporary
folder; no real output is read and no number here is a result.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # package root (tests moved to optional_checks/tests)
sys.path.insert(0, str(ROOT / "support"))
import pipeline_checks as pc  # noqa: E402


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# audit item 2: duplicate rows must not collapse
# ---------------------------------------------------------------------------

def test_multiset_keeps_duplicates():
    cv = ["A", "A", "B"]
    final = ["A", "B"]
    assert set(cv) == set(final)
    r = pc.multiset_compare(final, cv)
    assert not r["identical"] and r["n_unmatched_b"] == 1


def _rows(ids, presence=None, fp=None):
    n = len(ids)
    return pd.DataFrame({"row_id": ids,
                         "presence": presence if presence is not None else [1] * n,
                         "emb_fingerprint": fp if fp is not None else np.arange(n, dtype=float),
                         "lon": [30.0 + 0.001 * i for i in range(n)],
                         "lat": [48.0] * n, "run_id": "R1",
                         "presence_score_mode": "PROBABILITY", "class1_index": 1,
                         "spatial_fold": [i % 5 for i in range(n)],
                         "fold_id": [i % 5 for i in range(n)]})


def test_row_identity():
    man = _rows(["r1", "r2", "r3"])
    fin = man.copy()
    cv = man.copy()
    assert pc.row_identity_check(fin, man, cv)["passed"]
    # one final row missing
    assert not pc.row_identity_check(fin.iloc[:2], man, cv)["passed"]
    # same IDs, a label differs
    bad = fin.copy()
    bad.loc[0, "presence"] = 0
    assert not pc.row_identity_check(bad, man, cv)["passed"]
    # same IDs, predictor values differ
    bad = fin.copy()
    bad.loc[1, "emb_fingerprint"] += 0.5
    assert not pc.row_identity_check(bad, man, cv)["passed"]
    # a repeated ID is an error even when counts agree
    rep = _rows(["r1", "r1", "r3"])
    assert not pc.row_identity_check(rep, rep, rep)["passed"]
    # scores without row_id (older export): identity not established -> fail
    cv2 = cv.drop(columns="row_id")
    r = pc.row_identity_check(fin, man, cv2)
    assert not r["passed"] and r["cv_scores_method"].startswith("coordinate")
    # a mix of versions: some score rows without row_id -> fail
    cv3 = cv.copy()
    cv3.loc[0, "row_id"] = np.nan
    assert not pc.row_identity_check(fin, man, cv3)["passed"]
    # scores from another GEE_03 run than the manifest -> fail
    assert not pc.row_identity_check(fin, man, cv.assign(run_id="R2"))["passed"]


def test_cv_score_contents():
    """"""
    man = _rows(["r1", "r2", "r3", "r4", "r5"], presence=[1, 1, 0, 0, 1])
    fin = man.copy()
    cv = man.copy()
    assert pc.row_identity_check(fin, man, cv)["passed"]
    bad = cv.copy()
    bad.loc[0, "presence"] = 0                      # response label changed
    r = pc.row_identity_check(fin, man, bad)
    assert not r["passed"] and r["cv_n_label_mismatch"] == 1 and r["final_n_label_mismatch"] == 0
    bad = cv.copy()
    bad.loc[1, "emb_fingerprint"] = 999.0           # predictor values changed
    r = pc.row_identity_check(fin, man, bad)
    assert not r["passed"] and r["cv_n_fingerprint_mismatch"] == 1
    bad = cv.copy()
    bad.loc[2, "fold_id"] = 4                       # held out in the wrong fold
    r = pc.row_identity_check(fin, man, bad)
    assert not r["passed"] and r["cv_n_fold_mismatch"] == 1
    # every partition of a scheme: held-out rows + training rows
    test = cv.assign(set="test", analysis_type="main_fixed", model="embeddings",
                     code_version="v13b12")
    train = pd.concat([cv.assign(fold_id=f) for f in range(5)])
    train = train[train.spatial_fold != train.fold_id].assign(
        set="train", analysis_type="main_fixed", model="embeddings", code_version="v13b12")
    scores = pd.concat([test, train], ignore_index=True)
    res = pc.cv_scores_vs_manifest(scores, man)
    assert res.passed.all(), res.problems.tolist()
    leak = pd.concat([scores, test.assign(set="train")], ignore_index=True)   # trained on own fold
    assert not pc.cv_scores_vs_manifest(leak, man).passed.any()
    wrong = scores.copy()
    wrong.loc[wrong.set == "train", "presence"] = 1 - wrong.loc[wrong.set == "train", "presence"]
    assert not pc.cv_scores_vs_manifest(wrong, man).passed.any()
    mixed = scores.copy()
    mixed.loc[0, "presence_score_mode"] = "MULTIPROBABILITY_CLASS1"
    assert not pc.cv_scores_vs_manifest(mixed, man).passed.any()


# ---------------------------------------------------------------------------
# audit item 1: agreement with either column must not pass
# ---------------------------------------------------------------------------

def _mode(col_label1=1, prob_equals=1, both_groups=True):
    """Synthetic SDM_ProbabilityModeCheck row. prob_equals: the column that
    PROBABILITY equals in every row, or 'winner' (probability of the predicted
    class: equals the label-1 column only where the label is 1)."""
    d = {"presence_score_mode": "PROBABILITY", "configured_class1_index": 1}
    for s in ("train", "mapped"):
        d[f"{s}_n_clear"] = 100
        d[f"{s}_share_argmax_col{col_label1}_is_label1"] = 1.0
        d[f"{s}_share_argmax_col{1 - col_label1}_is_label1"] = 0.0
        for g in (1, 0):
            d[f"{s}_pred{g}_n"] = 50 if (both_groups or g == 1) else 0
            for c in (0, 1):
                if prob_equals == "winner":
                    label_col = col_label1 if g == 1 else 1 - col_label1
                    diff = 0.0 if c == label_col else 0.4
                else:
                    diff = 0.0 if c == prob_equals else 0.6
                d[f"{s}_pred{g}_max_abs_diff_col{c}"] = diff
    return d


def test_probability_verdict():
    assert pc.probability_mode_verdict(_mode(1, 1))["probability_output_is_presence_probability"]
    # the audit's false pass: d0 = 0, d1 = 0.8 with label-1 column 1
    v = pc.probability_mode_verdict(_mode(1, 0))
    assert not v["probability_output_is_presence_probability"]
    # probability of the predicted class: matches only where label is 1
    v = pc.probability_mode_verdict(_mode(1, "winner"))
    assert not v["probability_output_is_presence_probability"]
    # label-1 column is 0 and PROBABILITY equals it: passes, column reported
    v = pc.probability_mode_verdict(_mode(0, 0))
    assert v["probability_output_is_presence_probability"] and v["label1_column"] == 0
    # a predicted group is empty: cannot be established
    assert not pc.probability_mode_verdict(_mode(1, 1, both_groups=False))[
        "probability_output_is_presence_probability"]
    assert not pc.probability_mode_verdict(None)["probability_output_is_presence_probability"]
    # a few clear rows where the class vote and the averaged probability
    d = _mode(1, 1)
    d["train_share_argmax_col1_is_label1"], d["train_share_argmax_col0_is_label1"] = 0.9978, 0.0022
    assert pc.probability_mode_verdict(d)["probability_output_is_presence_probability"]
    # ... but an unclear assignment is still rejected
    d["train_share_argmax_col1_is_label1"], d["train_share_argmax_col0_is_label1"] = 0.95, 0.05
    assert not pc.probability_mode_verdict(d)["probability_output_is_presence_probability"]


# ---------------------------------------------------------------------------
# audit items 5 and 6: ratios
# ---------------------------------------------------------------------------

def test_ratios():
    r, s = pc.safe_ratio(5.0, 0.0)
    assert math.isnan(r) and s == "no_detections"
    r, s = pc.safe_ratio(5.0, float("nan"))
    assert math.isnan(r) and s == "missing"
    assert pc.safe_ratio(1.0, 4.0) == (0.25, "ok")
    j, s = pc.jaccard(0, 0, 0)
    assert math.isnan(j) and s == "both_empty"
    assert pc.jaccard(10, 10, 5)[0] == 5 / 15
    j, s = pc.jaccard(1170, 900, 1500)
    assert math.isnan(j) and s == "invalid_areas"
    assert pc.jaccard(-5, 900, 0)[1] == "invalid_areas"
    assert pc.jaccard(100, 900, -1)[1] == "invalid_areas"
    assert pc.jaccard(100, 100, 100.005)[1] == "ok"        # within the tolerance


def test_null_diagnostics_ratios_and_split():
    nd = load(ROOT / "support" / "Python_null_diagnostics.py", "nd")
    ex = pd.DataFrame({"comparison": ["year_2022"], "z_class": [2.5],
                       "observed_loss_km2": [0.0], "prewar_benchmark_loss_km2": [12.0],
                       "observed_gain_km2": [40.0], "prewar_benchmark_gain_km2": [10.0]})
    bt = nd.benchmark_table(ex)
    loss = bt[bt.direction == "loss"].iloc[0]
    assert math.isnan(loss.benchmark_ratio) and loss.status == "no_detections"
    assert bt[bt.direction == "gain"].benchmark_ratio.iloc[0] == 0.25
    # closing adds a cell: 24 exceedances, 20 kept, 1 added -> 21 in result
    sn = pd.DataFrame({"null_year": [2019, 2020, 2021], "z_threshold": 3.0,
                       "stricter_loss_km2": 21.0, "pixel_loss_km2": 24.0,
                       "stricter_loss_in_exceedance_km2": 20.0, "stricter_loss_added_km2": 1.0,
                       "pixel_loss_removed_km2": 4.0,
                       "stricter_gain_km2": 0.0, "pixel_gain_km2": 0.0,
                       "stricter_gain_in_exceedance_km2": 0.0, "stricter_gain_added_km2": 0.0,
                       "pixel_gain_removed_km2": 0.0})
    out = nd.stricter_prewar_summary(sn).iloc[0]
    assert np.isclose(out.postfilter_to_exceedance_area_ratio_loss, 21 / 24)
    assert np.isclose(out.retained_share_loss, 20 / 24)
    assert math.isnan(out.retained_share_gain) and out.status_gain == "no_exceedances"
    for broken in (sn.iloc[:2], sn.assign(stricter_loss_km2=[21.0, np.nan, 21.0])):
        try:
            nd.stricter_prewar_summary(broken)
            raise AssertionError("incomplete pre-war table accepted")
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# audit item 2: ambiguous files
# ---------------------------------------------------------------------------

def test_find_unique():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        (t / "a").mkdir()
        (t / "a" / "X_v13b12.csv").write_text("x\n1\n")
        p, st, _ = pc.find_unique([t], "X_v13b12.csv")
        assert st == "ok" and p.name == "X_v13b12.csv"
        (t / "X_v13b12 (1).csv").write_text("x\n1\n")
        p, st, c = pc.find_unique([t], "X_v13b12.csv")
        assert p is None and st == "ambiguous" and len(c) == 2
        assert pc.find_unique([t], "Y.csv")[1] == "missing"
    assert pc.version_rank("CV_x_fold0_v13b12.csv") > pc.version_rank("CV_x_fold0_v13b5.csv")


# ---------------------------------------------------------------------------
# audit items 3, 6, 7: Python_03 end to end on synthetic tables
# ---------------------------------------------------------------------------

STATS = ["median_loss", "median_gain", "year_2022_loss", "year_2022_gain",
         "year_2023_loss", "year_2023_gain", "year_2024_loss", "year_2024_gain",
         "dominant_loss_mapped", "dominant_gain_mapped", "dominant_loss_ge2_mapped",
         "dominant_gain_ge2_mapped", "union_loss_ge1_mapped", "union_loss_ge2_mapped",
         "union_gain_ge1_mapped", "union_gain_ge2_mapped", "overlap_loss_and_gain_mapped",
         "stricter_loss", "stricter_gain"]


def write_mock(base: Path, T: dict, stem: str):
    asset_run = "GEE04_v13b12_20261001T000000Z"
    for scheme, t in T.items():
        d = base / f"{stem}_{scheme}"
        d.mkdir(parents=True)
        rid = f"GEE05_v13b12_{scheme}_20261002T000000Z"

        def w(name, frame):
            frame = frame.copy()
            frame.insert(0, "analysis_scheme", scheme)
            frame.insert(0, "code_version", "v13b12")
            frame.insert(0, "run_id", rid)
            frame.to_csv(d / name, index=False)

        zc = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        obs = {"median_headline": (900.0, 5.0), "composite_sensitivity": (700.0, 3.0),
               "year_2022": (0.0, 3000.0), "year_2023": (800.0, 30.0), "year_2024": (2500.0, 0.0)}
        ex = [{"comparison": c, "z_sd_factor": 1, "z_class": z,
               "observed_loss_km2": lo * (t / z) if z != t else lo,
               "prewar_benchmark_loss_km2": 60.0 / z,
               "observed_gain_km2": ga * (t / z) if z != t else ga,
               "prewar_benchmark_gain_km2": 300.0 / z,
               "eligible_ground_area_km2": 150000.0}
              for c, (lo, ga) in obs.items() for z in zc]
        w("Areas_Excess_by_Zclass.csv", pd.DataFrame(ex))
        dom = {1: 2000.0, 2: 500.0, 3: 50.0, -1: 2800.0, -2: 4.0, -3: 0.0}
        un = {"union_loss_ge1": 2600.0, "union_loss_ge2": 560.0, "union_gain_ge1": 2900.0,
              "union_gain_ge2": 4.0, "overlap_loss_and_gain": 90.0}
        cum = [{"variant": v, "set": "dominant_class", "class": k, "observed_km2": a}
               for v in ("pixel", "mapped") for k, a in dom.items()]
        cum += [{"variant": v, "set": s, "class": np.nan, "observed_km2": a}
                for v in ("pixel", "mapped") for s, a in un.items()]
        w("Cumulative_Confidence_Class_Areas.csv", pd.DataFrame(cum))
        head = pd.DataFrame([
            {"variant": "primary_all_valid_input_pixels", "damage_km2": 400.0, "rewild_km2": 2.0,
             "damage_exceedance_km2": 520.0, "damage_in_exceedance_km2": 380.0,
             "damage_added_by_closing_km2": 20.0, "damage_exceedance_removed_km2": 140.0,
             "rewild_exceedance_km2": 6.0, "rewild_in_exceedance_km2": 2.0,
             "rewild_added_by_closing_km2": 0.0, "rewild_exceedance_removed_km2": 4.0},
            {"variant": "composite_representation_sensitivity", "damage_km2": 300.0, "rewild_km2": 1.0},
            {"variant": "quality_screen_with_novelty_sensitivity", "damage_km2": 280.0, "rewild_km2": 1.0},
            {"variant": "quality_screen_without_novelty_sensitivity", "damage_km2": 290.0, "rewild_km2": 1.0},
            {"variant": "stricter_no_closing", "damage_km2": 370.0, "rewild_km2": 2.0}])
        w("Areas_Headline_annualbase_annual_median.csv", head)
        w("Debiased_RF_Diagnostics_revision_annualbase_annual_median.csv", pd.DataFrame(
            {"period": ["median", "composite", "2022", "2023", "2024", "null_2019", "null_2020",
                        "null_2021", "null_pooled_sample"], "z_cal_mean": [0.1] * 9}))
        sn = pd.DataFrame({"null_year": [2019, 2020, 2021], "z_threshold": 3.0,
                           "stricter_loss_km2": 30.0, "stricter_gain_km2": 40.0,
                           "stricter_loss_screened_km2": 20.0, "stricter_gain_screened_km2": 30.0,
                           "pixel_loss_km2": 60.0, "pixel_gain_km2": 70.0,
                           "stricter_loss_in_exceedance_km2": 28.0,
                           "stricter_gain_in_exceedance_km2": 39.0,
                           "stricter_loss_added_km2": 2.0, "stricter_gain_added_km2": 1.0,
                           "pixel_loss_removed_km2": 32.0, "pixel_gain_removed_km2": 31.0,
                           "stricter_no_closing_loss_km2": 27.0, "stricter_no_closing_gain_km2": 38.0})
        w("Areas_Stricter_Null_v13b12.csv", sn)
        sc = ["domain", "median_loss", "median_gain", "year_2022_loss", "year_2022_gain",
              "year_2023_loss", "year_2023_gain", "year_2024_loss", "year_2024_gain",
              "class_loss_mapped", "class_gain_mapped", "class_loss2_mapped",
              "union_loss_mapped", "union_gain_mapped", "stricter_loss", "stricter_gain"]
        w("Areas_by_Screen_v13b12.csv", pd.DataFrame(
            {"statistic": sc, "all_pixels_km2": 10.0, "wartime_sd_km2": 7.0,
             "interior_propensity_km2": 8.0, "no_annual_extrapolation_km2": 9.0,
             "all_three_screens_km2": 5.0, "at_sd_floor_km2": 1.0}))
        w("SD_Floor_Diagnostics_v13b12.csv", pd.DataFrame({"share_pixels_at_floor": [0.01]}))
        grid = [{"z_threshold": z, "min_patch_px": p, "is_main_parameters": int(z == 3.0 and p == 8),
                 "damage_km2": 400.0 * (3.0 / z) ** 2 * (8 / p) ** 0.5,
                 "rewild_km2": 2.0 * (3.0 / z) ** 2 * (8 / p) ** 0.5}
                for z in (2.5, 3.0, 3.5, 4.0) for p in (4, 8, 16)]
        w("Areas_vs_Threshold_and_PatchSize.csv", pd.DataFrame(grid))
        w("GEE05_RunManifest_v13b12.csv", pd.DataFrame([{
            "period_probability_run_id": asset_run, "annual_probability_run_id": asset_run,
            "novelty_run_id": asset_run, "period_probability_code_version": "v13b12",
            "period_probability_presence_score_mode": "PROBABILITY"}]))
        main = {"median_loss": 900.0, "median_gain": 5.0, "year_2022_loss": 0.0,
                "year_2022_gain": 3000.0, "year_2023_loss": 800.0, "year_2023_gain": 30.0,
                "year_2024_loss": 2500.0, "year_2024_gain": 0.0,
                "dominant_loss_mapped": 2550.0, "dominant_gain_mapped": 2804.0,
                "dominant_loss_ge2_mapped": 550.0, "dominant_gain_ge2_mapped": 4.0,
                "union_loss_ge1_mapped": 2600.0, "union_loss_ge2_mapped": 560.0,
                "union_gain_ge1_mapped": 2900.0, "union_gain_ge2_mapped": 4.0,
                "overlap_loss_and_gain_mapped": 90.0, "stricter_loss": 400.0,
                "stricter_gain": 2.0}
        for name, fam, var, designs, scale in (
                ("main_main", "main", "main", ["main"], 1.0),
                ("bias_sigma_50000m", "bias_sigma", "50000m", ["three_years"], 1.3),
                ("bias_sigma_200000m", "bias_sigma", "200000m", ["three_years"], 0.8),
                ("dropped_year_2019", "dropped_year", "without_2019",
                 ["variance_only", "baseline_and_variance"], 0.7),
                ("dropped_year_2020", "dropped_year", "without_2020",
                 ["variance_only", "baseline_and_variance"], 1.1),
                ("dropped_year_2021", "dropped_year", "without_2021",
                 ["variance_only", "baseline_and_variance"], 0.9),
                ("sd_floor_0p01", "sd_floor", "0.01", ["three_years"], 1.05),
                ("sd_floor_0p04", "sd_floor", "0.04", ["three_years"], 0.9)):
            rows = [{"family": fam, "variant": var, "design": dz, "statistic": st,
                     "variant_km2": main[st] * scale, "main_km2": main[st],
                     "overlap_with_main_km2": main[st] * min(scale, 1.0) * 0.9 if scale != 1.0
                     else main[st]}
                    for dz in designs for st in STATS]
            w(f"Sensitivity_{name}_v13b12.csv", pd.DataFrame(rows))
        g4 = base / "RF_revision_v13b_direct_embeddings"
        g4.mkdir(exist_ok=True)
        cvm = base / "RF_revision_v13b_metrics"
        cvm.mkdir(exist_ok=True)
        fv = cvm / "final_vs_cv_training_rows.csv"
        prev = pd.read_csv(fv) if fv.exists() else pd.DataFrame()
        pd.concat([prev, pd.DataFrame([{"sampling_scheme": scheme, "passed": True,
                                        "gee04_run_id": asset_run,
                                        "presence_score_mode": "PROBABILITY",
                                        "problems": ""}])]).to_csv(fv, index=False)
        pd.DataFrame([{"run_id": asset_run, "code_version": "v13b12",
                       "analysis_scheme": scheme}]).to_csv(
            g4 / f"GEE04_RunManifest_{scheme}_v13b12.csv", index=False)
    write_mock_verdict(base / "RF_revision_v13b_metrics", asset_run)


def write_mock_verdict(cvm: Path, asset_run: str, passed: bool = True):
    """Synthetic Python_01 outputs: partition table and overall verdict."""
    p1 = load(ROOT / "Python_01_calculate_cv_metrics.py", "p1v")
    schemes = ["primary_effort", "access_matched", "recurrent_effort", "recurrent_access_matched"]
    pd.DataFrame([{"sampling_scheme": s, "analysis_type": "main_fixed", "model": "embeddings",
                   "run_id": "GEE03_X", "passed": passed, "problems": ""} for s in schemes]
                 ).to_csv(cvm / "cv_scores_vs_manifest.csv", index=False)
    final = pd.read_csv(cvm / "final_vs_cv_training_rows.csv")
    p1.write_verdict(cvm, cvm, final, pd.read_csv(cvm / "cv_scores_vs_manifest.csv"),
                     [] if passed else ["synthetic failure"])


def _run_p03(base: Path):
    p3 = load(ROOT / "Python_03_collect_manuscript_numbers.py", "p3")
    p3.BASE_DIR = base
    p3.SCHEME_DIRS = {s: [base / f"{p3.FOLDER_STEM}_{s}"] for s in p3.THRESHOLD}
    p3.CV_METRICS_DIR = base / "RF_revision_v13b_metrics"
    for k in ("OUT_OK", "OUT_FAILED", "OUT_CHECKS", "OUT_MANIFEST"):
        setattr(p3, k, base / Path(getattr(p3, k)).name)
    code = p3.main()
    checks = pd.read_csv(base / Path(p3.OUT_CHECKS).name)
    return p3, code, checks


def test_python03_end_to_end():
    stem = "debiased_rf_revision_v13b_direct_embeddings"
    T = {"primary_effort": 2.5, "recurrent_effort": 3.0}
    with tempfile.TemporaryDirectory() as t:
        base = Path(t)
        write_mock(base, T, stem)
        p3, code, checks = _run_p03(base)
        assert code == 0, checks[checks.status == "fail"].to_string()
        nums = pd.read_csv(p3.OUT_OK)
        r = nums[nums.item.str.contains("2022, loss benchmark / detected")]
        assert r.status.iloc[0] == "no_detections"      # zero detected area -> NA + status
        assert nums.new.notna().sum() > 100

        # a missing value must fail, never count as zero
        d = base / f"{stem}_primary_effort"
        ex = pd.read_csv(d / "Areas_Excess_by_Zclass.csv")
        ex.loc[(ex.comparison == "year_2024") & (ex.z_class == 2.5), "observed_gain_km2"] = np.nan
        ex.to_csv(d / "Areas_Excess_by_Zclass.csv", index=False)
        _, code, checks = _run_p03(base)
        assert code == 1 and not (base / Path(p3.OUT_OK).name).exists()
        assert (base / Path(p3.OUT_FAILED).name).exists()

    with tempfile.TemporaryDirectory() as t:
        base = Path(t)
        write_mock(base, T, stem)
        # main-parameter sensitivity does not reproduce the headline (gain)
        d = base / f"{stem}_recurrent_effort"
        s = pd.read_csv(d / "Sensitivity_main_main_v13b12.csv")
        s.loc[s.statistic == "union_gain_ge1_mapped", ["variant_km2", "main_km2",
                                                       "overlap_with_main_km2"]] = 2950.0
        s.to_csv(d / "Sensitivity_main_main_v13b12.csv", index=False)
        _, code, checks = _run_p03(base)
        fails = checks[checks.status == "fail"].check.tolist()
        assert code == 1 and any("reproduces headline: union_gain_ge1_mapped" in f for f in fails)

    # 0.076 km2); more than one cell must still fail
    for shift, ok in ((0.08, True), (0.5, False)):
        with tempfile.TemporaryDirectory() as t:
            base = Path(t)
            write_mock(base, T, stem)
            d = base / f"{stem}_primary_effort"
            s = pd.read_csv(d / "Sensitivity_main_main_v13b12.csv")
            cols = ["variant_km2", "main_km2", "overlap_with_main_km2"]
            s.loc[s.statistic == "year_2024_loss", cols] += shift
            s.to_csv(d / "Sensitivity_main_main_v13b12.csv", index=False)
            for f in d.glob("Sensitivity_*_v13b12.csv"):      # other tables repeat main_km2
                if f.name == "Sensitivity_main_main_v13b12.csv":
                    continue
                o = pd.read_csv(f)
                o.loc[o.statistic == "year_2024_loss", "main_km2"] += shift
                o.to_csv(f, index=False)
            _, code, checks = _run_p03(base)
            fails = checks[checks.status == "fail"].check.tolist()
            assert (code == 0) == ok, fails
            assert any("reproduces headline: year_2024_loss" in f for f in fails) != ok

    for label, name, cut in (
            ("dropped-year table reduced to one row", "Sensitivity_dropped_year_2019_v13b12.csv",
             lambda f: f[(f.design == "variance_only") & (f.statistic == "median_loss")]),
            ("one design missing", "Sensitivity_dropped_year_2020_v13b12.csv",
             lambda f: f[f.design == "variance_only"]),
            ("pre-war year 2021 removed", "Areas_Stricter_Null_v13b12.csv",
             lambda f: f[f.null_year != 2021]),
            ("a z class removed", "Areas_Excess_by_Zclass.csv",
             lambda f: f[~((f.comparison == "year_2023") & (f.z_class == 4.0))]),
            ("a union row removed", "Cumulative_Confidence_Class_Areas.csv",
             lambda f: f[~((f.variant == "mapped") & (f["set"] == "union_gain_ge2"))])):
        with tempfile.TemporaryDirectory() as t:
            base = Path(t)
            write_mock(base, T, stem)
            path = base / f"{stem}_primary_effort" / name
            cut(pd.read_csv(path)).to_csv(path, index=False)
            _, code, checks = _run_p03(base)
            fails = checks[checks.status == "fail"].check.tolist()
            assert code == 1 and any("expected rows" in f for f in fails), (label, fails)

    with tempfile.TemporaryDirectory() as t:
        base = Path(t)
        write_mock(base, T, stem)
        path = base / f"{stem}_primary_effort" / "Sensitivity_bias_sigma_50000m_v13b12.csv"
        f = pd.read_csv(path)
        f.loc[f.statistic == "median_loss", "overlap_with_main_km2"] = 1500.0
        f.to_csv(path, index=False)
        _, code, checks = _run_p03(base)
        fails = checks[checks.status == "fail"].check.tolist()
        assert code == 1 and any("0 <= overlap <= min" in f for f in fails), fails

    # a decomposition that does not add up / a subset larger than its set
    with tempfile.TemporaryDirectory() as t:
        base = Path(t)
        write_mock(base, T, stem)
        path = base / f"{stem}_recurrent_effort" / "Areas_Headline_annualbase_annual_median.csv"
        f = pd.read_csv(path)
        f.loc[f.variant == "quality_screen_with_novelty_sensitivity", "damage_km2"] = 999.0
        f.to_csv(path, index=False)
        _, code, checks = _run_p03(base)
        assert code == 1 and any("all three screens <= two screens" in c for c in
                                 checks[checks.status == "fail"].check)

    with tempfile.TemporaryDirectory() as t:
        base = Path(t)
        write_mock(base, T, stem)
        # a Python_01 check table edited after its verdict
        fv = base / "RF_revision_v13b_metrics" / "final_vs_cv_training_rows.csv"
        v = pd.read_csv(fv)
        v.loc[v.sampling_scheme == "recurrent_effort", "passed"] = False
        v.to_csv(fv, index=False)
        _, code, checks = _run_p03(base)
        assert code == 1 and "Python_01: final_vs_cv_training_rows.csv unchanged since the verdict" in \
            checks[checks.status == "fail"].check.tolist()

    for label, damage in (("verdict failed", lambda cvm, run: write_mock_verdict(cvm, run, False)),
                          ("stale CHECKS_FAILED.txt", lambda cvm, run: (cvm / "CHECKS_FAILED.txt")
                           .write_text("Python_01 started and has not finished\n")),
                          ("verdict missing", lambda cvm, run: (cvm / "CV_CHECKS_VERDICT_v13b12.json")
                           .unlink())):
        with tempfile.TemporaryDirectory() as t:
            base = Path(t)
            write_mock(base, T, stem)
            damage(base / "RF_revision_v13b_metrics", "GEE04_v13b12_20261001T000000Z")
            _, code, checks = _run_p03(base)
            assert code == 1, label

    with tempfile.TemporaryDirectory() as t:
        base = Path(t)
        write_mock(base, T, stem)
        # one table from another run
        d = base / f"{stem}_primary_effort"
        g = pd.read_csv(d / "Areas_vs_Threshold_and_PatchSize.csv")
        g["run_id"] = "GEE05_v13b12_primary_effort_OTHER"
        g.to_csv(d / "Areas_vs_Threshold_and_PatchSize.csv", index=False)
        _, code, checks = _run_p03(base)
        assert code == 1 and "all tables carry the manifest run_id" in \
            checks[checks.status == "fail"].check.tolist()


def test_python01_row_check_files():
    p1 = load(ROOT / "Python_01_calculate_cv_metrics.py", "p1")
    with tempfile.TemporaryDirectory() as t:
        base = Path(t)
        raws = []
        for scheme in ("primary_effort", "recurrent_effort"):
            m = _rows(["a", "b", "c", "d"], presence=[1, 1, 0, 0])
            m.to_csv(base / f"CV_RowManifest_{scheme}_v13b12.csv", index=False)
            m.to_csv(base / f"SDM_FullModelPointPredictions_{scheme}_v13b12.csv", index=False)
            pd.DataFrame([dict(_mode(1, 1), run_id="R1")]).to_csv(
                base / f"SDM_ProbabilityModeCheck_{scheme}_v13b12.csv", index=False)
            pd.DataFrame([{"n_rows_current_rule": 4, "run_id": "R1"}]).to_csv(
                base / f"SDM_TrainingRowCheck_{scheme}_v13b12.csv", index=False)
            raws.append(m.assign(analysis_type="main_fixed", model="embeddings", set="test",
                                 sampling_scheme=scheme))
        raw = pd.concat(raws, ignore_index=True)
        out = p1.final_training_row_check(base, raw)
        assert out.passed.all(), out.problems.tolist()
        # a Drive duplicate of one input makes the check fail, not choose
        (base / "SDM_ProbabilityModeCheck_primary_effort_v13b12 (1).csv").write_text(
            (base / "SDM_ProbabilityModeCheck_primary_effort_v13b12.csv").read_text())
        out = p1.final_training_row_check(base, raw)
        assert not out.set_index("sampling_scheme").loc["primary_effort", "passed"]
        # the audit's [A, A, B] vs [A, B] case on the CV side
        (base / "SDM_ProbabilityModeCheck_primary_effort_v13b12 (1).csv").unlink()
        raw2 = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
        out = p1.final_training_row_check(base, raw2)
        assert not out.set_index("sampling_scheme").loc["primary_effort", "passed"]
        # CV scores made with another score mode than the verified one
        raw3 = raw.assign(presence_score_mode="MULTIPROBABILITY_CLASS1")
        out = p1.final_training_row_check(base, raw3)
        assert not out.passed.any()
        # the probability check from a different GEE_04 run
        pd.DataFrame([dict(_mode(1, 1), run_id="R9")]).to_csv(
            base / "SDM_ProbabilityModeCheck_recurrent_effort_v13b12.csv", index=False)
        out = p1.final_training_row_check(base, raw)
        assert not out.set_index("sampling_scheme").loc["recurrent_effort", "passed"]


def test_python03_several_gee04_manifests():
    """Several downloads of the GEE_04 folder: identical copies of the run
    used by GEE_05 pass, copies of other runs are ignored, differing copies
    of the used run fail, and no copy of the used run fails."""
    stem = "debiased_rf_revision_v13b_direct_embeddings"
    T = {"primary_effort": 2.5, "recurrent_effort": 3.0}
    run = "GEE04_v13b12_20261001T000000Z"
    for case in ("identical_copies", "other_run_ignored", "differing_copy", "no_copy_of_run"):
        with tempfile.TemporaryDirectory() as t:
            base = Path(t)
            write_mock(base, T, stem)
            src = base / "RF_revision_v13b_direct_embeddings" / "GEE04_RunManifest_primary_effort_v13b12.csv"
            for k in (1, 2):
                d = base / f"download_{k}" / "RF_revision_direct_embeddings"
                d.mkdir(parents=True)
                (d / src.name).write_bytes(src.read_bytes())
            f2 = base / "download_2" / "RF_revision_direct_embeddings" / src.name
            if case == "other_run_ignored":
                pd.DataFrame([{"run_id": "GEE04_v13b12_OLD", "code_version": "v13b12"}]).to_csv(f2, index=False)
            if case == "differing_copy":
                pd.DataFrame([{"run_id": run, "code_version": "v13b12", "extra": 1}]).to_csv(f2, index=False)
            if case == "no_copy_of_run":
                for pth in base.rglob(src.name):
                    pd.DataFrame([{"run_id": "GEE04_v13b12_OLD"}]).to_csv(pth, index=False)
            _, code, checks = _run_p03(base)
            fails = checks[checks.status == "fail"].check.tolist()
            if case in ("identical_copies", "other_run_ignored"):
                assert code == 0, (case, fails)
            else:
                assert code == 1 and any("GEE04_RunManifest_primary_effort" in f for f in fails), (case, fails)


def make_cv_inputs(d: Path, gee04_run: str, n: int = 200, seed: int = 1):
    """"""
    rng = np.random.default_rng(seed)
    tags = {"primary_effort": "nondvi_outdoor_effort_guard0_matched2019_2021",
            "access_matched": "nondvi_outdoor_accessmatch_guard0_matched2019_2021",
            "recurrent_effort": "recurrent_core_nondvi_effort_guard0_matched2019_2021",
            "recurrent_access_matched": "recurrent_core_nondvi_accessmatch_guard0_matched2019_2021"}
    mtry = {"embeddings": 16, "baseline_simple": 3, "ndvi_null": 1}
    for scheme, tag in tags.items():
        pres = np.r_[np.ones(n // 2, int), np.zeros(n // 2, int)]
        lon, lat = rng.uniform(34, 40, n), rng.uniform(46, 51, n)
        man = pd.DataFrame({"run_id": "GEE03_v13b12_T", "code_version": "v13b12",
                            "sampling_scheme": scheme,
                            "row_id": [f"{scheme[:4]}{i:05d}_0" for i in range(n)],
                            "emb_fingerprint": rng.normal(0, 10, n), "presence": pres,
                            "match_pair_id": np.repeat(np.arange(n // 2), 2) if "access" in scheme
                            else np.nan,
                            "spatial_fold": np.floor((lon - 34) / 1.2).astype(int) % 5,
                            "lon": lon, "lat": lat, "presence_score_mode": "PROBABILITY",
                            "class1_index": 1})
        man.to_csv(d / f"CV_RowManifest_{scheme}_v13b12.csv", index=False)
        models = list(mtry) if scheme in ("primary_effort", "recurrent_effort") else ["embeddings"]
        for m in models:
            for f in range(5):
                parts = []
                for st, rows in (("test", man[man.spatial_fold == f]),
                                 ("train", man[man.spatial_fold != f].sample(frac=0.5, random_state=f))):
                    x = rows.drop(columns=["spatial_fold", "sampling_scheme"]).copy()
                    x = x.assign(analysis_type="main_fixed", pipeline_version="v13b", model=m,
                                 sampling_scheme=scheme, run_tag=tag, repeat_id="det",
                                 block_size=0.45 if "recurrent" in scheme else 0.3, min_leaf=20,
                                 variables_per_split=mtry[m], fold_id=f, set=st,
                                 prob=np.clip(0.5 + 0.25 * (2 * rows.presence.to_numpy() - 1)
                                              + rng.normal(0, 0.2, len(rows)), 0.01, 0.99),
                                 ndvi=rng.uniform(0.2, 0.8, len(rows)), esa=10,
                                 dist_infra_m=rng.uniform(10, 3000, len(rows)),
                                 log_effort=rng.uniform(0, 6, len(rows)),
                                 in_ordlo=(rows.lon.to_numpy() > 38).astype(int), platform="flickr")
                    parts.append(x)
                out = pd.concat(parts)
                out.to_csv(d / f"CV_MainBlocked_{m}_{scheme}_leaf20_mtry{mtry[m]}_fold{f}_v13b12.csv",
                           index=False)
                if m == "embeddings" and scheme in ("primary_effort", "recurrent_effort"):
                    out.assign(analysis_type="random_cv", repeat_id="random").to_csv(
                        d / f"CV_RandomFold_{m}_{scheme}_leaf20_mtry16_fold{f}_v13b12.csv", index=False)
        if scheme in ("primary_effort", "recurrent_effort"):
            man.assign(run_id=gee04_run, fitted_probability=0.5).to_csv(
                d / f"SDM_FullModelPointPredictions_{scheme}_v13b12.csv", index=False)
            pd.DataFrame([dict(_mode(1, 1), run_id=gee04_run)]).to_csv(
                d / f"SDM_ProbabilityModeCheck_{scheme}_v13b12.csv", index=False)
            pd.DataFrame([{"n_rows_current_rule": n, "run_id": gee04_run}]).to_csv(
                d / f"SDM_TrainingRowCheck_{scheme}_v13b12.csv", index=False)


def test_python01_to_python03_integration():
    """"""
    import subprocess
    stem = "debiased_rf_revision_v13b_direct_embeddings"
    T = {"primary_effort": 2.5, "recurrent_effort": 3.0}
    run = "GEE04_v13b12_20261001T000000Z"
    for corrupt in (False, True):
        with tempfile.TemporaryDirectory() as t:
            base = Path(t)
            cvin = base / "RF_v13b12"
            cvin.mkdir()
            make_cv_inputs(cvin, run)
            if corrupt:        # one changed response label in a baseline-model file
                f = cvin / "CV_MainBlocked_baseline_simple_recurrent_effort_leaf20_mtry3_fold2_v13b12.csv"
                x = pd.read_csv(f)
                i = x.index[x.set == "test"][0]
                x.loc[i, "presence"] = 1 - x.loc[i, "presence"]
                x.to_csv(f, index=False)
            write_mock(base, T, stem)                       # GEE_05 tables + a passing verdict
            cvm = base / "RF_revision_v13b_metrics"
            for p in cvm.iterdir():
                p.unlink()                                  # replaced by the real Python_01 run
            r = subprocess.run([sys.executable, str(ROOT / "Python_01_calculate_cv_metrics.py"),
                                "--input-dir", str(cvin), "--output-dir", str(cvm)],
                               capture_output=True, text=True, timeout=600)
            fv = pd.read_csv(cvm / "final_vs_cv_training_rows.csv")
            assert fv.passed.all()                          # the final-model rows pass either way
            _, code, checks = _run_p03(base)
            fails = checks[checks.status == "fail"].check.tolist()
            if corrupt:
                assert r.returncode == 1 and (cvm / "CHECKS_FAILED.txt").exists()
                assert code == 1 and "Python_01: overall verdict passed" in fails, fails
            else:
                assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
                assert not (cvm / "CHECKS_FAILED.txt").exists()
                assert code == 0, fails
                # a new run that stops early voids the verdict and leaves the marker
                r2 = subprocess.run([sys.executable, str(ROOT / "Python_01_calculate_cv_metrics.py"),
                                     "--input-dir", str(base / "empty"), "--output-dir", str(cvm)],
                                    capture_output=True, text=True, timeout=600)
                assert r2.returncode != 0
                assert not (cvm / "CV_CHECKS_VERDICT_v13b12.json").exists()
                assert (cvm / "CHECKS_FAILED.txt").exists()
                _, code, _ = _run_p03(base)
                assert code == 1


def test_python01_separate_gee04_folder():
    """GEE_03 and GEE_04 downloads in their own folders (support/folders.py);
    the GEE_04 Drive zip may hold an inner folder."""
    import shutil
    import subprocess
    with tempfile.TemporaryDirectory() as t:
        base = Path(t)
        cv3, cv4 = base / "GEE03_cross_validation", base / "GEE04_final_models" / "GEE04_final_models"
        cv3.mkdir()
        cv4.mkdir(parents=True)
        make_cv_inputs(cv3, "GEE04_v13b12_20261001T000000Z")
        for f in cv3.glob("SDM_*.csv"):
            shutil.move(str(f), str(cv4 / f.name))
        out = base / "Python01_cv_metrics"
        r = subprocess.run([sys.executable, str(ROOT / "Python_01_calculate_cv_metrics.py"),
                            "--input-dir", str(cv3), "--gee04-dir", str(cv4.parent),
                            "--output-dir", str(out)], capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        assert pd.read_csv(out / "final_vs_cv_training_rows.csv").passed.all()
    import importlib
    sys.path.insert(0, str(ROOT / "support"))
    folders = importlib.import_module("folders")
    with tempfile.TemporaryDirectory() as t:
        d = Path(t) / "GEE05_regular"
        (d / "GEE05_regular").mkdir(parents=True)
        (d / "GEE05_regular" / "x.csv").write_text("a\n1\n")
        assert folders.data_dir(d) == d / "GEE05_regular"
        (d / "y.csv").write_text("a\n1\n")
        assert folders.data_dir(d) == d


def test_upload_names_match_gee02():
    """The ZIP files of Python_A1-A3 carry the names of the assets that GEE_02
    reads (ASSET_PREFIX + name), and every GEE script uses the same prefix."""
    import re
    sys.path.insert(0, str(ROOT / "support"))
    import folders
    js = (ROOT / "GEE" / "GEE_02_sample_all_training_tables.js").read_text(encoding="utf-8")
    read = set(re.findall(r"OWN \+ '(points_[a-z_]+)'", js))
    a1 = load(ROOT / "Python_A1_prepare_presences_and_backgrounds.py", "a1")
    written = set(a1.UPLOADED_LAYERS)
    for f in ("Python_A2_match_recurrent_accessibility.py", "Python_A3_match_nonrecurrent_accessibility.py"):
        m = load(ROOT / f, f[:9])
        written |= {m.OUTPUT_PRESENCE_BASENAME, m.OUTPUT_BASENAME}
    assert read == written and len(read) == 10, (read ^ written)
    for f in list((ROOT / "GEE").glob("GEE_0*.js")) + list((ROOT / "optional_checks").glob("GEE_0*.js")):
        prefixes = set(re.findall(r"var ASSET_PREFIX = '([^']*)'", f.read_text(encoding="utf-8")))
        assert prefixes == {folders.ASSET_PREFIX}, (f.name, prefixes)
    assert set(read) <= set(folders.TABLES.values())
    import geopandas as gpd
    from shapely.geometry import Point
    with tempfile.TemporaryDirectory() as t:
        a1.work_dir, a1.UPLOAD_DIR = t, str(Path(t) / "up")
        g = gpd.GeoDataFrame({"presence": [0], "in_ordlo": [0]}, geometry=[Point(37, 48)], crs="EPSG:4326")
        a1.export_pair(g, str(Path(t) / "a.gpkg"), a1.GEE_EXPORT_NAMES["pa_nondvi_matched"])
        a1.export_pair(g, str(Path(t) / "b.gpkg"), a1.GEE_EXPORT_NAMES["pa_ndvi050_matched"])
        assert [p.name for p in (Path(t) / "up").iterdir()] == [
            folders.ASSET_PREFIX + a1.GEE_EXPORT_NAMES["pa_nondvi_matched"] + ".zip"]


def test_cell_close():
    assert pc.cell_close(6479.889215686275, 6479.96568627451)
    assert pc.cell_close(100.0, 100.25)
    assert not pc.cell_close(100.0, 100.5)
    assert not pc.cell_close(np.nan, 1.0)
    assert not pc.area_close(6479.889215686275, 6479.96568627451)


def test_stale_support_modules_replaced():
    """A console that ran another copy of the package (e.g. %run of v49) keeps
    its pipeline_checks and folders in memory; the scripts must load their own."""
    import types
    saved = {m: sys.modules.get(m) for m in ("pipeline_checks", "folders")}
    try:
        for m in saved:
            sys.modules[m] = types.ModuleType(m)       # stale module without the names
        p3 = load(ROOT / "Python_03_collect_manuscript_numbers.py", "p3_stale")
        assert Path(sys.modules["pipeline_checks"].__file__).parent == ROOT / "support"
        assert callable(p3.cell_close)
    finally:
        for m, mod in saved.items():
            if mod is None:
                sys.modules.pop(m, None)
            else:
                sys.modules[m] = mod


def test_python01_conflicting_duplicates():
    """"""
    p1 = load(ROOT / "Python_01_calculate_cv_metrics.py", "p1b")
    base_rows = pd.DataFrame({
        "analysis_type": "main_fixed", "model": "embeddings", "sampling_scheme": "primary_effort",
        "run_tag": "nondvi_outdoor_effort_guard0_matched2019_2021", "repeat_id": "det",
        "block_size": 0.3, "min_leaf": 20, "variables_per_split": 16, "fold_id": 0,
        "set": "test", "presence": [0, 1], "prob": [0.1, 0.9], "row_id": ["a", "b"],
        "run_id": "R1", "code_version": "v13b12"})
    name = "CV_MainBlocked_embeddings_primary_effort_leaf20_mtry16_fold0_v13b12"
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        base_rows.to_csv(t / f"{name}.csv", index=False)
        base_rows.to_csv(t / f"{name} (1).csv", index=False)
        raw, inv = p1.load_raw_scores(t)
        assert len(raw) == 2 and (inv.status == "identical_copies").any()
        base_rows.assign(prob=[0.9, 0.1]).to_csv(t / f"{name} (1).csv", index=False)
        try:
            p1.load_raw_scores(t)
            raise AssertionError("conflicting copies accepted")
        except RuntimeError as err:
            assert "Conflicting CV score files" in str(err)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"passed  {fn.__name__}")
    print(f"All {len(tests)} tests passed.")
