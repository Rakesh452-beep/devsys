"""
Synthetic chip-test dataset for the AI-Optimized Yield Advisor.

Generates wafer sort (CP), retest, final test (FT) and burn-in data for a fictional
product, joined by MES genealogy and WAT, with known yield problems injected so that
each AI engine in the design document has a ground truth to recover.

Usage:  python generate.py [--lots 40] [--seed 2026] [--out data]
"""
import argparse, os, json, hashlib
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--lots", type=int, default=40)
ap.add_argument("--wafers", type=int, default=25)
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
OUT = a.out; GT = os.path.join(OUT, "ground_truth"); os.makedirs(GT, exist_ok=True)

PRODUCT, FAB = "AX7-SOC", "FAB2"
GRID = 26                                    # die grid across the wafer diameter
R_EDGE = GRID / 2 - 0.4
ys, xs = np.mgrid[0:GRID, 0:GRID]
cx = cy = (GRID - 1) / 2
rr = np.hypot(xs - cx, ys - cy) / R_EDGE      # normalised radius 0..1
valid = rr <= 1.0
DX, DY, R = xs[valid], ys[valid], rr[valid]
ANG = np.arctan2(DY - cy, DX - cx)
NDIE = valid.sum()
SITE = (DX % 2) + 2 * (DY % 2) + 1            # 4-site parallel probing

# ---------------- test program ----------------
LIMITS = pd.DataFrame([
    # order = execution order on the tester (stop on first fail)
    ("T1000", "CONTACT_RES", "ohm", None, 2.0, 30, 5, 18),
    ("T2100", "IDDQ", "uA", 40.0, 65.0, 12, 3, 42),
    ("T2200", "LEAKAGE", "nA", None, 60.0, 21, 3, 35),
    ("T3000", "SCAN_FAIL_COUNT", "count", None, 0, 3, 4, 310),
    ("T3500", "MBIST_FAIL", "flag", None, 0, 9, 4, 260),
    ("T4000", "PLL_LOCK", "us", None, 25.0, 15, 6, 28),
    ("T5000", "VMIN", "V", None, 0.68, 7, 2, 190),
    ("T5100", "FMAX", "MHz", 2250.0, None, 5, 2, 230),
    ("T4120", "RING_OSC", "MHz", 1120.0, None, 6, 2, 22),
    ("T6000", "ADC_INL", "LSB", None, 1.5, 18, 6, 95),
], columns=["test_num", "test_name", "unit", "lo_limit", "hi_limit", "soft_bin", "hard_bin", "test_time_ms"])
LIMITS["program_rev"] = "AX7_CP1_R4.2"
SBIN_NAME = {1: "PASS", 30: "CONTACT", 12: "IDDQ", 21: "LEAKAGE", 3: "SCAN", 9: "MBIST", 15: "PLL_LOCK",
             6: "RING_OSC", 7: "VMIN", 5: "FMAX", 18: "ADC_INL"}

# ---------------- lots, timeline, genealogy ----------------
t0 = pd.Timestamp("2026-08-18 06:00")
sort_ts = [t0 + pd.Timedelta(hours=float(h)) for h in np.cumsum(rng.uniform(30, 38, a.lots))]
lot_ids = [f"L26{2200 + i:04d}" for i in range(a.lots)]
ETCH_PM = pd.Timestamp("2026-09-30 14:00")     # ETCH-07 chamber C preventive maintenance
VT_SHIFT_LOTS = set(range(int(a.lots * 0.65), int(a.lots * 0.65) + 6))
CMP_WINDOW = set(range(int(a.lots * 0.25), int(a.lots * 0.25) + 7))
RETICLE_LOTS = set(range(int(a.lots * 0.45), int(a.lots * 0.45) + 3))
etch_tools = ["ETCH-05", "ETCH-06", "ETCH-07", "ETCH-08"]
lot_rows, ctx_rows, wat_rows = [], [], []
lot_meta = []
for i, lot in enumerate(lot_ids):
    st = sort_ts[i]
    etch_ts = st - pd.Timedelta(days=3.5)
    etool = etch_tools[i % 4] if not (etch_ts > ETCH_PM and i % 2 == 0) else "ETCH-07"   # ETCH-07 busier after PM
    m = dict(lot=lot, idx=i, sort_ts=st, etch_tool=etool, etch_ts=etch_ts,
             cmp_tool=["CMP-01", "CMP-02", "CMP-03"][rng.integers(3)],
             reticle="RT-118B" if i in RETICLE_LOTS else rng.choice(["RT-118A", "RT-118C"]),
             litho=f"LIT-0{rng.integers(1, 5)}", implant=f"IMP-0{rng.integers(1, 3)}",
             anneal=f"ANN-0{rng.integers(1, 3)}", dep=f"DEP-0{rng.integers(1, 4)}",
             clean=rng.choice(["WET-01", "WET-02"], p=[0.6, 0.4]), handler=rng.choice(["ROB-01", "ROB-02"]),
             tester=f"T0{rng.integers(1, 7)}", probe_card=["PC-4410", "PC-4411", "PC-4412", "PC-4413"][i % 4],
             vt_shift=18.0 if i in VT_SHIFT_LOTS else 0.0)
    if i in CMP_WINDOW and i % 2 == 1: m["cmp_tool"] = "CMP-03"
    lot_meta.append(m)
    lot_rows.append(dict(lot_id=lot, product=PRODUCT, fab=FAB, technology="N7-class", wafer_count=a.wafers,
                         fab_start_ts=(st - pd.Timedelta(days=58)).isoformat(), sort_start_ts=st.isoformat(),
                         tester_id=m["tester"], probe_card_id=m["probe_card"], program_rev="AX7_CP1_R4.2",
                         prober_temp_c=85))
    steps = [("DEP", m["dep"], None, "DEP_TEOS_v3", 30), ("LITHO", m["litho"], m["reticle"], "LITH_M1_v7", 24),
             ("ETCH", etool, "chamber", "ETCH_M1_v12", 3.5), ("CMP", m["cmp_tool"], None, "CMP_CU_v5", 3.0),
             ("IMPLANT", m["implant"], None, "IMP_NWELL_v2", 40), ("ANNEAL", m["anneal"], None, "RTA_1050_v4", 38),
             ("WET_CLEAN", m["clean"], None, "SC1_v9", 2.5), ("WAFER_SORT_LOAD", m["handler"], None, "LOAD_v1", 0.2)]
    m["chambers"] = rng.choice(list("ABCD"), a.wafers)
    for w in range(1, a.wafers + 1):
        for step, tool, extra, recipe, days_before in steps:
            tin = st - pd.Timedelta(days=days_before) + pd.Timedelta(minutes=float(w * 7 + rng.uniform(0, 5)))
            ctx_rows.append(dict(lot_id=lot, wafer_id=w, step=step, equipment_id=tool,
                                 chamber=m["chambers"][w - 1] if extra == "chamber" else "",
                                 reticle_id=extra if step == "LITHO" else "", recipe=recipe,
                                 ts_in=tin.isoformat(), ts_out=(tin + pd.Timedelta(minutes=float(rng.uniform(4, 40)))).isoformat()))
        for s, (sx, sy) in enumerate([(0, 0), (0.6, 0), (-0.6, 0), (0, 0.6), (0, -0.6)], 1):
            wat_rows.append(dict(lot_id=lot, wafer_id=w, wat_site=s, site_x=sx, site_y=sy,
                                 vt_n_mV=round(rng.normal(312 + m["vt_shift"], 4.5), 2),
                                 vt_p_mV=round(rng.normal(-298, 4.5), 2),
                                 idsat_n_uA_um=round(rng.normal(1080 - 2.0 * m["vt_shift"], 18), 1),
                                 rs_poly_ohm_sq=round(rng.normal(7.8, 0.15), 3),
                                 contact_chain_ohm=round(rng.normal(41.0, 1.2), 2)))

# probe card touchdown counter (card wear) and site-3 drift for PC-4412
td = {pc: 18000 for pc in ["PC-4410", "PC-4411", "PC-4412", "PC-4413"]}

# ---------------- die-level simulation ----------------
die_rows, retest_rows, ft_rows, die_gt, wafer_gt = [], [], [], [], []
neighbors = {}
idx_of = {(x, y): k for k, (x, y) in enumerate(zip(DX, DY))}
NB = [[idx_of.get((x + dx, y + dy)) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx or dy)] for x, y in zip(DX, DY)]

def stop_on_fail(P):
    """Return soft bin and executed test time for every die (vectorised)."""
    sb = np.ones(NDIE, int); tt = np.zeros(NDIE); alive = np.ones(NDIE, bool)
    for _, L in LIMITS.iterrows():
        v = P[L.test_name]
        fail = np.zeros(NDIE, bool)
        if pd.notna(L.lo_limit): fail |= v < L.lo_limit
        if pd.notna(L.hi_limit): fail |= v > L.hi_limit
        tt[alive] += L.test_time_ms
        newf = alive & fail
        sb[newf] = L.soft_bin; alive &= ~fail
    return sb, tt

hb_of = dict(zip(LIMITS.soft_bin, LIMITS.hard_bin)); hb_of[1] = 1
for m in lot_meta:
    i, lot = m["idx"], m["lot"]
    lot_offset = rng.normal(0, 0.6)
    for w in range(1, a.wafers + 1):
        wk = f"{lot}-W{w:02d}"
        chamber = m["chambers"][w - 1]
        defect = np.zeros(NDIE, bool); source = np.array([""] * NDIE, dtype=object)
        labels, causes = [], []
        # random defects: negative binomial (gamma-mixed Poisson), D0 ~ 0.10/cm2, die area ~ 0.75 cm2
        lam = rng.gamma(2.0, 0.075 / 2.0) * np.ones(NDIE)
        rnd = rng.random(NDIE) < 1 - np.exp(-lam)
        defect |= rnd; source[rnd] = "random"
        # edge die loss baseline
        e = (R > 0.93) & (rng.random(NDIE) < 0.06); defect |= e; source[e & (source == "")] = "edge_baseline"
        # 1) ETCH-07 chamber C after PM -> edge ring
        if m["etch_tool"] == "ETCH-07" and chamber == "C" and m["etch_ts"] > ETCH_PM:
            p = 0.75 * (R > 0.80); hit = rng.random(NDIE) < p
            defect |= hit; source[hit] = "etch07C_edge_ring"; labels.append("Edge-Ring"); causes.append("ETCH-07/C post-PM uniformity")
        # 2) CMP-03 pad issue -> center
        if m["cmp_tool"] == "CMP-03" and i in CMP_WINDOW and rng.random() < 0.55:
            hit = rng.random(NDIE) < 0.7 * np.exp(-(R / 0.30) ** 2)
            defect |= hit; source[hit] = "cmp03_center"; labels.append("Center"); causes.append("CMP-03 pad wear")
        # 3) ROB-02 scratch
        if m["handler"] == "ROB-02" and rng.random() < 0.06:
            th = rng.uniform(0, np.pi); off = rng.uniform(-3, 3)
            d = np.abs((DX - cx) * np.sin(th) - (DY - cy) * np.cos(th) - off)
            seg = np.abs((DX - cx) * np.cos(th) + (DY - cy) * np.sin(th)) < rng.uniform(5, 10)
            hit = (d < 0.7) & seg & (rng.random(NDIE) < 0.9)
            defect |= hit; source[hit] = "rob02_scratch"; labels.append("Scratch"); causes.append("ROB-02 end effector")
        # 4) chuck contamination -> local cluster
        if rng.random() < 0.03:
            px, py = rng.uniform(4, GRID - 4, 2)
            hit = (np.hypot(DX - px, DY - py) < rng.uniform(1.8, 3.2)) & (rng.random(NDIE) < 0.85)
            defect |= hit; source[hit] = "local_cluster"; labels.append("Loc"); causes.append("prober chuck particle")
        # 5) reticle RT-118B -> repeating die in 3x3 field position (1,2)
        if m["reticle"] == "RT-118B":
            hit = ((DX % 3) == 1) & ((DY % 3) == 2) & (rng.random(NDIE) < 0.8)
            defect |= hit; source[hit] = "reticle_118B"; labels.append("Reticle-Repeat"); causes.append("RT-118B mask defect")
        if not labels: labels = ["Clean"]

        # ---- parametric values ----
        wafer_off = rng.normal(0, 0.8)
        vt = m["vt_shift"] + rng.normal(0, 3)
        P = {}
        P["IDDQ"] = 52 + lot_offset + wafer_off + 3.0 * R ** 2 + rng.normal(0, 1.3, NDIE)
        P["LEAKAGE"] = np.exp(rng.normal(np.log(12), 0.25, NDIE))
        P["SCAN_FAIL_COUNT"] = np.zeros(NDIE)
        P["MBIST_FAIL"] = np.zeros(NDIE)
        P["PLL_LOCK"] = rng.normal(18, 1.4, NDIE)
        P["VMIN"] = 0.62 + 0.0015 * vt + 0.004 * R + rng.normal(0, 0.012, NDIE)
        P["FMAX"] = 2400 - 3000 * (P["VMIN"] - 0.62) + rng.normal(0, 22, NDIE)
        P["RING_OSC"] = 0.5 * P["FMAX"] + rng.normal(0, 2.5, NDIE)
        P["ADC_INL"] = np.abs(rng.normal(0.40, 0.15, NDIE))
        # contact resistance with probe card wear; PC-4412 site 3 needle degrading
        cr = 0.80 + rng.normal(0, 0.07, NDIE) + 0.00000001 * td[m["probe_card"]]
        if m["probe_card"] == "PC-4412":
            drift = max(0, (td["PC-4412"] - 30000) / 18000)
            s3 = SITE == 3
            cr[s3] += drift * 0.9 + rng.exponential(0.25 * drift + 1e-6, s3.sum())
        P["CONTACT_RES"] = cr
        # failure modes for defective die, by source
        mode_p = {"random": [0.35, 0.30, 0.20, 0.10, 0.05], "edge_baseline": [0.30, 0.25, 0.30, 0.10, 0.05],
                  "etch07C_edge_ring": [0.10, 0.40, 0.45, 0.03, 0.02], "cmp03_center": [0.50, 0.10, 0.05, 0.33, 0.02],
                  "rob02_scratch": [0.55, 0.20, 0.10, 0.15, 0.00], "local_cluster": [0.40, 0.30, 0.20, 0.10, 0.00],
                  "reticle_118B": [0.85, 0.05, 0.00, 0.10, 0.00]}
        for src, pr in mode_p.items():
            k = np.where(defect & (source == src))[0]
            if not len(k): continue
            mode = rng.choice(5, len(k), p=pr)
            P["SCAN_FAIL_COUNT"][k[mode == 0]] = rng.integers(1, 400, (mode == 0).sum())
            P["IDDQ"][k[mode == 1]] += rng.uniform(16, 140, (mode == 1).sum())
            P["LEAKAGE"][k[mode == 2]] *= rng.uniform(6, 60, (mode == 2).sum())
            P["MBIST_FAIL"][k[mode == 3]] = 1
            P["PLL_LOCK"][k[mode == 4]] += rng.uniform(8, 30, (mode == 4).sum())
        # latent-defect outliers: pass all spec limits but abnormal IDDQ
        latent = (~defect) & (rng.random(NDIE) < 0.003)
        P["IDDQ"][latent] = np.minimum(P["IDDQ"][latent] + rng.uniform(8, 11, latent.sum()), 64.5)

        sb, tt = stop_on_fail(P)
        td[m["probe_card"]] += NDIE // 4 + 1
        passed = sb == 1
        # neighbourhood fail count (for GDBN ground truth)
        fails = ~passed
        nbf = np.array([sum(fails[j] for j in nb if j is not None) for nb in NB])
        gdbn = passed & (nbf >= 5)

        base = dict(lot_id=lot, wafer_id=w, insertion=1, stage="CP1", tester_id=m["tester"],
                    probe_card_id=m["probe_card"], program_rev="AX7_CP1_R4.2", temp_c=85)
        df = pd.DataFrame({"die_x": DX, "die_y": DY, "site": SITE})
        for kk, vv in base.items(): df[kk] = vv
        df["hard_bin"] = [hb_of[s] for s in sb]; df["soft_bin"] = sb; df["pass"] = passed.astype(int)
        df["test_time_ms"] = tt.round(1)
        for t in LIMITS.test_name:
            v = P[t].copy()
            # tests after the failing test are not executed (stop on fail) -> blank
            order = list(LIMITS.test_name).index(t)
            fail_order = np.array([list(LIMITS.soft_bin).index(s) if s != 1 else 99 for s in sb])
            v = np.where(fail_order >= order, v, np.nan)
            df[t] = np.round(v, 4 if t == "VMIN" else 3)
        df["ts"] = (m["sort_ts"] + pd.Timedelta(minutes=(w - 1) * 21) + pd.to_timedelta(np.arange(NDIE) * 2.4, unit="s")).astype(str)
        die_rows.append(df)

        # retest: contact (SB30) fails get a second insertion; most recover after re-seat
        rt = np.where(sb == 30)[0]
        if len(rt):
            P2 = {kk: vv[rt].copy() for kk, vv in P.items()}
            P2["CONTACT_RES"] = np.where(rng.random(len(rt)) < 0.74, 0.8 + rng.normal(0, 0.07, len(rt)), P2["CONTACT_RES"])
            # evaluate retest with same limits
            sb2 = np.ones(len(rt), int); alive = np.ones(len(rt), bool); tt2 = np.zeros(len(rt))
            for _, L in LIMITS.iterrows():
                v = P2[L.test_name]; f = np.zeros(len(rt), bool)
                if pd.notna(L.lo_limit): f |= v < L.lo_limit
                if pd.notna(L.hi_limit): f |= v > L.hi_limit
                tt2[alive] += L.test_time_ms; sb2[alive & f] = L.soft_bin; alive &= ~f
            r = df.iloc[rt].copy(); r["insertion"] = 2; r["soft_bin"] = sb2; r["hard_bin"] = [hb_of[s] for s in sb2]
            r["pass"] = (sb2 == 1).astype(int); r["test_time_ms"] = tt2.round(1)
            for t in LIMITS.test_name:
                order = list(LIMITS.test_name).index(t)
                fo = np.array([list(LIMITS.soft_bin).index(s) if s != 1 else 99 for s in sb2])
                r[t] = np.where(fo >= order, np.round(P2[t], 4 if t == "VMIN" else 3), np.nan)
            retest_rows.append(r)
            passed = passed.copy(); passed[rt[sb2 == 1]] = True

        # final test + burn-in for die that passed sort (after retest)
        g = np.where(passed)[0]
        ecid = [hashlib.sha1(f"{wk}-{x}-{y}".encode()).hexdigest()[:12].upper() for x, y in zip(DX[g], DY[g])]
        assy_fail = rng.random(len(g)) < 0.007
        speed_fail = P["FMAX"][g] + rng.normal(0, 15, len(g)) < 2235
        ft_pass = ~(assy_fail | speed_fail)
        bi_p = np.full(len(g), 0.0003)
        bi_p[latent[g]] = 0.35; bi_p[gdbn[g]] = np.maximum(bi_p[gdbn[g]], 0.03)
        bi_p[(R[g] > 0.93)] = np.maximum(bi_p[(R[g] > 0.93)], 0.001)
        bi_fail = ft_pass & (rng.random(len(g)) < bi_p)
        ft_rows.append(pd.DataFrame({"ecid": ecid, "lot_id": lot, "wafer_id": w, "die_x": DX[g], "die_y": DY[g],
                                     "assembly_lot": f"A{lot[3:]}-{(w - 1) // 13 + 1}", "ft_stage": "FT_HOT",
                                     "ft_pass": ft_pass.astype(int),
                                     "ft_soft_bin": np.where(assy_fail, 40, np.where(speed_fail, 5, 1)),
                                     "ft_fmax_mhz": np.round(P["FMAX"][g] + rng.normal(-10, 15, len(g)), 1),
                                     "burnin_fail": bi_fail.astype(int)}))
        die_gt.append(pd.DataFrame({"lot_id": lot, "wafer_id": w, "die_x": DX, "die_y": DY,
                                    "defect_source": np.where(source == "", "none", source),
                                    "is_latent_outlier": latent.astype(int), "is_gdbn": gdbn.astype(int),
                                    "neighbor_fail_count": nbf}))
        wafer_gt.append(dict(lot_id=lot, wafer_id=w, pattern_labels="|".join(labels), injected_causes="|".join(causes),
                             etch_tool=m["etch_tool"], etch_chamber=chamber, cmp_tool=m["cmp_tool"], reticle_id=m["reticle"],
                             handler=m["handler"], probe_card_id=m["probe_card"], wat_vt_shift_mV=m["vt_shift"],
                             probe_card_touchdowns=td[m["probe_card"]]))

# ---------------- write ----------------
cols = ["lot_id", "wafer_id", "die_x", "die_y", "site", "stage", "insertion", "tester_id", "probe_card_id",
        "program_rev", "temp_c", "hard_bin", "soft_bin", "pass", "test_time_ms"] + list(LIMITS.test_name) + ["ts"]
die = pd.concat(die_rows + retest_rows, ignore_index=True)[cols].sort_values(["lot_id", "wafer_id", "insertion", "die_y", "die_x"])
die.to_csv(f"{OUT}/sort_die_results.csv", index=False)
pd.DataFrame(lot_rows).to_csv(f"{OUT}/lots.csv", index=False)
pd.DataFrame(ctx_rows).to_csv(f"{OUT}/lot_context_mes.csv", index=False)
pd.DataFrame(wat_rows).to_csv(f"{OUT}/wat_pcm.csv", index=False)
LIMITS.assign(soft_bin_name=LIMITS.soft_bin.map(SBIN_NAME)).to_csv(f"{OUT}/test_limits.csv", index=False)
pd.DataFrame([dict(soft_bin=k, name=v, hard_bin=hb_of.get(k, 1)) for k, v in SBIN_NAME.items()]).to_csv(f"{OUT}/bin_definitions.csv", index=False)
ft = pd.concat(ft_rows, ignore_index=True); ft.to_csv(f"{OUT}/final_test_burnin.csv", index=False)

# wafer summary from final disposition (insertion 2 overrides 1 when present)
final = die.sort_values("insertion").drop_duplicates(["lot_id", "wafer_id", "die_x", "die_y"], keep="last")
ws = final.groupby(["lot_id", "wafer_id"]).agg(gross_die=("pass", "size"), good_die=("pass", "sum")).reset_index()
first = die[die.insertion == 1].groupby(["lot_id", "wafer_id"])["pass"].mean().rename("first_pass_yield").reset_index()
ws = ws.merge(first); ws["final_yield"] = (ws.good_die / ws.gross_die).round(4); ws["first_pass_yield"] = ws.first_pass_yield.round(4)
ws.merge(pd.DataFrame(lot_rows)[["lot_id", "sort_start_ts"]]).to_csv(f"{OUT}/wafer_summary.csv", index=False)

pd.DataFrame(wafer_gt).to_csv(f"{GT}/wafer_labels.csv", index=False)
pd.concat(die_gt, ignore_index=True).to_csv(f"{GT}/die_labels.csv", index=False)
events = [
    ("EXCURSION_ETCH", "Edge-Ring pattern on wafers etched in ETCH-07 chamber C after PM", f"etch ts > {ETCH_PM}", "Root cause, wafer map AI, excursion detector"),
    ("CMP_CENTER", "Center pattern on ~55% of CMP-03 wafers", "lots " + ", ".join(lot_ids[j] for j in sorted(CMP_WINDOW) if j % 2 == 1), "Wafer map AI, root cause"),
    ("VT_SHIFT", "WAT Vt_n +18 mV raises Vmin; SB07 VMIN and SB05 FMAX fails rise", "lots " + ", ".join(lot_ids[j] for j in sorted(VT_SHIFT_LOTS)), "Excursion detector, WAT-to-sort correlation, yield forecast"),
    ("PROBE_SITE3", "PC-4412 site 3 needle wear; contact resistance drifts up with touchdowns, SB30 fails ~74% recover on retest", "PC-4412 after ~30k touchdowns", "Root cause (probe vs process), retest advice"),
    ("SCRATCH", "Occasional scratch on wafers loaded by ROB-02", "~6% of ROB-02 wafers", "Wafer map AI"),
    ("LOCAL_CLUSTER", "Random chuck-particle clusters", "~3% of all wafers", "Wafer map AI, GDBN"),
    ("RETICLE", "Repeating fail at field position (1,2) of a 3x3 die field", "lots " + ", ".join(lot_ids[j] for j in sorted(RETICLE_LOTS)), "Wafer map AI (open-set: class not in WM-811K)"),
    ("LATENT_OUTLIERS", "0.3% of die with IDDQ +8 to 11 uA, inside spec; 35% fail burn-in", "die_labels.is_latent_outlier", "DPAT / NNR evaluation against burnin_fail"),
    ("GDBN", "Passing die with >=5 failing neighbours have 3% burn-in fail rate (vs 0.03%)", "die_labels.is_gdbn", "GDBN screen evaluation"),
    ("REDUNDANT_TEST", "RING_OSC is ~0.5 x FMAX (r > 0.98); ADC_INL has almost no fails", "T4120, T6000", "Test time optimizer"),
]
pd.DataFrame(events, columns=["event_id", "description", "scope", "engines_to_test"]).to_csv(f"{GT}/injected_events.csv", index=False)

# quick stats for README
stats = dict(lots=a.lots, wafers=len(ws), die_per_wafer=int(NDIE), sort_rows=len(die), retest_rows=int((die.insertion == 2).sum()),
             ft_rows=len(ft), mean_final_yield=round(float(ws.final_yield.mean()), 4),
             mean_first_pass_yield=round(float(ws.first_pass_yield.mean()), 4),
             burnin_fails=int(ft.burnin_fail.sum()))
sbp = final[final["pass"] == 0].soft_bin.map(SBIN_NAME).value_counts().to_dict()
stats["fail_pareto_final"] = sbp
json.dump(stats, open(f"{OUT}/dataset_stats.json", "w"), indent=2, default=str)
print(json.dumps(stats, indent=2, default=str))
