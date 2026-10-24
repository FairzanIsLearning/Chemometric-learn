"""hs_med_lib.py - helpers for HS_antibiotics_all_meds.ipynb  (medication-history features + the cutoff feature builder from hs_pipeline.py)."""
import numpy as np, pandas as pd
from scipy import stats
D = "../data"        # folder with the csv files; the notebook overrides it: lib.D = DATA_DIR
END = pd.Timestamp("2025-12-31"); T0 = pd.Timestamp("2016-01-02")
DRUGS = {"Clindamycin topical": "topical", "Clindamycin + Rifampicin": "clinrif", "Tetracyclines (oral)": "tetra",
         "Adalimumab": "adali", "Secukinumab": "secuk", "Bimekizumab": "bimek"}

def load_raw():
    dem = pd.read_csv(f"{D}/demographics.csv", parse_dates=["symptom_onset_date", "diagnosis_date"])
    trt = pd.read_csv(f"{D}/treatments.csv", parse_dates=["start_date", "end_date"])
    sur = pd.read_csv(f"{D}/surgeries.csv", parse_dates=["procedure_date"])
    asg = pd.read_csv(f"{D}/patient_assignment.csv")
    tg = pd.read_csv(f"{D}/targets.csv").set_index("patient_id").avg_ihs4_next6
    dem["hurley"] = dem.hurley_stage_baseline.astype(str).str.extract(r"(\d)")[0].astype(int)
    return dem, trt, sur, asg, tg

def med_features(c, dem, trt):
    """Lifetime medication-history features known at cutoff c (one row per patient with dx <= c). Everything uses start_date <= c."""
    c = pd.Timestamp(c); d = dem[dem.diagnosis_date <= c].set_index("patient_id")
    t = trt[trt.start_date <= c].copy(); t["k"] = t.drug_name.map(DRUGS)
    t["end_c"] = t.end_date.where(t.end_date.notna() & (t.end_date <= c), c)       # open or future end -> censor at c
    t["days"] = (t.end_c - t.start_date).dt.days.clip(lower=0)
    f = pd.DataFrame(index=d.index)
    obs_start = d.diagnosis_date.clip(lower=T0)                                       # left-truncated at 2016-01-02
    f["obs_yrs"] = ((c - obs_start).dt.days / 365.25).clip(lower=.25)
    a = t[t.treatment_class == "Antibiotic"].sort_values(["patient_id", "start_date"]); g = a.groupby("patient_id")
    f["n_abx"] = g.size()
    for k in ["topical", "clinrif", "tetra"]:
        f[f"n_{k}"] = a[a.k == k].groupby("patient_id").size()
    f = f.fillna({c_: 0 for c_ in f.columns if c_.startswith("n_")})
    f["abx_rate"] = f.n_abx / f.obs_yrs
    for k in ["topical", "clinrif", "tetra"]:
        f[f"sh_{k}"] = (f[f"n_{k}"] / f.n_abx.replace(0, np.nan))
    f["n_oral"] = f.n_clinrif + f.n_tetra; f["oral_rate"] = f.n_oral / f.obs_yrs
    f["ever_oral"] = (f.n_oral > 0).astype(int); f["ever_rif"] = (f.n_clinrif > 0).astype(int)
    f["n_abx_types"] = (f[["n_topical", "n_clinrif", "n_tetra"]] > 0).sum(axis=1)
    f["abx_days"] = g.days.sum()
    f["abx_first_gap_d"] = (g.start_date.min() - d.diagnosis_date.clip(lower=T0)).dt.days
    gap = g.start_date.diff().dt.days
    f["abx_gap_med"] = gap.groupby(a.patient_id).median()
    f["abx_gap_cv"] = gap.groupby(a.patient_id).std() / gap.groupby(a.patient_id).mean()          # burstiness (0 regular, >1 clustered)
    f["abx_gap_min"] = gap.groupby(a.patient_id).min()
    for y in (1, 2, 3):                                                                              # rate in last y years
        lo = c - pd.DateOffset(years=y); f[f"abx_last{y}y"] = a[a.start_date > lo].groupby("patient_id").size()
        f[f"oral_last{y}y"] = a[(a.start_date > lo) & (a.k != "topical")].groupby("patient_id").size()
    f[["abx_last1y", "abx_last2y", "abx_last3y", "oral_last1y", "oral_last2y", "oral_last3y"]] = f[["abx_last1y", "abx_last2y", "abx_last3y", "oral_last1y", "oral_last2y", "oral_last3y"]].fillna(0)
    prior = (f.n_abx - f.abx_last2y) / (f.obs_yrs - 2).clip(lower=.5)
    f["abx_trend"] = (f.abx_last2y / 2) - prior                                                       # recent rate minus earlier rate
    f["abx_since"] = (c - g.start_date.max()).dt.days
    f["last_abx_oral"] = (a.groupby("patient_id").tail(1).set_index("patient_id").k != "topical").astype(float)
    # ---- biologics
    b = t[t.treatment_class == "Biologic"]; gb = b.groupby("patient_id")
    f["n_bio_lines"] = gb.size(); f["n_bio_drugs"] = gb.k.nunique()
    for k in ["adali", "secuk", "bimek"]: f[f"ever_{k}"] = b[b.k == k].groupby("patient_id").size().gt(0).astype(float)
    f["bio_days"] = gb.days.sum(); f["bio_share"] = f.bio_days / (f.obs_yrs * 365.25)
    f["ever_bio"] = (f.n_bio_lines.fillna(0) > 0).astype(int)
    f["bio_start_delay_d"] = (gb.start_date.min() - d.diagnosis_date).dt.days
    f["n_bio_stops"] = b[b.end_date.notna() & (b.end_date <= c)].groupby("patient_id").size()
    f = f.fillna({"n_bio_lines": 0, "n_bio_drugs": 0, "bio_days": 0, "bio_share": 0, "n_bio_stops": 0, "ever_adali": 0, "ever_secuk": 0, "ever_bimek": 0})
    f["n_drugs_ever"] = f.n_abx_types + f.n_bio_drugs
    f["any_treatment"] = (f.n_abx + f.n_bio_lines > 0).astype(int)
    f["hurley"] = d.hurley
    return f

def ols_robust(y, X, names):
    """OLS with HC1 robust SE. X includes constant. Returns DataFrame(coef, se, z, p)."""
    X = np.asarray(X, float); y = np.asarray(y, float); n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X); b = XtXi @ X.T @ y; e = y - X @ b
    V = XtXi @ (X.T * (e ** 2)) @ X @ XtXi * n / (n - k); se = np.sqrt(np.diag(V)); z = b / se
    return pd.DataFrame({"coef": b, "se": se, "z": z, "p": 2 * stats.norm.sf(abs(z))}, index=names)

# ---------------- baseline feature builder (copied from hs_pipeline.py, unchanged) ----------------
from sklearn.ensemble import HistGradientBoostingRegressor as HGR
def _cnt(df, col, c, days, name, pid="patient_id"):
    m = (df[col] <= c) & (df[col] > c - pd.Timedelta(days=days))
    return df[m].groupby(pid).size().rename(name)

def build(c, dem, asg, trt, sur, vis, cats=None, pstats=None):
    cats = {} if cats is None else cats          # category -> code maps, frozen at train time
    def code(series, key):
        s_ = series.where(series.isna(), series.astype(str))
        if key not in cats: cats[key] = sorted(s_.dropna().unique())
        return pd.Categorical(s_, categories=cats[key]).codes
    c = pd.Timestamp(c)
    f = dem[dem.diagnosis_date <= c].set_index("patient_id").copy()
    f["age"] = c.year - f.birth_year
    f["yrs_since_dx"] = (c - f.diagnosis_date).dt.days / 365.25
    f["prevalent"] = (f.diagnosis_date == pd.Timestamp("2016-01-02")).astype(int)
    f["sex"] = (f.sex.astype(str).str[0] == "M").astype(int)
    for k in ["ethnicity","smoking_status","family_history_hs"]:
        f[k] = code(f[k], k)
    f["hurley"] = f.hurley_stage_baseline.astype(str).str.extract(r"(\d)")[0].astype(float)
    f = f.drop(columns=["birth_year","symptom_onset_date","diagnosis_date","hurley_stage_baseline"])
    # ---- treatments
    t = trt[trt.start_date <= c]
    a = t[t.treatment_class == "Antibiotic"]
    for d in (90, 180, 365): f = f.join(_cnt(a, "start_date", c, d, f"abx_{d}d"))
    f = f.join(a.groupby("patient_id").start_date.agg(abx_all="size", abx_last="max"))
    f["abx_since"] = (c - f.abx_last).dt.days; f = f.drop(columns="abx_last")
    b = t[t.treatment_class == "Biologic"].copy()
    b["open"] = b.end_date.isna() | (b.end_date > c)
    cur = b[b.open].sort_values("start_date").groupby("patient_id").tail(1).set_index("patient_id")
    f["on_bio"] = f.index.isin(cur.index).astype(int)
    f = f.join(cur[["treatment_line"]].rename(columns={"treatment_line":"cur_line"}))
    f["days_on_bio"] = (c - cur.start_date).dt.days.reindex(f.index)
    f["bio_drug"] = pd.Series(code(cur.drug_name, "drug"), cur.index).reindex(f.index)
    f["n_lines"] = b.groupby("patient_id").size().reindex(f.index)
    last_end = b[~b.open].groupby("patient_id").end_date.max()
    f["days_since_bio_stop"] = (c - last_end).dt.days.reindex(f.index)
    first_b = b.groupby("patient_id").start_date.min()
    f["days_dx_to_bio"] = (first_b - dem.set_index("patient_id").diagnosis_date).dt.days.reindex(f.index)
    # ---- surgeries
    s = sur[sur.procedure_date <= c]
    for d in (180, 365, 730): f = f.join(_cnt(s, "procedure_date", c, d, f"surg_{d}d"))
    f = f.join(s.groupby("patient_id").procedure_date.agg(surg_all="size", surg_last="max"))
    f["surg_since"] = (c - f.surg_last).dt.days; f = f.drop(columns="surg_last")
    # ---- visits (the new part)
    v = vis[vis.visit_date <= c].copy()
    v["age_d"] = (c - v.visit_date).dt.days
    v["flare"] = (v.visit_type == "flare_reactive").astype(int)
    sc = v[v.ihs4_score.notna()]
    g = v.groupby("patient_id")
    f["v_n_all"] = g.size()
    f["v_days_since"] = g.age_d.min()
    f["v_first_days"] = g.age_d.max()
    for w in (30, 90, 180, 365):
        vw = v[v.age_d < w]; sw = vw[vw.ihs4_score.notna()]
        f[f"v_n_{w}"] = vw.groupby("patient_id").size()
        f[f"v_flare_share_{w}"] = vw.groupby("patient_id").flare.mean()
        f[f"v_scored_{w}"] = vw.groupby("patient_id").ihs4_score.apply(lambda x: x.notna().mean())
        f[f"s_mean_{w}"] = sw.groupby("patient_id").ihs4_score.mean()
        f[f"s_max_{w}"] = sw.groupby("patient_id").ihs4_score.max()
        f[f"s_min_{w}"] = sw.groupby("patient_id").ihs4_score.min()
        nf = sw[sw.flare == 0]
        f[f"s_nonflare_mean_{w}"] = nf.groupby("patient_id").ihs4_score.mean()
    for ty in ["background","follow_up","treatment_monitoring","flare_reactive"]:
        x = sc[(sc.visit_type == ty) & (sc.age_d < 365)]
        f[f"s_{ty}_365"] = x.groupby("patient_id").ihs4_score.mean()
    # last scores, decayed means
    last = sc.groupby("patient_id").tail(3)
    r = last.groupby("patient_id").cumcount(ascending=False)
    for k in (0, 1, 2):
        f[f"s_last{k}"] = last[r == k].set_index("patient_id").ihs4_score.reindex(f.index) if False else last[last.groupby("patient_id").cumcount(ascending=False) == k].set_index("patient_id").ihs4_score
    f["s_last0_age"] = sc.groupby("patient_id").age_d.min()
    f["s_last0_flare"] = sc.groupby("patient_id").tail(1).set_index("patient_id").flare
    for hl in (45, 120, 365):
        w_ = np.exp(-sc.age_d / hl); sc = sc.assign(w=w_, ws=w_ * sc.ihs4_score)
        gg = sc.groupby("patient_id")[["w","ws"]].sum()
        f[f"s_ewm_{hl}"] = gg.ws / gg.w; f[f"s_ewm_w_{hl}"] = gg.w
    f["s_all_mean"] = sc.groupby("patient_id").ihs4_score.mean()
    # trend: mean of last 180d minus mean of 180-365d; visit-rate trend
    f["s_trend"] = f.s_mean_180 - (sc[(sc.age_d >= 180) & (sc.age_d < 365)].groupby("patient_id").ihs4_score.mean())
    f["v_rate_trend"] = f.v_n_180 / (f.v_n_365 - f.v_n_180).replace(0, np.nan)
    # gaps
    gp = v.sort_values(["patient_id","visit_date"]).groupby("patient_id").visit_date.diff().dt.days
    f["v_gap_med"] = gp.groupby(v.patient_id).median()
    f["v_gap_max_365"] = gp[v.age_d < 365].groupby(v.patient_id).max()
    # practitioner scoring rate and offset (computed from visits before c only)
    pr = v.merge(asg, on="patient_id", suffixes=("","_a"))
    prs = pr.groupby("practitioner_id_a").ihs4_score.agg(p_score_rate=lambda x: x.notna().mean(), p_mean="mean") if pstats is None else pstats
    f = f.join(asg.set_index("patient_id").practitioner_id.rename("pid")); f = f.join(prs, on="pid").drop(columns="pid")
    f = f.drop(columns=[x for x in ["practitioner_id"] if x in f])
    f = f.fillna({k: 0 for k in f.columns if k.startswith(("abx_","surg_","v_n_"))} | {"n_lines": 0})
    f.attrs["pstats"] = prs; f.attrs["cats"] = cats
    return f
GB = lambda **k: HGR(**({"loss": "absolute_error", "learning_rate": .05, "max_iter": 400, "max_leaf_nodes": 24, "min_samples_leaf": 40, "l2_regularization": 1., "random_state": 0} | k))
GB2 = lambda: HGR(loss="absolute_error", learning_rate=.03, max_iter=900, max_leaf_nodes=16, min_samples_leaf=60, l2_regularization=3., random_state=0)
GB3 = lambda: HGR(loss="squared_error", learning_rate=.04, max_iter=600, max_leaf_nodes=16, min_samples_leaf=50, l2_regularization=3., random_state=0)

def fit_blend(X, Y):
    """Three gradient-boosting models; predictions are averaged. The third learns log(1+y)."""
    return [GB().fit(X, Y), GB2().fit(X, Y), GB3().fit(X, np.log1p(Y))]
def predict_blend(ms, X):
    return np.mean([ms[0].predict(X), ms[1].predict(X), np.expm1(ms[2].predict(X))], axis=0)


