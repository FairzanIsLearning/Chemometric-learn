# HS (hidradenitis suppurativa) models: forecast + three applications

For Healthtech 2030 Barcelona hackathon (Almirall case, group project). Multimodal analysis the dynamic of HS disease from clinical records, RNAseq, and imaging data. Purpose: create a model to predict the disease severity, marked by IHS4 score, 6 months in the future.

## Contributions and tools

**My role in the team:**
- Directed the multimodal data exploration and decided which findings entered the model (e.g. dropping the practitioner family, excluding response types as an  input, the hidden-flare block) and what output that is relevant.
- Defined the validation approach: as-of leakage rule, grouped splits, ablation gates, the PASS/WARN/FAIL ledger, and the CLINICAL / ARTEFACT / UNDETERMINED labelling of findings.
- Interpreted results and wrote the limitations.
- Expand model application for other four use case
**AI assistance:** The modelling code was mostly written with Claude (Anthropic) under my direction. I reviewed the outputs and take responsibility for the analytical decisions. 

**Gabriel's role:** Bizplan, implementation for patient's side, roadmap planning.

Packages

| package | role |
|---|---|
| `numpy`, `pandas` | arrays and tables, dates, grouping |
| `scipy.stats` | tests, intervals |
| `scikit-learn` | `HistGradientBoostingRegressor` / `Classifier` (forecast, classifiers), `KFold` / `GroupKFold` / `StratifiedKFold` (honest splits), `PCA`, `StandardScaler`, `RidgeCV`, `make_pipeline` (embedding correction), `LogisticRegression` (stacking), `GaussianMixture` (response types), `roc_auc_score` |
| `seaborn`, `matplotlib` | figures |
| `google.colab` | Drive mount and downloads (handler block only) |
| `os`, `time`, `json`, `urllib`, `pathlib`, `shutil`, `warnings` | files, timing, GitHub reads |

## 0. I/O
All tables join on `patient_id`. Training uses the 21,000-patient train folder; a test set has the same tables.

| file | what it holds | notes |
|---|---|---|
| `demographics.csv` | sex, birth year, age at diagnosis, ethnicity, BMI, smoking, family history, onset and diagnosis dates, diagnostic delay, baseline Hurley stage | some fields missing; missingness can carry information |
| `visits.csv` (or `visits1.csv` + `visits2.csv`) | visit date, visit type (background / flare_reactive / follow_up / treatment_monitoring), `ihs4_score` | scores are blank at some visits |
| `treatments.csv` | antibiotic courses and biologic lines: drug, route, line number, start, end | a biologic with a blank end date is still running |
| `surgeries.csv` | procedure type and date | |
| `patient_assignment.csv` | anonymous practitioner (clinic) per patient | not a model input (see section 5) |
| `targets.csv` | `avg_ihs4_next6` | training only |
| `rnaseq_*`, `images_*` | pre-computed embeddings (RNAseq 128-d, 1,000 samples; images 256-d, 4,000 samples) plus metadata (batch / device, collection date) | subsets only; Exercise 3 only |

Outputs
| file | columns | model |
|---|---|---|
| Exercise 1 | `patient_id, pred_avg_ihs4_next6` | forecast **without** embeddings |
| Exercise 2 | `patient_id, pred_avg_ihs4_next6, super_responder_prob` | same forecast; probability from the detector (patients on a biologic) |
| Exercise 3 images | `patient_id, pred_avg_ihs4_next6` | forecast + image correction |
| Exercise 3 images + RNAseq | `patient_id, pred_avg_ihs4_next6` | forecast + RNAseq correction |
| `patient_report_<name>.csv` | Additional classification based on the data exploration result that might be interesting | |

> Exercise 3 is scored against the Exercise 1 predictions for the same patients, so Exercise 1 and 2 use the clean no-embedding forecast by default. `ex1_uses_emb=True` adds the RNAseq correction to them (small overall gain, but the Exercise 3 comparison then shows nothing).

## 1. What the model predicts

| item | definition |
|---|---|
| target | `avg_ihs4_next6`: the patient's **underlying, noise-free** disease activity averaged over every day of 1 Jan - 30 Jun 2026, **including flares that never produce a visit** |
| cutoff | 31 Dec 2025 (`END`); nothing after it may be used |
| population | patients who do not start, stop or switch a biologic during the window; treatment status at the cutoff stays in force |
| error measure | MAE (mean absolute error, in IHS4 points) |

IHS4 is an integer score that is noisy even when measured, and is recorded at only 64% of visits.

## 2. How the model is built

### 2.1 The "as-of" rule (leakage control)
`as_of(T, c)` cuts every table at date `c`: events after `c` are dropped, and biologic end dates after `c` are blanked (on date `c` the line was still running). `build(T, c, ...)` computes all features from that cut. Training uses the same builder at the real cutoff and, for the time-forward backtest, at earlier dates with pseudo-labels. A test deliberately corrupts the future data and checks that the features do not change.

### 2.2 Features (`build`, grouped into families)
| family | examples |
|---|---|
| `hurley` | baseline Hurley stage |
| `demographics` | age, sex, BMI, smoking, ethnicity, family history, years since diagnosis, age at diagnosis, diagnostic delay, prevalent flag (patients diagnosed before records start) |
| `antibiotics` | courses in the last 90 / 180 / 365 days, total courses, days since the last |
| `surgeries` | surgeries in the last 180 / 365 / 730 days |
| `biologic` | on a biologic now, current line, days on it, drug, number of lines, days since the last stop, days from diagnosis to first biologic |
| `visit_scores` | scored-visit statistics (`s_*`): non-flare, decayed, shrunk levels |
| `visit_behaviour` | visit counts (`v_n_*`), flare-visit share, visit gaps, scoring rate, days since last visit |
| `hidden_flare` | `hid_*`, `seen_*`, `flare_cover_abx_365` |
| `practitioner` | clinic score rate and mean (artefact family, off) |

### 2.3 Baseline ladder (the floor every model must beat)
| baseline | MAE |
|---|---|
| B0 population median | 4.33 |
| B1 Hurley-stage median | 2.67 |
| B2 Hurley x biologic | 2.30 |
| B3 last score | 2.64 |
| B7 **shrunk, decayed, non-flare level** (best baseline) | 1.87 |

B7 estimates each patient's usual level from scores outside flares, weights recent scores more, and pulls patients with few scores toward their Hurley-by-biologic group (empirical-Bayes shrinkage). The raw average of recent scores is **not** used: visits are informative (more frequent in flares), so it overstates the usual level (finding F4b).

### 2.4 Hidden-flare block
The target includes flares that never produce a scored visit. An antibiotic start marks a flare, and surgeries follow antibiotic courses. For each course or surgery in the last year, the model asks whether a scored visit sits near it (antibiotics: -7 to +21 days; surgeries: +/-14 days). If not, it is a **hidden flare** candidate. `fit_uplift` measures, per Hurley stage, how many IHS4 points a score taken inside an antibiotic course exceeds the same patient's other scores; `hidden_flare_feats` turns the counts into an expected unseen burden (`hid_burden_180/365`). A right-censoring guard skips courses started less than 21 days before the cutoff. Surgeries are matched to visits by date, not by `visit_id`, because the share of surgeries carrying a `visit_id` drifts over time.

### 2.5 Forecast core and blend
Gradient boosting on tabular + visit features (+ hidden flare). The shipped model (`make_models`, `fit_blend`, `predict_blend`) averages three `HistGradientBoostingRegressor`s:

| model | loss | notes |
|---|---|---|
| 1 | absolute error | learning rate 0.05, 400 iterations, 24 leaves |
| 2 | absolute error | learning rate 0.03, 900 iterations, 16 leaves, L2 3.0 |
| 3 | squared error on `log1p(target)` | learning rate 0.04, 600 iterations, 16 leaves; back-transformed with `expm1` |

The prediction is the mean of the three, clipped at 0.

### 2.6 Ablation gates (what stays in)
A feature family enters only if the paired-bootstrap 95% CI of its MAE gain is above 0 (`paired_gain`). Decisions written to `CFG`:

| option | decision |
|---|---|
| `use_hidden_flare` | **True** (gain +0.0066, CI 0.0035 to 0.0096) |
| `use_practitioner` | **False** (gain +0.0010 with CI crossing 0; negative when whole practitioners are held out) |
| `use_response_types` | **False** (gain +0.0017, CI crosses 0): classification is not an input |

### 2.7 Validation layer
Every check is logged with `check(step, name, ok, detail, warn)` as PASS / WARN / FAIL (WARN = a known limit that is reported, not hidden).

### 2.8 Results
| model | MAE |
|---|---|
| tabular only | 1.99 |
| visits only | 1.69 |
| tabular + visits | 1.44 |
| + hidden flare | 1.435 |
| 3-model blend | **1.43** |
| Exercise 2a (on a biologic) | 1.17 |
| dry run on 4,200 unseen patients | 1.41 |

Evaluation is 5-fold out of fold. The best baseline B7 scores 1.87.

## 3. How to run
1. Set `DATA_DIR` (train folder) in the setup cell.
2. Run the notebook top to bottom. Sections 1 to 10 build and validate the model; 10c fits it on all 21,000 train patients; 10d fits the super-responder detector; 10e fits the patient report (about 3 minutes).
3. In section 11, set the test set and the paths of the templates; run the handler. The four answer files and `patient_report_<name>.csv` are written to the answers folder (and copied to Drive in Colab).
4. Read section 12 (the ledger) and section 13 (limits) before using the numbers.

## 4. Limits

* **Surgery benefit is confounded by indication**: an association, not an effect.
* **Predicted response types are weak** before treatment (AUC about 0.6 to 0.7); observed types (from first-line scores) are reliable.
* **Images help only recent biologic starters**; RNAseq supersedes images where both exist; claims are scoped to those subsets.
* **The super-responder probability is calibrated to a proxy label** built from the same scores; only 49 positives had RNAseq.
* **Severity extrapolation fails**: do not apply the model to patients much sicker than the training set.
* **Hidden flares are estimated, not observed**: the gain from the hidden-flare block is small (0.007 MAE).
* **Test embeddings were not available at training time**, so the test_1 / test_2 shift in the embedding effect has not been measured.
* **Findings are for this cohort.** Findings are labelled CLINICAL / ARTEFACT / UNDETERMINED in the ledger to show which are likely to carry over.

## 5. Application of the model

| notebook | question it answers | main output |
|---|---|---|
| `HS_forecast_model.ipynb` | What will each patient's average IHS4 be over 1 Jan - 30 Jun 2026? Who will worsen? Who is a super-responder? | 4 answer files + `patient_report_<name>.csv` |
| `HS_trial_simulation.ipynb` | How many patients, which endpoint, which visit schedule, which enrichment for an HS trial? | sample-size tables, power curves, sensitivity ranges |
| `HS_patient_type_classifier.ipynb` | Which type of patient is this, given a (possibly incomplete) record? | type probabilities, "uncertain" when the record is too thin |
| `HS_prognostic_enrichment.ipynb` | Can a frozen prognostic model shrink a trial, balance arms, select active patients? | variance/sample-size gains, type I error, frozen model + hash |

All four are executed top to bottom and end with a PASS / WARN / FAIL ledger. A WARN is a documented limit, not a hidden failure.

---

## 5a. `HS_trial_simulation.ipynb`

### What it is
A **generative** model, deliberately separate from the forecast (which predicts; this one simulates), used to choose sample size, endpoint, visit schedule and enrichment before a trial.

### Building blocks (each estimated from data, on half A of the patients)
| block | model | estimate |
|---|---|---|
| baseline patients | Hurley mix; stable level S lognormal per stage, by method of moments | median S: 5.3 / 11.0 / 19.6 for Hurley 1 / 2 / 3 |
| latent activity | S plus Poisson flares, gamma amplitude, exponential decay | amplitude about 4.9, decay about 15 days, 1.3-2.5 marked flares per year |
| hidden flares | multiplier on flare rate, calibrated on the semivariogram | 1.5 (UNDETERMINED mechanism) |
| visit and scoring | visits by stage and biologic status, reactive visits after flares, scoring probability by visit type, clinic random effect | scoring probability SD between clinics 0.18 |
| measurement noise | variance n0 + n1 * level from same-day pairs and the semivariogram nugget | n0 about 2.0 |
| treatment | response-type mixture (deep / partial / minimal = 15 / 30 / 54%), logistic ramp over months 1-4, plateau 0.22 / 0.53 / 0.95 of pre-drug level, slow drift, flare-rate multipliers, stopping, return after stopping | from first-line starters |
| placebo and drop-out | **assumptions** with ranges | placebo 0-18%, drop-out 5-35% per year |

### Validation
* **V1 parameter recovery:** the estimators are run on simulated tables; most parameters come back within 20-25%. Known weak: flare decay time (biased up by hidden flares), return time after stopping, deep-responder stopping rate.
* **V2 untreated half B (held out):** mean score by Hurley within 10%, semivariogram within 12%, regression to the mean in range. One WARN: simulated visit frequency is about 4 per patient-year vs 7 real.
* **V3 first-line starters (held out):** median on/pre ratio, share responding at least 50%, spread and 12-month stop rate within 25%.

### Trial simulator and answers
Candidates are drawn, **two screening scores** at day -28 and 0, eligibility, stratified 1:1 randomisation, drug and placebo curves on the same patient, scheduled visits with attendance and drop-out. Same-seed comparisons across designs.

| design question | finding |
|---|---|
| endpoint | averaging weeks 8-24 needs fewer patients than a single week-16 score; ANCOVA beats change-from-baseline |
| enrichment | a higher screening threshold lowers n per arm and raises the number to screen |
| screening scores | selecting on one score gives a larger regression to the mean than the mean of two |
| recommended design (mean weeks 8-24, every 4 weeks, ANCOVA, mean of two screening scores >= 9) | about **50 per arm**; Monte Carlo power 83%, null-trial type I error 4.6% |
| sensitivity | across all assumptions the requirement runs from about 15 to about 250 per arm (5th to 95th percentile); the largest drivers are placebo response, plateau / deep share and the true drug effect |

### Limitations
* **Placebo response cannot be estimated** (no placebo arm): scenario parameter.
* **Hidden flares** absorb any unexplained 10-60 day variation.
* **On-drug curves are biased** (drugs start near flare peaks; only patients who stay on drug contribute late scores).
* **No dense scheduled scores** in the data: noise and flare shape come from irregular clinic visits. A pilot with scores every 2-4 weeks for 24 weeks would replace three assumptions.
* The response types are descriptive mixture classes, not proven biology.

### Data needed
Demographics (Hurley), visits with scores, treatments with dates (antibiotic courses and biologic lines), clinic id. For a new disease or population, the same tables are needed to re-estimate every block; the placebo and drop-out values must come from outside (prior trials).

---

## 5b. `HS_patient_type_classifier.ipynb`

### Types
`H1_mild`, `H23_never_bio`, `bio1_early` (first biologic under 12 months), `bio1_deep` / `bio1_partial` / `bio1_minimal` (first biologic for 12+ months, by response), `bio2plus_on`, `stopped`. Response type = 3-component mixture on the shrunk `log(on-drug level / pre-drug level)`. Patients on a first biologic for 12+ months who cannot be typed (no scores in both windows) are left out (1,455 patients), not given fake labels.

### How it works
* Features by **block** (demographics, visits, scores, antibiotics, surgeries, treatments); a block that is unknown is NaN with a `has_<block>` flag. **Unknown is not the same as empty.**
* Training uses **block-dropout copies**: each patient appears once complete and three times with random blocks removed.
* Gradient boosting for the 8-way probability; an **abstain** rule ("uncertain" below 0.6).
* Evaluation: 4-fold by patient, under fixed scenarios.

| scenario | accuracy |
|---|---|
| full record | 0.99 |
| no surgeries / no antibiotics | 0.99 |
| no scores | 0.90 |
| no treatment history | 0.75 (visit types still reveal biologic status) |
| visits only, or demographics + visits | about 0.70 |
| demographics only | 0.58 (majority class 0.32) |

With a full record the model mostly **re-derives rule-based labels** (a consistency check). The informative results are the degraded scenarios, and recall by type (deep / partial / minimal drops sharply without scores).

### Separate response-type model
Predicts the deep / partial / minimal type (read from months 9-24 on the first biologic) from information known **before the start**, **plus month 3**, **plus month 6**. AUC for deep responders: 0.70 -> 0.73 -> 0.87. Before treatment the type is only weakly predictable.

### API
`classify_patient(demographics, visits, treatments, surgeries)`: pass `None` for an unknown table and an empty table for known-empty; returns the call (or "uncertain"), the top probabilities and which blocks were missing.

### Limitations
* Six of eight types follow from treatment and Hurley information, so high accuracy with those blocks is expected.
* The response types are statistical classes; a patient near a boundary can switch class.
* One record date (31 Dec 2025) in training; other dates need time-forward re-checks.
* Training drops blocks at random; real missingness is not random, so the scenario table is optimistic.
* Early-score monitoring is not a reason to stop a drug without clinical review.

### Data needed
At minimum demographics; each added block (visits with scores, treatments, antibiotics, surgeries) increases accuracy as in the table above. Dates are required for any dated block.

---

## 5c. `HS_prognostic_enrichment.ipynb`

### Purpose and set-up
Use a **prognostic** model (predicts the untreated course; never sees treatment assignment) in a trial for: (1) covariate adjustment, (2) balanced arms via stratification, (3) selecting patients with enough active disease.

Because no randomised trial exists in the data, **pseudo-trials** are built from untreated windows: an index date `t0` = the second screening visit (two scored visits 14-56 days apart), no biologic from 120 days before to 168 days after, and an outcome = mean score in days 56-168. 17,738 windows from 8,651 patients.

Safeguards:
* **Independent development data:** a random half of the patients, windows up to mid-2022. **Trial pool:** the other patients, windows from 2023, one per patient (2,833 patients; 1,712 eligible = Hurley 2-3 with screening mean >= 8). No shared patients, outcomes end before the pool starts.
* **Pre-index inputs only**, verified by recomputing the features from physically truncated tables (0 of 150 windows differ).
* **Frozen and hashed** (SHA-256) before the pool outcomes are opened; hash re-checked at the end.
* **Two screening scores**, not one.

### Results
| check | result |
|---|---|
| transport | R2 0.73 on the later pool (same as in development); calibration slope 1.02; stable across 2023-2025; leaving clinics out costs nothing |
| gain beyond the standard ANCOVA (baseline, Hurley, prior biologic) | small: variance of the effect estimate x 0.935 (CI 0.913-0.957) in the observational R2; about 1.5% fewer patients in the pseudo-trial power analysis. Against an unadjusted analysis the saving is about 23%, almost all from the standard covariates |
| null simulation (stratified randomisation, 3000 trials) | type I error 5.7% (standard) and 5.3% (with the prognostic score); unadjusted 1.2% (conservative, ignores the stratification variable); estimates unbiased |
| known heterogeneous effect | about 20 per arm adjusted vs 26 unadjusted, confirmed on fresh trials (80.8% power) |
| balance | stratifying on Hurley x prognostic tertile roughly halves the between-arm difference in the prognostic score vs simple randomisation |
| enrichment | prognostic score above the 75th percentile: larger absolute effect, fewer patients per arm, 4 times as many to screen |
| binary endpoints | g-computation of the marginal risk difference is unbiased; IHS4-55 (relative to baseline) gains nothing from covariates; low-activity endpoint gains about 3% |
| regression to the mean | selecting on one score vs the mean of two differs by about 0.4 points in the real data (smaller than in the simulator; undetermined why) |

### Limitations
* The outcome is built from **irregular clinic visits** (more visits in flares), so the R2 is an **upper bound** for a scheduled-visit trial.
* Pseudo-trials are untreated; the treatment effect is **added** (multiplicative, response-type mixture). Drug-specific behaviour is covered by the trial-simulation notebook.
* The binary responder is a stand-in for IHS4-55 (different time window, two-score baseline).
* Enrichment narrows the population the result applies to.
* A frozen model needs a stated life: after new data or new sites, refit, re-freeze, re-hash, repeat the transport checks.

### Data needed
For development: historical untreated (or placebo-arm) patients with the **same endpoint definition and timing**, independent of the trial. For the trial: baseline data at screening (two scores, Hurley, prior biologic, strata) and the blinded outcome. The model must be frozen before database lock.

---

## 6. How to run

1. Set `DATA_DIR` (local folder of the training tables or the GitHub folder link; the trial, classifier and prognostic notebooks also accept the environment variable `HS_DATA`).
2. Run cells in order. Run times (a normal machine): trial simulation about 5 min, classifier about 2 min, prognostic about 2 min; the forecast notebook is longer (the final report fit adds about 3 min).
3. Forecast notebook: set the test folder and the paths of the answer templates in the runner cell; the four files and the patient report are written to the answers folder.
4. Read the ledger at the end of each notebook before trusting any number.