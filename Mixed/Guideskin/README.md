# Multimodal skin characterisation explorer (simulated data)

A demo for multimodal characterisation analysis use to define which skin descriptors related the most to a molecule's skin permeation performance under different skin state (in vivo vs ex vivo, healthy vs wounded). 
> all data are simulated with reference. The notebook just to show a simplified pipeline (preprocessing)
The proposed framework is available in `Main_framework.png`, the explored application idea is available in `Probable_application.png`.

## Use of AI assistance

AI assistant (Claude) was used for building the simulators and modelling codes. The author is responsible for the scientific and analytical decisions and does:
- the choice of which data, literature values and parameters to use;
- verification of the simulator against the reference numbers;
- debugging and quality control of every cell and result;
- the analysis, the interpretation and the statement of limitations.

## What it does

| # | Section | Output |
|---|---|---|
| 1 | Raw data generation | Simulated Raman spectra and biophysical probes from hidden skin properties |
| 2 | Preprocessing | Cleaned, normalised spectra |
| 3 | Raman descriptor extraction | Table of band descriptors, plus validity checks |
| 4 | Multi-block data fusion | Spectra, descriptors and biophysics in one frame, plus heatmap |
| 5 | PCA | Unsupervised structure (single block and multi-block) |
| 6 | Baseline model | As a model comparator for the PLS. |
| 7 | Flux model ladder | PLS models, scored on a blind test |

## The simulated study

- **Groups:** `invivo_healthy`, `exvivo_healthy`, `exvivo_burnt` (training) and `invivo_wounded` (blind test only).
- **Sites:** body, arm, leg.
- **Size:** [customizable, but I use 30] donors × 3 sites × 4 groups
- **Hidden properties**: lipid order (0–1), hydration, sebum, TEWL and a donor fingerprint.
- **Spectra:** Gaussian bands from these properties, plus fluorescence, noise and cosmic spikes.
- **Descriptor references** : available in the notebook (block 3)

## Pipeline

1. **Preprocessing:** despike → Savitzky–Golay smoothing → AsLS baseline → vector normalisation.
2. **Descriptors** are recipes in `DESC_SPEC`, evaluated by `eval_term`. Term types:
   - `H` height, `A` area, `POS` position (sub-grid parabola)
   - `RATIO`, `SUM`, `LIN` (combinations of other terms)
   - 14 targeted descriptors (`TARGETED`) that might follow the flux [based on lipid or water condition] and 10 control descriptors (`DESC_SPEC_CTRL`) that should not follow the flux/negative control [based on protein condition].
3. **Validity checks:** within-group correlation (`within_r`: group means removed first; chance level about |r| ≈ 0.10 for n = 360) and VIP from PLS.
4. **Fusion:** z-scored blocks. Each block gets weight so that its total variance is 1; spectra are centred only.
5. **PCA:** autoscaled, fit on all groups and on training only. Loadings, `relevance` (|loading| × r² of the PC with flux), multi-block PCA with `block_fit` / `block_apply`, PCA by site.
6. **Baseline model:** Minimum model to beat. Needs no spectra and mode, trained from the flux value of the training dataset
7. **Flux ladder:** target = ln flux, scored on the blind wounded set.

## To prevent leakage in the model, do:

- Fit scalers, PCA and PLS on training set only.
- Choose `k` and descriptors inside the cross-validation.
- Split CV by donor (`GroupKFold`), not by piece.
- Never use `true_*` columns or flux as model inputs. (it will include the test dataset too)

## How to run

1. Open `Framework_demo.ipynb` in Google Colab, Jupyter, or Python 3 with `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`. The first code cell installs or imports packages.
2. Run top to bottom. Set `SEED` in section 1 (the default is 678).
3. Section 7 needs these variables: `D`, `desc`, `bio`, `X`, `idx_tr`, `idx_te`, `SITES`, `SEED`, `block_fit`, `block_apply`.
4. For multi-seed figures: change `SEED`, run again down to the ladder cell. Each run appends to `ladder_results.csv`. Delete the file to start fresh.

## Known limitations

- Havent consider the covariates (really forget.)
- only consider the stratum corneum layer.
- Simulated data only with no real instrument behaviour.
- only an assumed flux is used. the real framework should be trained and tested with a combination of formula and actives.
- Wounded pieces share the same 30 donors as training and the spectra carry a donor fingerprint. A real study must hold out whole donors.
- The `k` selection CV score is slightly optimistic (the same folds choose and score `k`).

## Reference
1. Tfayli A. *Caractérisation structurale et moléculaire de la peau par microspectroscopies optiques vibrationnelles.* PhD dissertation, Université de Reims Champagne-Ardenne, 2007. Tables Ch.I.01-I.06, Equ. Ch.I.15, Ch.III Table 01/02.
2. Choe C, Lademann J, Darvin ME. *Analyst* 2016;141:1981-1987. doi:10.1039/C5AN02373D
3. Lunter D, et al. Novel aspects of Raman spectroscopy in skin research. *Exp Dermatol* 2022;31(9):1311-1329. doi:10.1111/exd.14645
4. Caspers PJ, et al. *J Invest Dermatol* 2001;116(3):434-442. doi:10.1046/j.1523-1747.2001.01258.x
5. Egawa M, Hirao T, Takahashi M. *Acta Derm Venereol* 2007;87:4-8. doi:10.2340/00015555-0183
6. Darvin ME, et al. *Biomed Opt Express* 2019;10(6):3092-3103. doi:10.1364/BOE.10.003092