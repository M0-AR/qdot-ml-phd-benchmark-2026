# Best practices applied (2026–2027 literature consensus)

This repo implements the DelftX "Machine Learning for Semiconductor Quantum Devices"
 Weeks 0–5 as a reproducible benchmark, following 2024–2026 SOTA lessons:

## 1. No device leakage (Samaha et al. IOP 2026; Yon et al. MLST 2024)
- Group K-fold by physical `device_id`. All patches from one device stay in one fold.
- Without grouping, sensor-coupling/defect memorisation inflates accuracy by ~5–15 pp.

## 2. Uncertainty-gated exploration (Yon et al. Nano Lett 2025, 95% online, 20 runs)
- `predict_proba` confidence; threshold 0.6 triggers 4-neighbour verification scan.
- Ablation `verify=True/False` quantifies the robustness gain (expect +5–12 pp, cf. >10% in Yon 2024).

## 3. Ray/patch-efficient measurement (NIST bootstrapping <8 min 96%; FPGA RF 9.8×)
- 28×28 patches, diagonal walk, early stop at target charge; count steps + verifications.
- Full-diagram U-Net (1015 CSDs, 80.0%, peak 88%) is the offline upper bound; patch search is the online budget.

## 4. Simulator choice (QDarts SciPost Codeb. 43; SimCATS IEEE TQE 2024; QArray; QDsim; QDFlow; RF-Squad JAX)
- From-scratch constant-interaction honeycomb + sensor + white/1f/telegraph/latching noise.
- Parameter ranges match FD-SOI/SiGe literature; `configs/experiment.yaml` pins every RNG seed.

## 5. Public data grounding (QDSD GitHub 3it-inpaqt; NIST QFlow lite/2.0 PDR 66492819760D3FF6E05324570681BA721894)
- `live_verify.py` fetches live GitHub/PyPI/PDR metadata with UTC timestamp into `results/live_manifest.json`.
- Bulk experimental binaries are NOT vendored (licence/size); synthetic data is calibrated to the same format so models transfer.

## 6. Unsupervised honesty (DelftX W3 kernel-PCA)
- kernel-PCA (RBF) + KMeans + DBSCAN; report ARI/NMI vs hidden charge labels + noise fraction.
- Hidden-pattern mining: Pearson r of crosstalk asymmetry/disorder vs per-device success (pre-registered H1/H2).

## 7. Hamiltonian learning (W4; EDSR/Rabi chevron)
- Fit g-factor slope + exchange intercept + T2* envelope; report MAE/MAPE over 40 draws.
- Separates `t_c` (exchange) from Zeeman — the fine-tuning quantity of interest.

## 8. Reproducibility (Docker + pinned config + pytest)
- `docker compose up qdotml` reproduces `results/benchmark.json`; `docker compose run test` runs unit tests.
- One-seed determinism; group-CV means ± spread reported, not point estimates.

## 9. Scope clarification
- Verification is on **device data** (synthetic CSDs + public QD/QFlow manifests).
- Financial "market/live-trading data" does not apply to qubit tuning and is explicitly out of scope.
