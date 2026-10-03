# PhD-paper outline (publishable extension of this benchmark)

Title: *Crosstalk Asymmetry Predicts Autotuning Difficulty in Semiconductor Quantum Dots:
a Reproducible Patch-Based Benchmark with Uncertainty-Gated Search*

1. Introduction — scaling bottleneck; DelftX tuning stages; 2026 SOTA (autonomous Ge-Si 10/13,
   U-Net 1015 CSDs 80%, online 95%, NIST <8 min 96%, FPGA 9.8×).
2. Related work — QFlow/QDSD datasets; QDarts/SimCATS/QArray/QDsim/QDFlow/RF-Squad simulators;
   uncertainty autotuning; Hamiltonian learning; ray-based tuning.
3. Methods — honeycomb generator (Sec. physics.py); patch MLP vs logreg baseline;
   closed-loop search with verification; kPCA clustering; Rabi-chevron Hamiltonian fit;
   group-CV protocol.
4. Results — W1 AUC/acc/ECE; W2 success Δ(verify); W3 ARI/NMI; hidden pattern Pearson r;
   W4 MAE(g), MAPE(T2*); compute budget.
5. Hidden-pattern finding (novel) — cross-capacitance asymmetry A correlates negatively
   with per-device success; disorder correlates with line-detector ECE. Propose asymmetry-aware
   scheduler: spend verification budget ∝ Â estimated from first 10 patches.
6. Limitations — synthetic-to-real gap; no barrier-gate dynamics; single sensor model;
   validation on QDSD/QFlow experimental patches as next step (protocol in docs/).
7. Reproducibility — Docker, seeds, live manifest, MIT code.
8. References — Kalantre 2019; Moon 2020; Zwolak QFlow 2018/2022; Yon 2024/2025;
   Samaha 2026; Schuff/Nature Electronics 2026; Zubchenko 2025; Hader SimCATS 2024;
   Krzywda QDarts 2024/25; Muto 2026; Kondo MBRL 2025; Greplova/DelftX course.

Target venues: PR Applied, MLST, IEEE TQE, SciPost Phys Codebases.
