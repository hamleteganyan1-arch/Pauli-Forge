# Source map (attached to this code package)

## Primary proofs (provided by user / project; interpret strictly within hypotheses)
- `references/Ts_Adaptive_Multiscale_Physical_Atlas_Proof_RU_2026-10-15.md`: weighted localization and atlas.
- `references/Ts_D1C_Certified_Weighted_Labels_Full_H1_Proof_RU_2026-10-15.md`: qualitative interval labels / exact density construction.
- `references/Ts_Polynomial_Catalog_Range_Entropy_and_Decoder_RU_2026-10-15.md`: value bank not full decoder.
- `references/Ts_Direct_Decoder_LogLip_Gram_SDP_Proof_RU_2026-10-18.md`: *conditional* polynomial decoder for log-Lipschitz regularity/moment access.
- `references/Ts_M1_Fifteen_Routes_Aggregate_Decoder_Proof_RU_2026-10-19.md`: error aggregation without per-block success.
- `references/Ts_Spectral_Excess_Overlap_Repair_RU_2026-10-15.md`: occupancy cap repair with energy conditions.
- `references/Ts_M1_Proofs_2026-10-21.md`: quantitative H1 and discrete graph solver with explicit scope.
- `references/physical_smoothing_uploaded.md`: endpoint regularity and M1 smoothing progress.
- `references/analysis_uploaded.md`: 2026-10-21 project research audit (note: some literature claims still need independent verification).

Not all library documents could be materialized into the package; `AGENTS.md` lists those missing by exact name. Their absence is not a license to assert their theorems without access.

## Official upstream frameworks
- STRUCTURES25 https://github.com/sciai-lab/structures25 (ML OF-DFT, LGPL-3.0)
- e3nn https://github.com/e3nn/e3nn (E(3) tensors)
- MACE https://github.com/ACEsuit/mace (equivariant message passing; not itself a density KEDF)
- PySCF https://github.com/pyscf/pyscf (reference data)

## Scientific evidence policy
Every implemented assertion is classified as proved math under assumptions, proof-conditional reduction, numerical finite basis, learned empirical, or open. No `epsilon*N` certification without both independently valid local intervals and global tail/localization control. Analytical Pauli positivity alone is NOT the exact matrix exclusion theorem for scalar ML predictions.
