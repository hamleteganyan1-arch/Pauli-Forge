# Source map and scope audit — 2026-10-22

The user supplied the starter research corpus, then five additional sources and
an October 22 upgrade roadmap. All new files are retained byte-for-byte under
`references/`. Exact paths, SHA256 hashes, source aliases and duplicate groups
are recorded in `SOURCE_INVENTORY.json`. These are user-supplied research texts;
numerical tests do not constitute independent theorem proofs or peer review.

## Canonical proof archive now available

[239-page archive](../references/3-Ts_Functional_Strict_Theorem_Archive_2026-10-09-1-.pdf)
restores the previously unavailable fundamental source. Physical PDF pages are
used below; appendix printed-page numbering can differ.

| Physical PDF pages | Result and applicable scope |
| --- | --- |
| 3–4 | Coulomb class and constrained-search definitions, exact diagonal and cap `2I`. |
| 4 | T1 compactness for the specified bounded, collision-free, bound physical ground-state density class; not unrestricted trial densities or arbitrary dissociation. |
| 5 | T2 attainment/lower semicontinuity, exact spin-summed `N<=2` sector and Pauli factor decomposition. |
| 6 | T5 exact-density recovery via coefficient Gram matrices and oscillatory dilation; existence does not provide a general fast SDP decoder. |
| 7–8 | Continuity and finite-observable approximation existence; no polynomial parameter/time estimate follows. |
| 12 | T17/P9-B spectral error `4G²/Lambda`, confined to separated Dirichlet fragments with fixed local electron counts and no interblock charge transfer. |
| 13 | T18 positivity and kinetic coordinate scaling; T19 corrected functional-kernel scaling. |
| 14–17 | Explicit open/conditional registry and algorithmic qualifications. |

Appendix starting physical pages: S1 18, S2 38, S3 51, S4 70, S5 118, S6 151,
S7 173, S8 195, S9 204. Text extraction is available in ignored
`results/source_review/strict_archive.txt` and its hash is in the inventory.

## Existing source-based modules

| Supplied source | Executable module and limitation |
| --- | --- |
| `Ts_Adaptive_Multiscale_Physical_Atlas_Proof_RU_2026-10-15.md` | `atlas.py`: full-diagonal blocks, finite-region adaptive tree, strong 2:1 balancing and robust active superset. Complete continuum tails/input access are not inferred from a finite grid. |
| `Ts_D1C_Certified_Weighted_Labels_Full_H1_Proof_RU_2026-10-15.md` | `teachers/dual.py`: finite-dimensional negative-spectrum Ts dual, especially source §5. This is not the absent quantitative direct Pauli-dual theorem. |
| `Ts_Polynomial_Catalog_Range_Entropy_and_Decoder_RU_2026-10-15.md` | Distinction between polynomial scalar-value bank and the still-unresolved full decoder description is retained. |
| `Ts_Direct_Decoder_LogLip_Gram_SDP_Proof_RU_2026-10-18.md` | `decoders/gram_sdp.py`, `analytic_moments.py`: regular-class assumptions and finite Gram/SDP calculations; numerical results do not certify continuum rounding/truncation. |
| `Ts_M1_Fifteen_Routes_Aggregate_Decoder_Proof_RU_2026-10-19.md` | `certificates.py`: aggregate-error arithmetic with separately checked proof premises. Narrow arbitrary intervals stay uncertified. |
| `Ts_Spectral_Excess_Overlap_Repair_RU_2026-10-15.md` | Occupancy-cap repair reasoning; mass/rank bounds alone are not kinetic-error bounds. |
| `Ts_M1_Proofs_2026-10-21.md` | Quantitative H1/discrete-graph results used only within their stated finite-input scope. |
| `analysis_uploaded.md` | Research audit and explicit disjoint rank-one packet subclass; its summaries do not replace absent primary proofs. |

## Newly supplied smoothing sources

1. `1-Ts_M1_Endpoint_Smoothing_and_Polylog_Bandwidth_Atomic_Algorithm_RU_2026-10-20.md`:
   §2 gives the physical fourth kinetic moment `C N^(5/3)` under its neutral
   Coulomb assumptions; §3 (3.1) gives the explicit conditional bound
   `13 t^a ||u||_{dot H^(1+a)}²`; §4 (4.5) gives the endpoint defect leading to
   `C_phys N^(4/3) sqrt(t)`; §5 concerns **an already supplied rational atomic
   list**, not its production from a physical density. The document explicitly
   does not supply a ready numerical value of `C_phys`. General M1 stays OPEN.
2. `2-Ts_M1_Five_Agent_Sparse_Integration_Search_and_Stability_RU_2026-10-20.md`:
   §3 gives mass-cover search with separately paid oracle cost; §4 computes the
   full heat convolution on a supplied sparse region; §5 (5.2) supplies a
   logarithmic dropped/transported-source stability bound for `W_t`. These are
   Weizsaecker heat-energy results, not a direct Pauli decoder.
3. `4-Ts_Adaptive_Multiscale_Physical_Atlas_Proof_RU_2026-10-15-1-.md` is exactly
   identical to the two atlas copies already present. It adds no independent
   evidence or changed theorem version.
4. `5-M1_Claude_Master_Research_Prompt_RU_2026-10-20.md` is a research handoff.
   Its requested investigations are not additional proved results. In particular,
   §6 flags historical smoothing-exponent conflicts; the available endpoint
   source above contains its explicit stronger derivation and qualifications.
5. `PauliForge_Mathematical_Upgrade_Roadmap_RU_2026-10-22.md` is an architectural
   proposal, explicitly not a proof of full M1. Its cap-I reservoir construction,
   moment-free length and finite-model support banks are candidate development
   routes. Missing primary proofs cited by the roadmap remain missing.

`decoders/smoothing.py` implements the general conditional §3/§5 budget
expressions, exact rational single-atom `W_t=3N/(8t)`, and a small dense
heat-mixture density/gradient diagnostic. It deliberately does **not** claim the
report's interval/polylogarithmic atomic integration algorithm, production of a
short physical atomic list, full Ts or a global epsilon-N certificate.

## Still unavailable exact primary sources

The October 9 archive was searched throughout its 239 pages and does not
reproduce these later titles/theorems:

- `Ts_Polynomial_Weighted_Adaptive_Packet_Decoder_RU_2026-10-18.md`
- `Ts_M1_Decoder_Audit_Exact_Mass_Gradient_9_Slabs_and_Sparse_Theorem_2026-10-20.md`
- `Ts_M1_Pauli_Direct_Dual_Quantitative_Lower_Certificates_2026-10-21.md`

The exact `PAFF`/`Pauli-dual` terms and the three titles are absent from its
extracted text. A later roadmap's citation or a secondary summary does not
supply these proofs. Broader corresponding integrations retain BLOCKED status;
explicit narrow packet, finite matrix and regular Gram implementations remain
usable under their own scopes.

## Official upstream frameworks

See `THIRD_PARTY.md` for reviewed official licenses, pinned upstream revisions
and tested public APIs. e3nn and PySCF are actually exercised. STRUCTURES25 is an
independent benchmark candidate with an incompatible historical dependency
stack; MACE is not substituted for a density functional.
