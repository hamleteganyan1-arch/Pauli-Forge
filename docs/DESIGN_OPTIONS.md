# Two interface styles (project/product design)

## A — Physics Research Console (recommended first)
- Dark graphite, navy panels, amber for unverified/gap, emerald only for certified checks, electric cyan for neutral scientific values.
- Left navigation: Domain DE / Density / Pauli Matrices / Atlas / Training / Certificates / Validation / Papers.
- Home scoreboard: allowed atoms, nuclear radius/separation, N, energy TW, Pauli lower/upper interval, epsilon*N budget and PASS/OPEN tags.
- Main canvas: 3D electron-density isosurface + atomic geometry. Side pane: multiscale cubes whose width represents alpha_Q; clicking shows local primal/dual gap, selected decoder, interval and source theorem.
- Evidence-first status: proven / conditional / empirical / open, provenance of each label, input convention and Git commit.

## B — Atlas Studio
- White warm background, indigo and violet bands for scale; diagram-first experiment builder.
- Interactive pipeline: Geometry K -> density u -> adaptive cubes -> Pauli primal/dual -> ML residual -> summed energy.
- Hover local cube: rho mass, H1 budget, interval gap, last teacher / ML step, cost and uncertainty.
- Subproject comparison with reversible toggle between weighted packet / Gram-SDP / sparse graph decoders.
- Publication mode: export reproducible figs, equation/assumption cards, convergence curves and machine-readable certificate reports.

## Shared design rules
- Do NOT label model outputs "certified" because they are positive or have passed numerical KS tests.
- Compare baseline vs trained model vs certified interval with visually separate columns.
- Invariance dashboard: translations/rotations/reflections/permutations/scalings, dissociation, occupied orbital limits.
- Accessibility: color is never the sole indication of mathematical status.
