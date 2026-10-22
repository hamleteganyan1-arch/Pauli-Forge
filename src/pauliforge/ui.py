"""Optional Physics Research Console, with explicit evidence and empty states.

The importable helpers require only core dependencies.  Streamlit and Plotly
are imported when their UI functions are invoked, never by model inference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from pauliforge.domain import FrozenDE, Geometry
from pauliforge.matrix import check_pauli_matrix
from pauliforge.model import PauliKEDF

EVIDENCE_ROWS = [
    {
        "Requirement": "Scalar E(3), permutation and scaling relations",
        "Status": "CONDITIONAL",
        "Evidence": "Architectural relations; discretization and quadrature remain separate",
    },
    {
        "Requirement": "Pauli = 0 for N ≤ 2",
        "Status": "CONDITIONAL",
        "Evidence": "Exact model branch in its spin-summed ensemble sector",
    },
    {
        "Requirement": "Joint density / matrix constraints",
        "Status": "NUMERICAL",
        "Evidence": "Finite-basis primal/dual diagnostics; scalar ML positivity is insufficient",
    },
    {
        "Requirement": "Physical ground-state density membership",
        "Status": "OPEN",
        "Evidence": "FrozenDE geometry checks do not establish ground-state provenance",
    },
    {
        "Requirement": "Universal polynomial Coulomb decoding",
        "Status": "OPEN",
        "Evidence": "Only explicitly supported subclasses may use a decoder guarantee",
    },
    {
        "Requirement": "ML accuracy and uncertainty",
        "Status": "EMPIRICAL",
        "Evidence": "Report data and split; no confidence score becomes a rigorous interval",
    },
]


def make_demo_payload(seed=17):
    """Deterministic synthetic trial-density example, not a physical benchmark."""
    torch.manual_seed(seed)
    dtype = torch.float64
    axis = torch.linspace(-5, 5, 9, dtype=dtype)
    coords = torch.cartesian_prod(axis, axis, axis)
    weights = torch.full((len(coords),), float((axis[1] - axis[0]) ** 3), dtype=dtype)
    raw = torch.exp(-0.5 * coords.square().sum(-1))
    rho = 3.0 * raw / (raw * weights).sum()
    grad_u = -0.5 * coords * rho.sqrt()[:, None]
    geometry = Geometry(
        (1, 1, 1), torch.tensor([[-1.5, 0, 0], [0, 0, 0], [1.5, 0, 0]], dtype=dtype), spin=1
    )
    domain = FrozenDE()
    domain.validate(geometry)
    model = PauliKEDF(hidden=32).double().eval()
    with torch.no_grad():
        prediction = model(coords, weights, rho, grad_u)
    return {
        "coords": coords,
        "weights": weights,
        "rho": rho,
        "grad_u": grad_u,
        "geometry": geometry,
        "domain": domain,
        "prediction": prediction,
        "history": [],
        "mode": "kinetic_hybrid",
        "model": model,
        "provenance": "synthetic_trial_density_not_a_Coulomb_ground_state",
        "status": "EMPIRICAL",
        "continuum_certified": False,
    }


def summarize_payload(payload):
    out = payload["prediction"]
    summary = {
        "mode": payload["mode"],
        "N": float(out.n),
        "domain_version": payload["domain"].version,
        "domain_hash": payload["domain"].domain_hash,
        "provenance": payload["provenance"],
        "status": payload["status"],
        "physical_ground_state_verified": False,
        "continuum_certified": False,
    }
    if payload["mode"] == "kinetic_hybrid":
        summary.update(
            TW=float(out.tw),
            Pauli=float(out.pauli),
            Ts=float(out.ts),
            analytical_pauli_floor=float(out.analytical_pauli_floor),
            learned_pauli=float(out.learned_pauli),
            kinetic_operator="-Delta",
        )
    else:
        summary.update(
            Exc_hartree=float(out.exc),
            exchange_hartree=float(out.exchange),
            correlation_hartree=float(out.correlation),
        )
    return summary


def _numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def density_figure(payload):
    import plotly.graph_objects as go

    xyz, rho = _numpy(payload["coords"]), _numpy(payload["rho"])
    # Deterministic thinning keeps large QC grids usable without changing any
    # density integration or energy calculation.
    keep = np.linspace(0, len(xyz) - 1, min(len(xyz), 6000)).astype(int)
    xyz, rho = xyz[keep], rho[keep]
    figure = go.Figure(
        go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode="markers",
            name="Density samples",
            marker={
                "size": 2 + 5 * np.sqrt(rho / max(float(rho.max()), 1e-30)),
                "color": rho,
                "colorscale": "Ice",
                "opacity": 0.45,
                "colorbar": {"title": "ρ [e / Bohr³]"},
            },
            hovertemplate="x=%{x:.2f}, y=%{y:.2f}, z=%{z:.2f}<br>ρ=%{marker.color:.4g}<extra></extra>",
        )
    )
    geometry = payload["geometry"]
    positions = _numpy(geometry.positions_bohr)
    symbols = {1: "H", 6: "C", 7: "N", 8: "O"}
    figure.add_trace(
        go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode="markers+text",
            text=[symbols.get(z, str(z)) for z in geometry.nuclear_charges],
            textposition="top center",
            name="Nuclei",
            marker={"size": 8, "color": "#fbbf24", "line": {"color": "#fff", "width": 1}},
        )
    )
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0c1728",
        plot_bgcolor="#0c1728",
        height=530,
        margin={"l": 0, "r": 0, "t": 15, "b": 0},
        scene={
            "aspectmode": "data",
            "xaxis_title": "x [Bohr]",
            "yaxis_title": "y [Bohr]",
            "zaxis_title": "z [Bohr]",
        },
    )
    return figure


def coarse_blocks(payload):
    """Numerical octant statistics for viewing, never a local certificate."""
    coords = payload["coords"]
    center = (coords * (payload["weights"] * payload["rho"])[:, None]).sum(0) / (
        payload["weights"] * payload["rho"]
    ).sum()
    groups = (coords >= center).to(torch.int64)
    identifiers = groups[:, 0] + 2 * groups[:, 1] + 4 * groups[:, 2]
    rows = []
    for index in range(8):
        mask = identifiers == index
        mass = float((payload["weights"][mask] * payload["rho"][mask]).sum())
        tw = float((payload["weights"][mask] * payload["grad_u"][mask].square().sum(-1)).sum())
        rows.append(
            {
                "block": f"octant-{index}",
                "mass_estimate": mass,
                "TW_estimate": tw,
                "status": "NUMERICAL",
                "certified_interval": False,
            }
        )
    return rows


def matrix_demo():
    """A separate, explicitly joint two-point finite-basis constrained search."""
    from pauliforge.teachers.matrix_sdp import solve_finite_basis

    return solve_finite_basis(
        np.array([1.5, 1.5]), np.eye(2), np.ones(2), np.array([[1.0, -0.3], [-0.3, 2.0]]), 3.0
    )


def run_console():
    try:
        import plotly.graph_objects as go
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError("Install the optional UI dependencies: pip install -e '.[ui]'") from exc

    st.set_page_config(
        page_title="PauliForge · Physics Research Console", page_icon="⚛", layout="wide"
    )
    st.markdown(
        """<style>
      .stApp {background:#07111f;color:#dce9f8} [data-testid=stSidebar]{background:#0c1728}
      [data-testid=stMetric] {background:#102137;padding:18px;border:1px solid #20364f;border-radius:12px}
      h1,h2,h3 {letter-spacing:-.025em} .pf-kicker {color:#38bdf8;letter-spacing:.18em;font-size:.8rem}
    </style>""",
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.markdown("## ⚛ PauliForge")
        st.caption("PHYSICS RESEARCH CONSOLE")
        page = st.radio(
            "Workspace",
            [
                "Density",
                "Domain D_E",
                "Pauli matrices",
                "Atlas",
                "Training",
                "Certificates",
                "Validation",
                "Papers",
            ],
        )
        checkpoint_path = st.text_input("Optional local checkpoint", "")
        density_path = st.text_input("Optional local reference .npz", "")
        st.caption("Local files only. No orbitals are used during ML prediction.")
    payload = make_demo_payload()
    if checkpoint_path or density_path:
        if not checkpoint_path or not density_path:
            st.info("Supply both a checkpoint and a reference density to inspect a trained model.")
        else:
            try:
                from pauliforge.training import load_checkpoint, read_npz

                loaded = load_checkpoint(checkpoint_path)
                sample = read_npz(Path(density_path), domain=loaded.domain, mode=loaded.config.mode)
                with torch.no_grad():
                    prediction = loaded.model(
                        sample["coords"], sample["weights"], sample["rho"], sample["grad_u"]
                    )
                payload = dict(
                    sample,
                    model=loaded.model,
                    domain=loaded.domain,
                    prediction=prediction,
                    mode=loaded.config.mode,
                    history=loaded.metadata["history"],
                    provenance=sample["metadata"].get("provenance", "precomputed_reference"),
                    status="EMPIRICAL",
                    continuum_certified=False,
                )
            except (ValueError, RuntimeError, OSError, KeyError) as exc:
                st.error(f"Reference could not be loaded: {exc}")
                st.stop()
    summary = summarize_payload(payload)
    st.markdown(
        '<div class="pf-kicker">DENSITY-ONLY · ORBITAL-FREE · EVIDENCE FIRST</div>',
        unsafe_allow_html=True,
    )
    st.title("Physics Research Console")
    st.caption("PauliForge v1 research implementation · atomic units · explicit scientific scope")
    st.warning(
        "EMPIRICAL trial-density demo. No ground-state, universal-accuracy or continuum certificate is asserted."
        if not checkpoint_path
        else "EMPIRICAL model prediction. Reference provenance and mathematical certificates are separate."
    )
    columns = st.columns(4)
    columns[0].metric("Electrons N", f"{summary['N']:.6f}")
    if payload["mode"] == "kinetic_hybrid":
        columns[1].metric("T_W · −Δ convention", f"{summary['TW']:.6f}")
        columns[2].metric("Pauli · ML point", f"{summary['Pauli']:.6f}")
        columns[3].metric("T_s · ML point", f"{summary['Ts']:.6f}")
    else:
        columns[1].metric("E_xc [Hartree]", f"{summary['Exc_hartree']:.6f}")
        columns[2].metric("Exchange [Hartree]", f"{summary['exchange_hartree']:.6f}")
        columns[3].metric("Correlation [Hartree]", f"{summary['correlation_hartree']:.6f}")
    if page == "Density":
        left, right = st.columns([2.3, 1])
        with left:
            st.subheader("Electronic density and nuclei")
            st.plotly_chart(density_figure(payload), width="stretch")
        with right:
            st.subheader("Energy decomposition")
            if payload["mode"] == "kinetic_hybrid":
                st.dataframe(
                    [
                        {"Component": key, "Value": summary[key]}
                        for key in ("TW", "analytical_pauli_floor", "learned_pauli", "Ts")
                    ],
                    hide_index=True,
                )
            else:
                st.dataframe(
                    [
                        {"Component": key, "Hartree": summary[key]}
                        for key in ("exchange_hartree", "correlation_hartree", "Exc_hartree")
                    ],
                    hide_index=True,
                )
            st.caption(
                "Positivity of the Pauli point estimate does not prove that a joint admissible Γ exists."
            )
            st.code(payload["provenance"], language=None)
    elif page == "Domain D_E":
        st.subheader("Frozen chemical space")
        st.json(payload["domain"].to_dict())
        st.code(payload["domain"].domain_hash, language=None)
        st.info(
            "Geometry, charge, spin and electron count are checked. Ground-state membership is a separate provenance requirement; this demo belongs to D_trial."
        )
    elif page == "Pauli matrices":
        st.subheader("Joint finite-basis matrix diagnostic")
        st.caption(
            "Independent two-point reference example, not a matrix representation of the displayed ML density."
        )
        if st.button("Solve joint matrix example"):
            result = matrix_demo()
            gamma = np.asarray(result["gamma"])
            eigenvalues = np.linalg.eigvalsh(gamma)
            valid = check_pauli_matrix(torch.tensor(gamma), 3.0)
            st.plotly_chart(go.Figure(go.Heatmap(z=gamma, colorscale="Ice")), width="stretch")
            st.dataframe(
                {
                    "occupation": eigenvalues,
                    "lower cap": [0.0] * len(eigenvalues),
                    "upper cap": [2.0] * len(eigenvalues),
                }
            )
            st.write(
                {
                    "matrix_constraints_numerically_satisfied": valid,
                    "trace": float(np.trace(gamma)),
                    "sampled_density_residual": result["constraint_residual"],
                    "status": "NUMERICAL_FINITE_BASIS",
                    "continuum_certified": False,
                }
            )
    elif page == "Atlas":
        st.subheader("Energy-active block overview")
        rows = coarse_blocks(payload)
        st.plotly_chart(
            go.Figure(
                go.Bar(
                    x=[r["block"] for r in rows],
                    y=[r["mass_estimate"] for r in rows],
                    marker_color="#38bdf8",
                )
            ),
            width="stretch",
        )
        st.dataframe(rows, hide_index=True)
        st.info(
            "Numerical octant overview only. A certified adaptive atlas additionally needs full local diagonals, source-valid intervals, localization and tail budgets."
        )
    elif page == "Training":
        st.subheader("Recorded optimization history")
        history = payload["history"]
        if not history:
            st.info(
                "No training history loaded. Train a model and load its checkpoint to display actual results."
            )
        else:
            figure = go.Figure(
                go.Scatter(
                    x=[row["epoch"] for row in history],
                    y=[row["train_mse_per_electron"] for row in history],
                    mode="lines+markers",
                    name="Training MSE / electron",
                    line_color="#38bdf8",
                )
            )
            figure.update_layout(
                template="plotly_dark", yaxis_title="Empirical MSE", xaxis_title="Epoch"
            )
            st.plotly_chart(figure, width="stretch")
            st.dataframe(history, hide_index=True)
        st.caption(
            "A good energy fit does not establish functional derivative or variational accuracy."
        )
    elif page == "Certificates":
        st.subheader("Certificate explorer")
        st.dataframe(EVIDENCE_ROWS, hide_index=True)
        st.info(
            "No rigorous [L,U] interval is attached to this ML prediction. A global certificate is unavailable until all local proofs and error budgets are verified."
        )
    elif page == "Validation":
        st.subheader("Validation record")
        st.json(summary)
        st.dataframe(EVIDENCE_ROWS, hide_index=True)
    else:
        st.subheader("Research sources")
        references = Path(__file__).resolve().parents[2] / "references"
        if references.exists():
            st.dataframe(
                [
                    {"Source": path.name, "Availability": "LOCAL_SOURCE"}
                    for path in sorted(references.iterdir())
                    if path.is_file()
                ],
                hide_index=True,
            )
        else:
            st.info("Source documents are distributed with the research checkout.")
        st.caption(
            "Unavailable theorem sources remain BLOCKED; numerical experiments do not supply missing proofs."
        )
    st.divider()
    st.caption(
        "PROVED / CONDITIONAL / NUMERICAL / EMPIRICAL / OPEN / BLOCKED · no status is communicated by color alone"
    )


if __name__ == "__main__":
    run_console()
