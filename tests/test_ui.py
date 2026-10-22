"""Console checks exercise actual numerical payloads and Streamlit rendering."""

from pathlib import Path

import pytest

from pauliforge.ui import (
    EVIDENCE_ROWS,
    coarse_blocks,
    density_figure,
    make_demo_payload,
    matrix_demo,
    summarize_payload,
)


def test_console_payload_and_blocks_are_honestly_tagged():
    payload = make_demo_payload()
    summary = summarize_payload(payload)
    assert summary["N"] == pytest.approx(3)
    assert summary["TW"] + summary["Pauli"] == pytest.approx(summary["Ts"])
    assert summary["physical_ground_state_verified"] is False
    assert summary["continuum_certified"] is False
    blocks = coarse_blocks(payload)
    assert sum(row["mass_estimate"] for row in blocks) == pytest.approx(3)
    assert sum(row["TW_estimate"] for row in blocks) == pytest.approx(summary["TW"])
    assert all(not row["certified_interval"] for row in blocks)
    assert any(row["Status"] == "OPEN" for row in EVIDENCE_ROWS)


def test_density_view_contains_samples_and_nuclei():
    pytest.importorskip("plotly")
    figure = density_figure(make_demo_payload())
    assert len(figure.data) == 2
    assert len(figure.data[0].x) > 100
    assert len(figure.data[1].x) == 3


def test_console_joint_matrix_example():
    pytest.importorskip("cvxpy")
    result = matrix_demo()
    assert result["constraint_residual"] < 1e-4
    assert result["label_quality"] == "numerical_finite_basis_only"


def test_streamlit_default_app_renders_without_exceptions():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    import pauliforge.ui

    app = AppTest.from_file(str(Path(pauliforge.ui.__file__))).run(timeout=25)
    assert not list(app.exception)
    assert app.title[0].value == "Physics Research Console"
    app.sidebar.radio[0].set_value("Certificates").run(timeout=25)
    assert not list(app.exception)
    assert len(app.dataframe) > 0


def test_console_pages_and_joint_matrix_interaction():
    pytest.importorskip("streamlit")
    pytest.importorskip("cvxpy")
    from streamlit.testing.v1 import AppTest

    import pauliforge.ui

    app = AppTest.from_file(str(Path(pauliforge.ui.__file__))).run(timeout=25)
    assert not list(app.exception)
    assert any("EMPIRICAL trial-density" in item.value for item in app.warning)
    for page in ("Domain D_E", "Atlas", "Training", "Validation", "Papers"):
        app.sidebar.radio[0].set_value(page).run(timeout=25)
        assert not list(app.exception), page
        if page == "Training":
            assert any("No training history" in item.value for item in app.info)
            assert not app.get("plotly_chart")
        elif page == "Atlas":
            assert len(app.dataframe) == 1
            assert any("Numerical octant overview only" in item.value for item in app.info)
        elif page == "Validation":
            assert '"continuum_certified": false' in app.json[0].value
    app.sidebar.radio[0].set_value("Pauli matrices").run(timeout=25)
    assert any("Independent two-point" in item.value for item in app.caption)
    app.button[0].click().run(timeout=25)
    assert not list(app.exception)
    assert len(app.dataframe) == 1
    occupations = app.dataframe[0].value["occupation"]
    assert occupations.min() >= -1e-7
    assert occupations.max() <= 2 + 1e-7
