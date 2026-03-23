"""
Streamlit page for Optuna hyperparameter optimisation (HPO) tracking.

Pulls a live or completed Optuna study from a SQLite / MySQL storage backend
and renders four interactive analysis views:

* **Parallel Coordinates** — All trials as lines across all hyperparameter axes.
* **Parameter Importance** — FANOVA-based importance scores with Slice plots for
  the top-4 most influential parameters.
* **Optimisation History** — Objective value over trial number, with rolling best.
* **Contour Plots** — 2-D contour surfaces for pairs of hyperparameters.

Integration into the existing dashboard::

    # In logic/src/ui/app.py add a new mode branch:
    elif mode == "hpo_tracker":
        render_hpo_tracker()

    # In logic/src/ui/components/sidebar.py, add "hpo_tracker" to the
    # mode selector options list.

Standalone usage::

    from logic.src.ui.pages.hpo_tracker import render_hpo_tracker
    render_hpo_tracker(
        storage_url="sqlite:///assets/hpo/study.db",
        study_name="vrpp_am_sweep",
    )
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

try:
    import optuna
    import optuna.visualization as ov
except ImportError:
    optuna = None
    ov = None

# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------


def render_hpo_tracker(
    storage_url: str = "sqlite:///assets/hpo/study.db",
    study_name: Optional[str] = None,
    n_top_trials: int = 10,
    show_parallel_coords: bool = True,
    show_importance: bool = True,
    show_history: bool = True,
    show_contour: bool = False,
    importance_evaluator: str = "fanova",
    height: int = 600,
) -> None:
    """
    Render the full HPO tracking dashboard in Streamlit.

    Connects to an Optuna study and renders interactive optimisation views.
    All chart dimensions and displayed study names are driven by the function
    arguments so there are no hardcoded values in the render logic.

    Args:
        storage_url: SQLAlchemy connection string for the Optuna storage backend.
            SQLite example: ``"sqlite:///assets/hpo/study.db"``.
            MySQL example:  ``"mysql://user:pass@host/db"``.
        study_name: Name of the study to visualise. When None, a study picker
            dropdown is shown if multiple studies exist in storage.
        n_top_trials: Number of best trials highlighted in the trial table.
            Default 10.
        show_parallel_coords: Render parallel coordinate plot. Default True.
        show_importance: Render FANOVA importance + slice plots. Default True.
        show_history: Render optimisation history curve. Default True.
        show_contour: Render 2-D contour plots (expensive for many params).
            Default False.
        importance_evaluator: ``"fanova"`` (default) or
            ``"mean_decrease_impurity"`` for the importance algorithm.
        height: Chart height in pixels for all plots. Default 600.
    """
    if optuna is None or ov is None:
        st.error("**optuna** is required for HPO tracking. Run: `pip install optuna plotly`")
        return

    st.title("Hyperparameter Optimisation Tracker")

    # ── Storage info bar ─────────────────────────────────────────────────────
    with st.expander("Storage settings", expanded=False):
        storage_url = st.text_input(
            "Optuna Storage URL",
            value=storage_url,
            key="hpo_storage_url",
            help="SQLAlchemy connection string for your Optuna storage backend.",
        )

    # ── Load study ───────────────────────────────────────────────────────────
    study = _load_study(storage_url, study_name, optuna)
    if study is None:
        return

    trials = study.trials
    completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    n_complete = len(completed)

    # ── Header KPIs ──────────────────────────────────────────────────────────
    _render_hpo_kpis(study, trials, n_complete, optuna)

    if n_complete < 2:
        st.info(f"At least 2 completed trials are required for visualisations (found {n_complete}).")
        _render_trial_table(completed, n_top_trials)
        return

    # ── Build tab list dynamically ───────────────────────────────────────────
    tab_labels: List[str] = []
    if show_parallel_coords:
        tab_labels.append("Parallel Coordinates")
    if show_importance:
        tab_labels.append("Parameter Importance")
    if show_history:
        tab_labels.append("Optimisation History")
    if show_contour:
        tab_labels.append("Contour Plots")
    tab_labels.append("Trial Table")

    tabs = st.tabs(tab_labels)
    t_idx = 0

    if show_parallel_coords:
        with tabs[t_idx]:
            _render_tab_parallel_coords(study, ov, height)
        t_idx += 1

    if show_importance:
        with tabs[t_idx]:
            _render_tab_importance(study, ov, height, importance_evaluator)
        t_idx += 1

    if show_history:
        with tabs[t_idx]:
            _render_tab_history(study, ov, height)
        t_idx += 1

    if show_contour:
        with tabs[t_idx]:
            _render_tab_contour(study, ov, height)
        t_idx += 1

    with tabs[t_idx]:
        _render_trial_table(completed, n_top_trials)


def _render_hpo_kpis(study: Any, trials: List[Any], n_complete: int, optuna: Any) -> None:
    """Render the header KPI metrics."""
    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Study", study.study_name)
    kpi_cols[1].metric("Total Trials", len(trials))
    kpi_cols[2].metric("Completed", n_complete)
    kpi_cols[3].metric("Best Value", f"{study.best_value:.6f}" if n_complete > 0 else "N/A")


def _render_tab_parallel_coords(study: Any, ov: Any, height: int, *args) -> None:
    """Render the Parallel Coordinates tab."""
    st.subheader("Parallel Coordinate Plot")
    st.caption("Each coloured line represents one trial. Lines are coloured by objective value.")
    try:
        fig = ov.plot_parallel_coordinate(study)
        _apply_layout(fig, height)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.error(f"Parallel coordinate plot failed: {exc}")


def _render_tab_importance(study: Any, ov: Any, height: int, evaluator_name: str) -> None:
    """Render the Parameter Importance tab."""

    st.subheader("Hyperparameter Importance")
    st.caption(f"Importance scores via **{evaluator_name}**. Higher bars = greater influence on the objective.")
    try:
        evaluator = _build_evaluator(evaluator_name, optuna)
        fig_imp = ov.plot_param_importances(study, evaluator=evaluator)
        _apply_layout(fig_imp, height)
        st.plotly_chart(fig_imp, use_container_width=True)

        importances: Dict[str, float] = optuna.importance.get_param_importances(study, evaluator=evaluator)
        top_params = list(importances.keys())[:4]
        if top_params:
            st.subheader(f"Slice Plots — Top {len(top_params)} Parameters")
            fig_slice = ov.plot_slice(study, params=top_params)
            _apply_layout(fig_slice, height)
            st.plotly_chart(fig_slice, use_container_width=True)
    except Exception as exc:
        st.error(f"Importance plot failed: {exc}")


def _render_tab_history(study: Any, ov: Any, height: int, *args) -> None:
    """Render the Optimisation History tab."""
    st.subheader("Optimisation History")
    st.caption("Objective value per trial. The green line shows the rolling best.")
    try:
        fig_hist = ov.plot_optimization_history(study)
        _apply_layout(fig_hist, height)
        st.plotly_chart(fig_hist, use_container_width=True)
    except Exception as exc:
        st.error(f"History plot failed: {exc}")


def _render_tab_contour(study: Any, ov: Any, height: int, *args) -> None:
    """Render the Contour Plots tab."""
    st.subheader("2-D Contour Plots")
    st.caption("Objective surface over pairs of hyperparameters. Select parameters via the Optuna figure controls.")
    try:
        fig_cont = ov.plot_contour(study)
        _apply_layout(fig_cont, int(height * 1.4))
        st.plotly_chart(fig_cont, use_container_width=True)
    except Exception as exc:
        st.error(f"Contour plot failed: {exc}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_study(
    storage_url: str,
    study_name: Optional[str],
    optuna: Any,
) -> Optional[Any]:
    """Load an Optuna study from storage with graceful error handling."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    try:
        if study_name:
            return optuna.load_study(study_name=study_name, storage=storage_url)

        # No study name provided — enumerate and offer a picker
        summaries = optuna.study.get_all_study_summaries(storage=storage_url)
        if not summaries:
            st.warning(f"No studies found in storage: `{storage_url}`")
            return None

        names = [s.study_name for s in summaries]
        chosen = st.selectbox(
            "Select Study",
            names,
            key="hpo_study_selector",
        )
        return optuna.load_study(study_name=chosen, storage=storage_url)

    except Exception as exc:
        st.error(f"Failed to load study from `{storage_url}`: **{exc}**")
        st.info("Ensure the storage file exists and Optuna is installed. Run: `pip install optuna sqlalchemy`")
        return None


def _build_evaluator(name: str, optuna: Any) -> Any:
    """Instantiate the requested FANOVA/MDI importance evaluator."""
    if name == "fanova":
        return optuna.importance.FanovaImportanceEvaluator()
    return optuna.importance.MeanDecreaseImpurityImportanceEvaluator()


def _apply_layout(fig: Any, height: int) -> None:
    """Apply consistent Plotly layout settings to all Optuna figures."""
    fig.update_layout(
        height=height,
        template="plotly_white",
        margin=dict(l=60, r=40, t=60, b=60),
    )


def _render_trial_table(
    completed_trials: List[Any],
    n_top: int,
) -> None:
    """Render the top-N completed trials as an interactive DataFrame."""

    st.subheader(f"Top {n_top} Completed Trials")

    if not completed_trials:
        st.info("No completed trials to display.")
        return

    sorted_trials = sorted(
        completed_trials,
        key=lambda t: t.value if t.value is not None else float("inf"),
    )

    rows: List[Dict[str, Any]] = []
    for rank, trial in enumerate(sorted_trials[:n_top], start=1):
        row: Dict[str, Any] = {
            "Rank": rank,
            "Trial #": trial.number,
            "Objective": round(trial.value, 6) if trial.value is not None else None,
            "Duration (s)": (
                round((trial.datetime_complete - trial.datetime_start).total_seconds(), 1)
                if trial.datetime_complete and trial.datetime_start
                else None
            ),
        }
        for param_name, param_val in trial.params.items():
            row[param_name] = round(param_val, 5) if isinstance(param_val, float) else param_val
        rows.append(row)

    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )
