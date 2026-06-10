from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ufc_predictor.config import settings
from ufc_predictor.models.predict import build_prediction_features, predict_fight


def _read_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _fighter_names(fights: pd.DataFrame, fighters: pd.DataFrame) -> list[str]:
    names: set[str] = set()
    if not fighters.empty and "name" in fighters.columns:
        names.update(fighters["name"].dropna().astype(str).tolist())
    if not fights.empty:
        for column in ["fighter_a", "fighter_b"]:
            if column in fights.columns:
                names.update(fights[column].dropna().astype(str).tolist())
    return sorted(names)


def _load_model():
    try:
        from ufc_predictor.models.train import load_model_bundle

        return load_model_bundle()
    except Exception:
        return None


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="UFC Fight Predictor", page_icon="UFC", layout="wide")
    st.title("UFC Fight Predictor")

    fights = _read_csv(settings.raw_data_dir / "fights.csv")
    fight_stats = _read_csv(settings.raw_data_dir / "fight_stats.csv")
    fighters = _read_csv(settings.raw_data_dir / "fighters.csv")
    scorecards = _read_csv(settings.external_data_dir / "scorecards.csv")
    names = _fighter_names(fights, fighters)

    with st.sidebar:
        fighter_a = st.selectbox("Fighter A", names, index=0 if names else None, placeholder="Select fighter A")
        fighter_b = st.selectbox("Fighter B", names, index=1 if len(names) > 1 else None, placeholder="Select fighter B")
        fight_date = st.date_input("Fight date")
        weight_classes = sorted(fights["weight_class"].dropna().astype(str).unique().tolist()) if "weight_class" in fights else []
        weight_class = st.selectbox("Weight class", weight_classes or ["Lightweight"])
        scheduled_rounds = st.radio("Scheduled rounds", [3, 5], horizontal=True)
        run = st.button("Predict", type="primary", use_container_width=True)

    model = _load_model()
    if run and fighter_a and fighter_b:
        prediction = predict_fight(
            model=model,
            fighter_a=fighter_a,
            fighter_b=fighter_b,
            fight_date=fight_date,
            weight_class=weight_class,
            scheduled_rounds=scheduled_rounds,
            fights=fights,
            fight_stats=fight_stats,
            fighters=fighters,
            scorecards=scorecards,
        )
        st.subheader(prediction["predicted_winner"])
        left, right, tier = st.columns([2, 2, 1])
        left.metric(fighter_a, f"{prediction['fighter_a_win_probability']:.1%}")
        right.metric(fighter_b, f"{prediction['fighter_b_win_probability']:.1%}")
        tier.metric("Confidence", prediction["confidence_tier"])
        st.progress(prediction["fighter_a_win_probability"], text=f"{fighter_a} win probability")
        st.progress(prediction["fighter_b_win_probability"], text=f"{fighter_b} win probability")

        features = build_prediction_features(
            fighter_a=fighter_a,
            fighter_b=fighter_b,
            fight_date=fight_date,
            weight_class=weight_class,
            scheduled_rounds=scheduled_rounds,
            fights=fights,
            fight_stats=fight_stats,
            fighters=fighters,
            scorecards=scorecards,
        )
        factor_col, table_col = st.columns([1, 1])
        with factor_col:
            st.subheader("Top Factors")
            for reason in prediction["top_factors_for_prediction"][:10]:
                st.write(reason)
        with table_col:
            st.subheader("Fighter Comparison")
            comparison_rows = []
            for metric in [
                "pre_fight_elo",
                "win_rate_before",
                "total_ufc_fights_before",
                "last_3_win_rate",
                "striking_differential",
                "takedown_defense",
                "finish_rate",
                "five_round_experience",
            ]:
                comparison_rows.append(
                    {
                        "metric": metric,
                        fighter_a: features.get(f"fighter_a_{metric}", pd.Series([None])).iloc[0],
                        fighter_b: features.get(f"fighter_b_{metric}", pd.Series([None])).iloc[0],
                    }
                )
            st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)

        recent_form = []
        if not fights.empty:
            dated = fights.copy()
            dated["fight_date"] = pd.to_datetime(dated["fight_date"], errors="coerce")
            for fighter in [fighter_a, fighter_b]:
                rows = dated[
                    (dated["fight_date"] < pd.Timestamp(fight_date))
                    & ((dated["fighter_a"] == fighter) | (dated["fighter_b"] == fighter))
                ].tail(5)
                for _, row in rows.iterrows():
                    opponent = row["fighter_b"] if row["fighter_a"] == fighter else row["fighter_a"]
                    recent_form.append(
                        {
                            "fighter": fighter,
                            "date": row["fight_date"].date(),
                            "opponent": opponent,
                            "result": "W" if row.get("winner") == fighter else "L",
                            "method": row.get("method", ""),
                        }
                    )
        st.subheader("Recent Form")
        st.dataframe(pd.DataFrame(recent_form), use_container_width=True, hide_index=True)

    backtest_path = settings.processed_data_dir / "backtest_results.json"
    if backtest_path.exists():
        metrics = json.loads(backtest_path.read_text(encoding="utf-8"))
        st.subheader("Backtest Performance")
        cols = st.columns(5)
        for col, metric in zip(cols, ["accuracy", "log_loss", "brier_score", "roc_auc", "expected_calibration_error"]):
            value = metrics.get(metric)
            col.metric(metric, f"{value:.3f}" if isinstance(value, (int, float)) else "n/a")


if __name__ == "__main__":  # pragma: no cover
    main()
