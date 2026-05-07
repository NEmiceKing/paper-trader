"""
Reflection Learning — tracks LLM analyst accuracy over time and adjusts
confidence weights dynamically. Inspired by QuantDinger's ReflectionWorker.

Key idea: each analyst has a track record. Analysts who consistently predict
correctly get higher weight. Those who don't get downweighted.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AnalystScore:
    analyst_name: str
    total_predictions: int = 0
    correct_predictions: int = 0
    accuracy: float = 0.5  # start at neutral
    recent_streak: int = 0  # consecutive correct (+N) or wrong (-N)
    weight_multiplier: float = 1.0  # dynamic weight for signal fusion
    last_updated: str = ""


class ReflectionTracker:
    """Tracks analyst prediction accuracy and adjusts weights over time.

    Persists to Parquet for durability across restarts.
    """

    def __init__(self, store_path: str = "data/reflection"):
        self.store_path = store_path
        self.scores: dict[str, AnalystScore] = {}
        self._pending_predictions: list[dict] = []  # predictions awaiting outcome
        self._load()

    def _load(self):
        import os
        path = f"{self.store_path}/analyst_scores.parquet"
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                for _, row in df.iterrows():
                    self.scores[row["analyst_name"]] = AnalystScore(
                        analyst_name=row["analyst_name"],
                        total_predictions=int(row.get("total_predictions", 0)),
                        correct_predictions=int(row.get("correct_predictions", 0)),
                        accuracy=float(row.get("accuracy", 0.5)),
                        recent_streak=int(row.get("recent_streak", 0)),
                        weight_multiplier=float(row.get("weight_multiplier", 1.0)),
                        last_updated=str(row.get("last_updated", "")),
                    )
            except Exception as e:
                logger.debug(f"Could not load reflection scores: {e}")

    def _save(self):
        import os
        os.makedirs(self.store_path, exist_ok=True)
        records = []
        for s in self.scores.values():
            records.append({
                "analyst_name": s.analyst_name,
                "total_predictions": s.total_predictions,
                "correct_predictions": s.correct_predictions,
                "accuracy": s.accuracy,
                "recent_streak": s.recent_streak,
                "weight_multiplier": s.weight_multiplier,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            })
        df = pd.DataFrame(records)
        df.to_parquet(f"{self.store_path}/analyst_scores.parquet")

    def record_prediction(self, symbol: str, analyst_name: str, signal: int,
                          confidence: float, price_at_prediction: float):
        """Record a prediction that will be evaluated later."""
        self._pending_predictions.append({
            "symbol": symbol,
            "analyst_name": analyst_name,
            "signal": signal,  # -1, 0, +1
            "confidence": confidence,
            "price": price_at_prediction,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def evaluate_predictions(self, symbol: str, current_price: float) -> dict:
        """Evaluate pending predictions against actual price movement.

        Returns a dict of {analyst_name: was_correct} for all evaluated predictions.
        """
        results = {}
        newly_evaluated = []

        for pred in self._pending_predictions:
            if pred["symbol"] != symbol:
                continue

            # Directional accuracy: did price move in the predicted direction?
            price_change = current_price - pred["price"]
            actual_direction = 1 if price_change > 0 else (-1 if price_change < 0 else 0)
            was_correct = (pred["signal"] * actual_direction > 0) if pred["signal"] != 0 else None

            name = pred["analyst_name"]
            if name not in self.scores:
                self.scores[name] = AnalystScore(analyst_name=name)

            score = self.scores[name]
            score.total_predictions += 1
            if was_correct is True:
                score.correct_predictions += 1
                score.recent_streak = max(0, score.recent_streak) + 1
            elif was_correct is False:
                score.recent_streak = min(0, score.recent_streak) - 1

            score.accuracy = score.correct_predictions / max(score.total_predictions, 1)

            # Weight multiplier: accuracy * streak bonus/penalty
            streak_bonus = 0.05 * score.recent_streak  # ±5% per consecutive correct
            score.weight_multiplier = np.clip(score.accuracy + streak_bonus, 0.2, 2.0)

            score.last_updated = datetime.now(timezone.utc).isoformat()
            results[name] = was_correct
            newly_evaluated.append(pred)

        # Remove evaluated predictions
        self._pending_predictions = [
            p for p in self._pending_predictions if p not in newly_evaluated
        ]

        if newly_evaluated:
            self._save()

        return results

    def get_weight(self, analyst_name: str, default: float = 1.0) -> float:
        """Get the current weight multiplier for an analyst."""
        score = self.scores.get(analyst_name)
        return score.weight_multiplier if score else default

    def get_all_weights(self) -> dict[str, float]:
        """Get weights for all known analysts."""
        return {name: s.weight_multiplier for name, s in self.scores.items()}

    def get_report(self) -> str:
        """Generate a human-readable reflection report."""
        if not self.scores:
            return "No reflection data yet."

        lines = ["Analyst Performance Report", "=" * 40]
        for name, s in sorted(self.scores.items(), key=lambda x: x[1].accuracy, reverse=True):
            streak_sign = "+" if s.recent_streak > 0 else ""
            lines.append(
                f"  {name:20s} | Acc={s.accuracy:.1%} | "
                f"N={s.total_predictions:4d} | "
                f"Streak={streak_sign}{s.recent_streak} | "
                f"Weight={s.weight_multiplier:.2f}x"
            )
        return "\n".join(lines)

    def reset_stale(self, max_days: int = 30):
        """Reset weights for analysts with no recent predictions."""
        if not self.scores:
            return
        cutoff = datetime.now(timezone.utc)
        for name, s in list(self.scores.items()):
            if s.last_updated:
                try:
                    last = datetime.fromisoformat(s.last_updated)
                    if (cutoff - last).days > max_days:
                        s.weight_multiplier = 1.0
                        s.recent_streak = 0
                        logger.info(f"Reset stale weights for {name}")
                except (ValueError, TypeError):
                    pass
        self._save()


def apply_reflection_weights(
    analyst_reports: list,
    tracker: ReflectionTracker,
) -> list:
    """Apply reflection-based weights to analyst reports.

    Each report's confidence is multiplied by the analyst's track record weight.
    """
    weighted_reports = []
    for r in analyst_reports:
        weight = tracker.get_weight(r.analyst_name, default=1.0)
        # Create a copy with adjusted confidence
        from dataclasses import replace
        adjusted = replace(r, confidence=min(r.confidence * weight, 1.0))
        weighted_reports.append(adjusted)
        if weight != 1.0:
            logger.debug(
                f"[Reflection] {r.analyst_name}: conf {r.confidence:.2f} → "
                f"{adjusted.confidence:.2f} (weight={weight:.2f}x)"
            )
    return weighted_reports
