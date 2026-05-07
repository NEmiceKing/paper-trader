"""
Signal Fusion: Combine RL signal with LLM analyst reports and debate results.

Three-tier fusion strategy:
  Tier 1: Feature Injection — LLM features added to RL observation space (always active when LLM enabled)
  Tier 2: Override — LLM consensus overrides RL when confidence and agreement exceed thresholds
  Tier 3: Debate Boost — Debate result adds confidence weight to final signal
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SignalFusionConfig:
    rl_weight: float = 0.7
    llm_override_threshold: float = 0.8
    agreement_threshold: float = 0.6
    debate_boost: float = 0.15
    min_override_confidence: float = 0.6


def fuse_signals(
    rl_signal,           # TradeSignal from RL model
    llm_reports: list,   # list of AnalystReport
    debate_result,       # DebateResult or None
    config: SignalFusionConfig,
):
    """Fuse RL and LLM signals into a final trading signal.

    Args:
        rl_signal: TradeSignal from PPO model
        llm_reports: List of AnalystReport from LLM analysts
        debate_result: Optional DebateResult from bull/bear debate
        config: Fusion configuration

    Returns:
        TradeSignal with fused direction, size, and confidence
    """
    from src.ibkr.order_manager import TradeSignal

    # Map RL action to score
    rl_map = {0: 0.0, 1: 1.0, 2: -1.0}
    rl_score = rl_map.get(rl_signal.direction, 0.0) * rl_signal.confidence

    # Default: RL only
    if not llm_reports and not debate_result:
        return rl_signal

    # Tier 1: Compute LLM consensus metrics (with dynamic reflection weights)
    llm_consensus = 0.0
    llm_agreement = 0.0
    if llm_reports:
        # Use reflection-adjusted confidence values directly
        scores = np.array([r.signal * r.confidence for r in llm_reports])
        llm_consensus = float(np.mean(scores))
        llm_agreement = float(1.0 - np.std(scores)) if len(scores) > 1 else 1.0

    # Dynamic RL weight from reflection if available
    effective_rl_weight = getattr(config, 'rl_weight', 0.2)

    # Start with RL-weighted score
    fusion_score = rl_score * effective_rl_weight
    if llm_reports and abs(llm_consensus) > 0:
        fusion_score += llm_consensus * (1.0 - effective_rl_weight)

    # Tier 2: Override check — LLM consensus can override RL
    override_triggered = False
    if llm_reports:
        conditions = [
            abs(llm_consensus) > config.llm_override_threshold,
            llm_agreement > config.agreement_threshold,
            rl_score * llm_consensus < 0,  # RL and LLM disagree
            rl_signal.confidence < config.min_override_confidence,
        ]
        if all(conditions):
            fusion_score = llm_consensus  # LLM takes over
            override_triggered = True
            logger.info(
                f"LLM override triggered: consensus={llm_consensus:.2f}, "
                f"agreement={llm_agreement:.2f}, rl_conf={rl_signal.confidence:.2f}"
            )

    # Tier 3: Debate boost
    if debate_result and abs(debate_result.composite_signal) > 0:
        debate_score = debate_result.composite_signal * debate_result.confidence
        # If debate strongly disagrees with current fusion, weight it more
        if fusion_score * debate_score < -0.3:
            fusion_score = fusion_score * 0.5 + debate_score * 0.5
        else:
            fusion_score += debate_score * config.debate_boost

    # Map back to TradeSignal
    abs_score = abs(fusion_score)
    if abs_score < 0.1:
        direction, size = 0, 0.0
    else:
        direction = 1 if fusion_score > 0 else 2
        size = min(abs_score, 1.0)

    confidence = float(min(abs_score, 1.0))

    return TradeSignal(
        symbol=rl_signal.symbol,
        direction=direction,
        size=size,
        confidence=confidence,
    )


def auto_adjust_weights(config, tracker, iteration_count: int,
                        adjust_every_n: int = 10) -> dict:
    """Auto-adjust fusion weights based on reflection tracker data.

    Called periodically (every N iterations) in the paper trading loop.

    Returns updated config dict with:
      - rl_weight: dynamically adjusted based on LLM analyst accuracy
      - override_threshold: lowered if LLM analysts are performing well
    """
    if tracker is None or iteration_count % adjust_every_n != 0:
        return {}

    all_weights = tracker.get_all_weights()
    if not all_weights:
        return {}

    # Average LLM analyst performance
    avg_llm_weight = float(np.mean(list(all_weights.values())))
    # Count analysts with above-average performance
    good_analysts = sum(1 for w in all_weights.values() if w >= 1.0)
    total_analysts = len(all_weights)

    # Dynamic RL weight: reduce RL influence if LLM analysts are accurate
    # If LLM analysts are collectively > 1.0 (above baseline), give them more weight
    new_rl_weight = max(0.05, min(0.5, 1.0 - avg_llm_weight * 0.5))
    # If >50% analysts are performing well, lower the override threshold
    llm_quality = good_analysts / max(total_analysts, 1)
    new_override = max(0.2, 1.0 - llm_quality * 0.8)

    updates = {}
    if abs(new_rl_weight - config.rl_weight) > 0.02:
        config.rl_weight = round(new_rl_weight, 3)
        updates["rl_weight"] = config.rl_weight

    if abs(new_override - config.llm_override_threshold) > 0.03:
        config.llm_override_threshold = round(new_override, 3)
        updates["llm_override_threshold"] = config.llm_override_threshold

    if updates:
        logger.info(
            f"Auto-adjusted weights (iteration {iteration_count}): "
            f"rl_weight={config.rl_weight:.2f}, "
            f"override_threshold={config.llm_override_threshold:.2f} "
            f"(LLM quality={llm_quality:.0%}, avg_weight={avg_llm_weight:.2f}x)"
        )

    return updates
