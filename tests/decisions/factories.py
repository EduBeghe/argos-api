from datetime import UTC, datetime
from decimal import Decimal

from app.decisions.persistence import ScoreEvidence
from app.market.enums import Asset, Interval, MarketType, Provider
from app.scores.calculator import ScoreInterpretation
from app.scores.models import MarketScore


def make_score(
    *,
    overall_score: Decimal = Decimal("0.60"),
    asset: Asset = Asset.BTC_USDT,
) -> MarketScore:
    return MarketScore(
        asset=asset,
        provider=Provider.BINANCE,
        market_type=MarketType.SPOT,
        interval=Interval.FIVE_MINUTES,
        candle_open_time=datetime(2026, 7, 15, 12, tzinfo=UTC),
        model_name="baseline_technical",
        model_version=1,
        calculated_at=datetime(2026, 7, 15, 12, 5, tzinfo=UTC),
        trend_score=Decimal("0"),
        momentum_score=Decimal("0"),
        volume_activity=Decimal("0"),
        volatility_ratio=Decimal("0"),
        directional_score=overall_score,
        volume_confirmation_factor=Decimal("1"),
        overall_score=overall_score,
        signal_strength=abs(overall_score),
        trend_contribution=overall_score,
        rsi_contribution=Decimal("0"),
        return_1_contribution=Decimal("0"),
        return_12_contribution=Decimal("0"),
        interpretation=(
            ScoreInterpretation.STRONGLY_BULLISH
            if overall_score >= Decimal("0.60")
            else ScoreInterpretation.BULLISH
            if overall_score >= Decimal("0.20")
            else ScoreInterpretation.NEUTRAL
            if overall_score > Decimal("-0.20")
            else ScoreInterpretation.BEARISH
            if overall_score > Decimal("-0.60")
            else ScoreInterpretation.STRONGLY_BEARISH
        ),
    )


def make_evidence(
    *, overall_score: Decimal = Decimal("0.60"), asset: Asset = Asset.BTC_USDT
) -> ScoreEvidence:
    return ScoreEvidence(
        score=make_score(overall_score=overall_score, asset=asset),
        source_market_price=Decimal("65000.123456789012345678"),
    )
