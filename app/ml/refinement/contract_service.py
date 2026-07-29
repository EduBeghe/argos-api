from decimal import Decimal

from app.ml.contract import (
    FUTURE_DURATIONS,
    MISSING_DATA_POLICY_VERSION,
    VARIABLE_REGISTRY_VERSION,
)
from app.ml.registry import VARIABLE_REGISTRY
from app.ml.screening.config import ScreeningEngine

from .contracts import (
    MAXIMUM_RUNTIME_SECONDS,
    MAXIMUM_TRIAL_BUDGET,
    ML_REFINEMENT_CLASSIFICATION_VERSION,
    ML_REFINEMENT_CONTRACT_VERSION,
    PredictionObjective,
)
from .schemas import (
    RefinementContractResponse,
    RefinementDefaultsResponse,
    RefinementLimitsResponse,
    RefinementVariableResponse,
)


def refinement_contract() -> RefinementContractResponse:
    return RefinementContractResponse(
        contract_version=ML_REFINEMENT_CONTRACT_VERSION,
        classification_version=ML_REFINEMENT_CLASSIFICATION_VERSION,
        variable_registry_version=VARIABLE_REGISTRY_VERSION,
        missing_data_policy_version=MISSING_DATA_POLICY_VERSION,
        supported_engines=[item.value for item in ScreeningEngine],
        supported_prediction_objectives=list(PredictionObjective),
        allowed_future_durations_seconds=[
            int(item.total_seconds()) for item in FUTURE_DURATIONS
        ],
        variables=[
            RefinementVariableResponse(
                name=item.name,
                description=item.description,
                category=item.category.value,
                lookback_seconds=int(item.lookback.total_seconds()),
                calculation=item.calculation,
                calculation_version=item.calculation_version,
                higher_is_directional=item.higher_is_directional,
            )
            for item in VARIABLE_REGISTRY
        ],
        validation_limits=RefinementLimitsResponse(
            maximum_trials=MAXIMUM_TRIAL_BUDGET,
            maximum_runtime_seconds=MAXIMUM_RUNTIME_SECONDS,
            maximum_notes_length=1000,
        ),
        ui_defaults=RefinementDefaultsResponse(
            fee_rate=Decimal("0.001"),
            slippage_rate=Decimal("0.0005"),
            deterministic_seed=1729,
        ),
    )
