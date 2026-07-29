from dataclasses import asdict, dataclass
from enum import StrEnum

ML_MODEL_SCREENING_VERSION = 1


class ScreeningEngine(StrEnum):
    RIDGE = "ridge"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"


@dataclass(frozen=True, slots=True)
class RidgeConfiguration:
    alpha: float = 1.0


@dataclass(frozen=True, slots=True)
class LightGBMConfiguration:
    n_estimators: int = 120
    max_depth: int = 5
    num_leaves: int = 24
    learning_rate: float = 0.05


@dataclass(frozen=True, slots=True)
class CatBoostConfiguration:
    iterations: int = 120
    depth: int = 5
    learning_rate: float = 0.05


@dataclass(frozen=True, slots=True)
class ModelScreeningConfiguration:
    seed: int = 1729
    thread_count: int = 2
    ridge: RidgeConfiguration = RidgeConfiguration()
    lightgbm: LightGBMConfiguration = LightGBMConfiguration()
    catboost: CatBoostConfiguration = CatBoostConfiguration()

    def __post_init__(self) -> None:
        if self.thread_count < 1:
            raise ValueError("thread_count must be positive")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


DEFAULT_SCREENING_CONFIGURATION = ModelScreeningConfiguration()
ALL_ENGINES = tuple(ScreeningEngine)
