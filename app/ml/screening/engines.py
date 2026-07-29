from importlib import import_module
from importlib.metadata import version
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .config import ModelScreeningConfiguration, ScreeningEngine


class RegressionEngine(Protocol):
    def fit_predict(
        self,
        training_features: NDArray[np.float32],
        training_targets: NDArray[np.float32],
        validation_features: NDArray[np.float32],
    ) -> NDArray[np.float64]: ...


def validate_engine_dependencies() -> None:
    """Import every native-backed engine dependency and fail without substitution."""

    for module_name in ("sklearn", "lightgbm", "catboost"):
        import_module(module_name)


def library_versions() -> dict[str, str]:
    return {
        "scikit-learn": version("scikit-learn"),
        "lightgbm": version("lightgbm"),
        "catboost": version("catboost"),
    }


class RidgeEngine:
    def __init__(self, configuration: ModelScreeningConfiguration) -> None:
        self.configuration = configuration

    def fit_predict(
        self,
        training_features: NDArray[np.float32],
        training_targets: NDArray[np.float32],
        validation_features: NDArray[np.float32],
    ) -> NDArray[np.float64]:
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model = make_pipeline(
            StandardScaler(), Ridge(alpha=self.configuration.ridge.alpha)
        )
        model.fit(training_features, training_targets)
        return np.asarray(model.predict(validation_features), dtype=np.float64)


class LightGBMEngine:
    def __init__(self, configuration: ModelScreeningConfiguration) -> None:
        self.configuration = configuration

    def fit_predict(
        self,
        training_features: NDArray[np.float32],
        training_targets: NDArray[np.float32],
        validation_features: NDArray[np.float32],
    ) -> NDArray[np.float64]:
        from lightgbm import LGBMRegressor

        config = self.configuration
        model = LGBMRegressor(
            n_estimators=config.lightgbm.n_estimators,
            max_depth=config.lightgbm.max_depth,
            num_leaves=config.lightgbm.num_leaves,
            learning_rate=config.lightgbm.learning_rate,
            random_state=config.seed,
            n_jobs=config.thread_count,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
        model.fit(training_features, training_targets)
        return np.asarray(model.predict(validation_features), dtype=np.float64)


class CatBoostEngine:
    def __init__(self, configuration: ModelScreeningConfiguration) -> None:
        self.configuration = configuration

    def fit_predict(
        self,
        training_features: NDArray[np.float32],
        training_targets: NDArray[np.float32],
        validation_features: NDArray[np.float32],
    ) -> NDArray[np.float64]:
        from catboost import CatBoostRegressor

        config = self.configuration
        model = CatBoostRegressor(
            iterations=config.catboost.iterations,
            depth=config.catboost.depth,
            learning_rate=config.catboost.learning_rate,
            random_seed=config.seed,
            thread_count=config.thread_count,
            task_type="CPU",
            allow_writing_files=False,
            verbose=False,
        )
        model.fit(training_features, training_targets)
        return np.asarray(model.predict(validation_features), dtype=np.float64)


def create_engine(
    engine: ScreeningEngine, configuration: ModelScreeningConfiguration
) -> RegressionEngine:
    if engine is ScreeningEngine.RIDGE:
        return RidgeEngine(configuration)
    if engine is ScreeningEngine.LIGHTGBM:
        return LightGBMEngine(configuration)
    return CatBoostEngine(configuration)
