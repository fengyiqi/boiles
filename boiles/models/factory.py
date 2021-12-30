#!/usr/bin/env python3

from logging import Logger
from typing import Any, Dict, List, Optional, Type

import torch
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.objective import MultiObjective
from ax.core.optimization_config import OptimizationConfig
from ax.core.search_space import SearchSpace
from ax.core.types import TConfig
from ax.modelbridge.multi_objective_torch import MultiObjectiveTorchModelBridge
from ax.modelbridge.registry import (
    Models,
    Y_trans,
)
from ax.modelbridge.torch import TorchModelBridge
from ax.modelbridge.transforms.base import Transform
from ax.models.torch.botorch import (
    TAcqfConstructor,
    TModelConstructor,
    TModelPredictor,
    TOptimizer,
)
from ax.models.torch.botorch_defaults import (
    get_and_fit_model,
    predict_from_model,
)

from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import checked_cast
from ax.models.torch.botorch_defaults import get_NEI

logger: Logger = get_logger(__name__)


DEFAULT_TORCH_DEVICE = torch.device("cpu")
DEFAULT_EHVI_BATCH_LIMIT = 5


from ax.modelbridge.transforms.one_hot import OneHot
from ax.modelbridge.transforms.choice_encode import OrderedChoiceEncode
from ax.modelbridge.transforms.remove_fixed import RemoveFixed
from ax.modelbridge.transforms.int_to_float import IntToFloat
from ax.modelbridge.transforms.log import Log
from ax.modelbridge.transforms.unit_x import UnitX
from ax.models.torch.botorch_moo_defaults import (
    get_EHVI,
    get_NEHVI,
    pareto_frontier_evaluator,
    scipy_optimizer_list,
    infer_objective_thresholds,
)

from .botorch_defaults import scipy_optimizer

Cont_X_trans: List[Type[Transform]] = [
    RemoveFixed,
    OrderedChoiceEncode,
    OneHot,
    IntToFloat,
    Log,
    UnitX,
]

def get_botorch(
    experiment: Experiment,
    data: Data,
    search_space: Optional[SearchSpace] = None,
    dtype: torch.dtype = torch.double,
    device: torch.device = DEFAULT_TORCH_DEVICE,
    transforms: List[Type[Transform]] = Cont_X_trans + Y_trans,
    transform_configs: Optional[Dict[str, TConfig]] = None,
    model_constructor: TModelConstructor = get_and_fit_model,
    model_predictor: TModelPredictor = predict_from_model,
    acqf_constructor: TAcqfConstructor = get_NEI,  # pyre-ignore[9]
    acqf_optimizer: TOptimizer = scipy_optimizer,  # pyre-ignore[9]
    refit_on_cv: bool = False,
    refit_on_update: bool = True,
    optimization_config: Optional[OptimizationConfig] = None,
) -> TorchModelBridge:
    """Instantiates a BotorchModel."""
    if data.df.empty:  # pragma: no cover
        raise ValueError("`BotorchModel` requires non-empty data.")
    return checked_cast(
        TorchModelBridge,
        Models.BOTORCH(
            experiment=experiment,
            data=data,
            search_space=search_space or experiment.search_space,
            torch_dtype=dtype,
            torch_device=device,
            transforms=transforms,
            transform_configs=transform_configs,
            model_constructor=model_constructor,
            model_predictor=model_predictor,
            acqf_constructor=acqf_constructor,
            acqf_optimizer=acqf_optimizer,
            refit_on_cv=refit_on_cv,
            refit_on_update=refit_on_update,
            optimization_config=optimization_config,
        ),
    )


def get_moo_ehvi_botorch(
    experiment: Experiment,
    data: Data,
    search_space: Optional[SearchSpace] = None,
    dtype: torch.dtype = torch.double,
    device: torch.device = DEFAULT_TORCH_DEVICE,
    transforms: List[Type[Transform]] = Cont_X_trans + Y_trans,
    transform_configs: Optional[Dict[str, TConfig]] = None,
    model_constructor: TModelConstructor = get_and_fit_model,
    model_predictor: TModelPredictor = predict_from_model,
    acqf_constructor: TAcqfConstructor = get_EHVI,  # pyre-ignore[9]
    acqf_optimizer: TOptimizer = scipy_optimizer,  # pyre-ignore[9]
    refit_on_cv: bool = False,
    refit_on_update: bool = True,
    optimization_config: Optional[OptimizationConfig] = None,
) -> TorchModelBridge:
    """Instantiates a BotorchModel."""
    if not isinstance(experiment.optimization_config.objective, MultiObjective):
        raise ValueError("Multi-objective optimization requires multiple objectives.")
    if data.df.empty:  # pragma: no cover
        raise ValueError("MultiObjectiveOptimization requires non-empty data.")
    return checked_cast(
        MultiObjectiveTorchModelBridge,
        Models.MOO(
            experiment=experiment,
            data=data,
            search_space=search_space or experiment.search_space,
            torch_dtype=dtype,
            torch_device=device,
            transforms=transforms,
            transform_configs=transform_configs,
            model_constructor=model_constructor,
            model_predictor=model_predictor,
            acqf_constructor=acqf_constructor,
            acqf_optimizer=acqf_optimizer,
            refit_on_cv=refit_on_cv,
            refit_on_update=refit_on_update,
            optimization_config=optimization_config,
            default_model_gen_options={
                "acquisition_function_kwargs": {"sequential": True},
                "optimizer_kwargs": {
                    # having a batch limit is very important for avoiding
                    # memory issues in the initialization
                    "batch_limit": DEFAULT_EHVI_BATCH_LIMIT
                },
            },
        ),
    )


def get_moo_parego_botorch(
    experiment: Experiment,
    data: Data,
    search_space: Optional[SearchSpace] = None,
    dtype: torch.dtype = torch.double,
    device: torch.device = DEFAULT_TORCH_DEVICE,
    transforms: List[Type[Transform]] = Cont_X_trans + Y_trans,
    transform_configs: Optional[Dict[str, TConfig]] = None,
    model_constructor: TModelConstructor = get_and_fit_model,
    model_predictor: TModelPredictor = predict_from_model,
    acqf_constructor: TAcqfConstructor = get_NEI,  # pyre-ignore[9]
    acqf_optimizer: TOptimizer = scipy_optimizer,  # pyre-ignore[9]
    refit_on_cv: bool = False,
    refit_on_update: bool = True,
    optimization_config: Optional[OptimizationConfig] = None,
) -> TorchModelBridge:
    """Instantiates a BotorchModel."""
    if not isinstance(experiment.optimization_config.objective, MultiObjective):
        raise ValueError("Multi-objective optimization requires multiple objectives.")
    if data.df.empty:  # pragma: no cover
        raise ValueError("MultiObjectiveOptimization requires non-empty data.")
    return checked_cast(
        MultiObjectiveTorchModelBridge,
        Models.MOO(
            experiment=experiment,
            data=data,
            search_space=search_space or experiment.search_space,
            torch_dtype=dtype,
            torch_device=device,
            transforms=transforms,
            transform_configs=transform_configs,
            model_constructor=model_constructor,
            model_predictor=model_predictor,
            acqf_constructor=acqf_constructor,
            acqf_optimizer=acqf_optimizer,
            refit_on_cv=refit_on_cv,
            refit_on_update=refit_on_update,
            optimization_config=optimization_config,
            default_model_gen_options={
                "acquisition_function_kwargs": {
                    "chebyshev_scalarization": True,
                    "sequential": True,
                }
            },
        ),
    )

