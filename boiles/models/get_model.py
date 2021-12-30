from typing import Any, List, Optional

import torch
from ax.models.torch.utils import (  # noqa F401
    _to_inequality_constraints,
    predict_from_model,
)
from .gp_model import GPModel
from ..config import *
from botorch.models.gpytorch import GPyTorchModel
from torch import Tensor
from ..test_cases.case_base import CaseBase

MIN_OBSERVED_NOISE_LEVEL = 1e-7


def get_GP(
        X: Tensor,
        Y: Tensor,
        Yvar: Tensor,
        case: CaseBase,
        task_feature: Optional[int] = None,
        fidelity_features: Optional[List[int]] = None,
        **kwargs: Any,
) -> GPyTorchModel:
    Yvar = Yvar.clone()
    Yvar = yvar_helper(Yvar, task_feature, Yvar_min=MIN_OBSERVED_NOISE_LEVEL)
    gp = GPModel(train_X=X, train_Y=Y, train_Yvar=Yvar, case=case, **kwargs)
    return gp


def yvar_helper(Yvar: Tensor, task_feature: Optional[int] = None, Yvar_min=None):
    Yvar = Yvar.clamp_min_(Yvar_min)
    is_nan = torch.isnan(Yvar)
    any_nan_Yvar = torch.any(is_nan)
    all_nan_Yvar = torch.all(is_nan)
    if any_nan_Yvar and not all_nan_Yvar:
        if task_feature:
            # TODO (jej): Replace with inferred noise before making perf judgements.
            Yvar[Yvar != Yvar] = MIN_OBSERVED_NOISE_LEVEL
        else:
            raise ValueError(
                "Mix of known and unknown variances indicates valuation function "
                "errors. Variances should all be specified, or none should be."
            )
    return Yvar
