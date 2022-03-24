#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, KeysView, List, Optional, Tuple, Union

from ax.core.types import TConfig
from ax.models.model_utils import best_observed_point, get_observed
from ax.models.torch.utils import (  # noqa F401
    _to_inequality_constraints,
    predict_from_model,
)
from botorch.acquisition.acquisition import (
    OneShotAcquisitionFunction,
)
from botorch.optim.initializers import (
    gen_one_shot_kg_initial_conditions,
)
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.logging import logger
from botorch.optim.fit import fit_gpytorch_torch
from ax.models.torch_base import TorchModel
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.exceptions.errors import UnsupportedError
from botorch.fit import fit_gpytorch_model
from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.multitask import FixedNoiseMultiTaskGP, MultiTaskGP
# from botorch.optim.optimize import optimize_acqf
from botorch.utils import (
    get_objective_weights_transform,
    get_outcome_constraint_transforms,
)
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.priors.lkj_prior import LKJCovariancePrior
from torch import Tensor
from botorch.generation.gen import gen_candidates_scipy
from botorch.optim.optimize import gen_batch_initial_conditions
from ..models.get_model import get_GP
from ..utils import *
from ..config import *

from .tree.tree2d import *

from .discrete.tree_based import get_tree_search_cand
from .discrete.constrained import get_constrained_cand

MIN_OBSERVED_NOISE_LEVEL = 1e-7


def get_and_fit_model(
    Xs: List[Tensor],
    Ys: List[Tensor],
    Yvars: List[Tensor],
    task_features: List[int],
    fidelity_features: List[int],
    metric_names: List[str],
    state_dict: Optional[Dict[str, Tensor]] = None,
    refit_model: bool = True,
    **kwargs: Any,
) -> GPyTorchModel:
    r"""Instantiates and fits a botorch GPyTorchModel using the given data.
    N.B. Currently, the logic for choosing ModelListGP vs other models is handled
    using if-else statements in lines 96-137. In the future, this logic should be
    taken care of by modular botorch.

    Args:
        Xs: List of X data, one tensor per outcome.
        Ys: List of Y data, one tensor per outcome.
        Yvars: List of observed variance of Ys.
        task_features: List of columns of X that are tasks.
        fidelity_features: List of columns of X that are fidelity parameters.
        metric_names: Names of each outcome Y in Ys.
        state_dict: If provided, will set model parameters to this state
            dictionary. Otherwise, will fit the model.
        refit_model: Flag for refitting model.

    Returns:
        A fitted GPyTorchModel.
    """
    if len(fidelity_features) > 0 and len(task_features) > 0:
        raise NotImplementedError(
            "Currently do not support MF-GP models with task_features!"
        )
    if len(fidelity_features) > 1:
        raise NotImplementedError(
            "Fidelity MF-GP models currently support only a single fidelity parameter!"
        )
    if len(task_features) > 1:
        raise NotImplementedError(
            f"This model only supports 1 task feature (got {task_features})"
        )
    elif len(task_features) == 1:
        task_feature = task_features[0]
    else:
        task_feature = None
    model = None

    # TODO: Better logic for deciding when to use a ModelListGP. Currently the
    # logic is unclear. The two cases in which ModelListGP is used are
    # (i) the training inputs (Xs) are not the same for the different outcomes, and
    # (ii) a multi-task model is used
    if task_feature is None:
        # TODO: Better logic for single-objective opt --Yiqi
        if len(Xs) == 1 and OC.dim_outputs == 1:
            # Use single output, single task GP
            model = get_GP(
                X=Xs[0],
                Y=Ys[0],
                Yvar=Yvars[0],
                case=OP.test_cases[0],
                task_feature=task_feature,
                fidelity_features=fidelity_features,
                **kwargs,
            )
        elif all(torch.equal(Xs[0], X) for X in Xs[1:]):
            # Use batched multioutput, single task GP
            Y = torch.cat(Ys, dim=-1)
            Yvar = torch.cat(Yvars, dim=-1)
            models = []
            for i in range(OC.dim_outputs):
                models.append(
                    get_GP(
                        X=Xs[0],
                        Y=Y[:, i].reshape(-1, 1),
                        Yvar=Yvar[:, i].reshape(-1, 1),
                        case=OP.test_cases[i],
                        task_feature=task_feature,
                        fidelity_features=fidelity_features,
                        **kwargs,
                    )
                )
            model = ModelListGP(*models)
    # TODO: Is this equivalent an "else:" here?
    model.to(Xs[0])
    # print(model)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    if state_dict is None or refit_model:
        # TODO: Add bounds for optimization stability - requires revamp upstream
        bounds = {}
        if isinstance(model, ModelListGP):
            # if CC.num_outputs == 1:
            #     mll = ExactMarginalLogLikelihood(model[0].likelihood, model[0])
            # else:
            mll = SumMarginalLogLikelihood(model.likelihood, model)
        else:
            # pyre-ignore: [16]
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
        kwargs['options'] = {"maxiter": OC.max_iteration, "disp": OC.disp, "lr": OC.lr}
        kwargs['optimizer'] = fit_gpytorch_torch
        # pop these two keys since they are invalide for fit_gpytorch_torch anymore. I don't know where these keys are defined.
        kwargs.pop("use_input_warping")
        kwargs.pop("use_loocv_pseudo_likelihood")
        mll = fit_gpytorch_model(mll, bounds=bounds, **kwargs)

        # if OC.sensitivity_control == 'prior_distribution' or OC.sensitivity_control == 'combine':
        # if OC.dim_outputs != 1:
        #     save_lengthscales_prior(mll)

    return model


def optimize_acqf(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    raw_samples: int,
    options: Optional[Dict[str, Union[bool, float, int, str]]] = None,
    inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    equality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    post_processing_func: Optional[Callable[[Tensor], Tensor]] = None,
    batch_initial_conditions: Optional[Tensor] = None,
    return_best_only: bool = True,
    sequential: bool = False,
    **kwargs: Any,
) -> Tuple[Tensor, Tensor]:
    r"""Generate a set of candidates via multi-start optimization.

    Args:
        acq_function: An AcquisitionFunction.
        bounds: A `2 x d` tensor of lower and upper bounds for each column of `X`.
        q: The number of candidates.
        num_restarts: The number of starting points for multistart acquisition
            function optimization.
        raw_samples: The number of samples for initialization.
        options: Options for candidate generation.
        inequality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`
        equality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) = rhs`
        fixed_features: A map `{feature_index: value}` for features that
            should be fixed to a particular value during generation.
        post_processing_func: A function that post-processes an optimization
            result appropriately (i.e., according to `round-trip`
            transformations).
        batch_initial_conditions: A tensor to specify the initial conditions. Set
            this if you do not want to use default initialization strategy.
        return_best_only: If False, outputs the solutions corresponding to all
            random restart initializations of the optimization.
        sequential: If False, uses joint optimization, otherwise uses sequential
            optimization.
        kwargs: Additonal keyword arguments.

    Returns:
        A two-element tuple containing

        - a `(num_restarts) x q x d`-dim tensor of generated candidates.
        - a tensor of associated acquisition values. If `sequential=False`,
            this is a `(num_restarts)`-dim tensor of joint acquisition values
            (with explicit restart dimension if `return_best_only=False`). If
            `sequential=True`, this is a `q`-dim tensor of expected acquisition
            values conditional on having observed canidates `0,1,...,i-1`.

    Example:
        >>> # generate `q=2` candidates jointly using 20 random restarts
        >>> # and 512 raw samples
        >>> candidates, acq_value = optimize_acqf(qEI, bounds, 2, 20, 512)

        >>> generate `q=3` candidates sequentially using 15 random restarts
        >>> # and 256 raw samples
        >>> qEI = qExpectedImprovement(model, best_f=0.2)
        >>> bounds = torch.tensor([[0.], [1.]])
        >>> candidates, acq_value_list = optimize_acqf(
        >>>     qEI, bounds, 3, 15, 256, sequential=True
        >>> )
    """
    if sequential and q > 1:
        if not return_best_only:
            raise NotImplementedError(
                "return_best_only=False only supported for joint optimization"
            )
        if isinstance(acq_function, OneShotAcquisitionFunction):
            raise NotImplementedError(
                "sequential optimization currently not supported for one-shot "
                "acquisition functions. Must have `sequential=False`."
            )
        candidate_list, acq_value_list = [], []
        candidates = torch.tensor([], device=bounds.device, dtype=bounds.dtype)
        base_X_pending = acq_function.X_pending
        for i in range(q):
            candidate, acq_value = optimize_acqf(
                acq_function=acq_function,
                bounds=bounds,
                q=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options=options or {},
                inequality_constraints=inequality_constraints,
                equality_constraints=equality_constraints,
                fixed_features=fixed_features,
                post_processing_func=post_processing_func,
                batch_initial_conditions=None,
                return_best_only=True,
                sequential=False,
            )
            candidate_list.append(candidate)
            acq_value_list.append(acq_value)
            candidates = torch.cat(candidate_list, dim=-2)
            acq_function.set_X_pending(
                torch.cat([base_X_pending, candidates], dim=-2)
                if base_X_pending is not None
                else candidates
            )
            logger.info(f"Generated sequential candidate {i+1} of {q}")
        # Reset acq_func to previous X_pending state
        acq_function.set_X_pending(base_X_pending)
        return candidates, torch.stack(acq_value_list)

    options = options or {}

    if batch_initial_conditions is None:
        ic_gen = (
            gen_one_shot_kg_initial_conditions
            if isinstance(acq_function, qKnowledgeGradient)
            else gen_batch_initial_conditions
        )
        # TODO: Generating initial candidates should use parameter constraints.
        batch_initial_conditions = ic_gen(
            acq_function=acq_function,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options=options,
        )

    batch_limit: int = options.get("batch_limit", num_restarts)
    batch_candidates_list: List[Tensor] = []
    batch_acq_values_list: List[Tensor] = []
    start_idcs = list(range(0, num_restarts, batch_limit))
    for start_idx in start_idcs:
        end_idx = min(start_idx + batch_limit, num_restarts)
        # optimize using random restart optimization
        batch_candidates_curr, batch_acq_values_curr = gen_candidates_scipy(
            initial_conditions=batch_initial_conditions[start_idx:end_idx],
            acquisition_function=acq_function,
            lower_bounds=bounds[0],
            upper_bounds=bounds[1],
            options={
                k: v
                for k, v in options.items()
                if k not in ("init_batch_limit", "batch_limit", "nonnegative")
            },
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            fixed_features=fixed_features,
        )
        batch_candidates_list.append(batch_candidates_curr)
        batch_acq_values_list.append(batch_acq_values_curr)
        logger.info(f"Generated candidate batch {start_idx+1} of {len(start_idcs)}.")
    batch_candidates = torch.cat(batch_candidates_list)
    batch_acq_values = torch.cat(batch_acq_values_list)

    if post_processing_func is not None:
        batch_candidates = post_processing_func(batch_candidates)

    if return_best_only:
        best = torch.argmax(batch_acq_values.view(-1), dim=0)
        batch_candidates = batch_candidates[best]
        batch_acq_values = batch_acq_values[best]

    if isinstance(acq_function, OneShotAcquisitionFunction):
        if not kwargs.get("return_full_tree", False):
            batch_candidates = acq_function.extract_candidates(X_full=batch_candidates)

    return batch_candidates, batch_acq_values


def scipy_optimizer(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    n: int,
    inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    rounding_func: Optional[Callable[[Tensor], Tensor]] = None,
    **kwargs: Any,
) -> Tuple[Tensor, Tensor]:
    r"""Optimizer using scipy's minimize module on a numpy-adpator.

    Args:
        acq_function: A botorch AcquisitionFunction.
        bounds: A `2 x d`-dim tensor, where `bounds[0]` (`bounds[1]`) are the
            lower (upper) bounds of the feasible hyperrectangle.
        n: The number of candidates to generate.
        inequality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`
        fixed_features: A map {feature_index: value} for features that should
            be fixed to a particular value during generation.
        rounding_func: A function that rounds an optimization result
            appropriately (i.e., according to `round-trip` transformations).

    Returns:
        2-element tuple containing

        - A `n x d`-dim tensor of generated candidates.
        - In the case of joint optimization, a scalar tensor containing
          the joint acquisition value of the `n` points. In the case of
          sequential optimization, a `n`-dim tensor of conditional acquisition
          values, where `i`-th element is the expected acquisition value
          conditional on having observed candidates `0,1,...,i-1`.
    """
    num_restarts: int = kwargs.get("num_restarts", 20)
    raw_samples: int = kwargs.get("num_raw_samples", 50 * num_restarts)

    if kwargs.get("joint_optimization", False):
        sequential = False
    else:
        sequential = True
        # use SLSQP by default for small problems since it yields faster wall times
        if "method" not in kwargs:
            kwargs["method"] = "SLSQP"

    X, expected_acquisition_value = optimize_acqf(
        acq_function=acq_function,
        bounds=bounds,
        q=n,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options=kwargs,
        inequality_constraints=inequality_constraints,
        fixed_features=fixed_features,
        sequential=sequential,
        post_processing_func=rounding_func,
    )
    X = discrete_helper(acq_function.model, X)
    return X, expected_acquisition_value


def discrete_helper(model, X):
    if OC.discrete["activate"]:
        # log("Discrete assistant is activated!")
        if OC.discrete["method"].lower() == "constrained":
            return get_constrained_cand(model, X)
        elif OC.discrete["method"].lower() == "tree":
            return get_tree_search_cand(model, X)
        else:
            raise Exception("Such method isn't implemented.")
    else:
        # log("Discrete assistant isn't activated")
        return X


def recommend_best_observed_point(
    model: TorchModel,
    bounds: List[Tuple[float, float]],
    objective_weights: Tensor,
    outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    model_gen_options: Optional[TConfig] = None,
    target_fidelities: Optional[Dict[int, float]] = None,
) -> Optional[Tensor]:
    """
    A wrapper around `ax.models.model_utils.best_observed_point` for TorchModel
    that recommends a best point from previously observed points using either a
    "max_utility" or "feasible_threshold" strategy.

    Args:
        model: A TorchModel.
        bounds: A list of (lower, upper) tuples for each column of X.
        objective_weights: The objective is to maximize a weighted sum of
            the columns of f(x). These are the weights.
        outcome_constraints: A tuple of (A, b). For k outcome constraints
            and m outputs at f(x), A is (k x m) and b is (k x 1) such that
            A f(x) <= b.
        linear_constraints: A tuple of (A, b). For k linear constraints on
            d-dimensional x, A is (k x d) and b is (k x 1) such that
            A x <= b.
        fixed_features: A map {feature_index: value} for features that
            should be fixed to a particular value in the best point.
        model_gen_options: A config dictionary that can contain
            model-specific options.
        target_fidelities: A map {feature_index: value} of fidelity feature
            column indices to their respective target fidelities. Used for
            multi-fidelity optimization.

    Returns:
        A d-array of the best point, or None if no feasible point was observed.
    """
    if target_fidelities:
        raise NotImplementedError(
            "target_fidelities not implemented for base BotorchModel"
        )

    x_best = best_observed_point(
        model=model,
        bounds=bounds,
        objective_weights=objective_weights,
        outcome_constraints=outcome_constraints,
        linear_constraints=linear_constraints,
        fixed_features=fixed_features,
        options=model_gen_options,
    )
    if x_best is None:
        return None
    return x_best.to(dtype=model.dtype, device=torch.device("cpu"))


def recommend_best_out_of_sample_point(
    model: TorchModel,
    bounds: List[Tuple[float, float]],
    objective_weights: Tensor,
    outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    model_gen_options: Optional[TConfig] = None,
    target_fidelities: Optional[Dict[int, float]] = None,
) -> Optional[Tensor]:
    """
    Identify the current best point by optimizing the posterior mean of the model.
    This is "out-of-sample" because it considers un-observed designs as well.

    Return None if no such point can be identified.

    Args:
        model: A TorchModel.
        bounds: A list of (lower, upper) tuples for each column of X.
        objective_weights: The objective is to maximize a weighted sum of
            the columns of f(x). These are the weights.
        outcome_constraints: A tuple of (A, b). For k outcome constraints
            and m outputs at f(x), A is (k x m) and b is (k x 1) such that
            A f(x) <= b.
        linear_constraints: A tuple of (A, b). For k linear constraints on
            d-dimensional x, A is (k x d) and b is (k x 1) such that
            A x <= b.
        fixed_features: A map {feature_index: value} for features that
            should be fixed to a particular value in the best point.
        model_gen_options: A config dictionary that can contain
            model-specific options.
        target_fidelities: A map {feature_index: value} of fidelity feature
            column indices to their respective target fidelities. Used for
            multi-fidelity optimization.

    Returns:
        A d-array of the best point, or None if no feasible point exists.
    """
    options = model_gen_options or {}
    fixed_features = fixed_features or {}
    acf_options = options.get("acquisition_function_kwargs", {})
    optimizer_options = options.get("optimizer_kwargs", {})

    X_observed = get_observed(
        Xs=model.Xs,  # pyre-ignore: [16]
        objective_weights=objective_weights,
        outcome_constraints=outcome_constraints,
    )

    if hasattr(model, "_get_best_point_acqf"):
        acq_function, non_fixed_idcs = model._get_best_point_acqf(  # pyre-ignore: [16]
            X_observed=X_observed,
            objective_weights=objective_weights,
            mc_samples=acf_options.get("mc_samples", 512),
            fixed_features=fixed_features,
            target_fidelities=target_fidelities,
            outcome_constraints=outcome_constraints,
            seed_inner=acf_options.get("seed_inner", None),
            qmc=acf_options.get("qmc", True),
        )
    else:
        raise RuntimeError("The model should implement _get_best_point_acqf.")

    inequality_constraints = _to_inequality_constraints(linear_constraints)
    # TODO: update optimizers to handle inequality_constraints
    # (including transforming constraints b/c of fixed features)
    if inequality_constraints is not None:
        raise UnsupportedError("Inequality constraints are not supported!")

    return_best_only = optimizer_options.get("return_best_only", True)
    bounds_ = torch.tensor(bounds, dtype=model.dtype, device=model.device)
    bounds_ = bounds_.transpose(-1, -2)
    if non_fixed_idcs is not None:
        bounds_ = bounds_[..., non_fixed_idcs]

    candidates, _ = optimize_acqf(
        acq_function=acq_function,
        bounds=bounds_,
        q=1,
        num_restarts=optimizer_options.get("num_restarts", 60),
        raw_samples=optimizer_options.get("raw_samples", 1024),
        inequality_constraints=inequality_constraints,
        fixed_features=None,  # handled inside the acquisition function
        options={
            "batch_limit": optimizer_options.get("batch_limit", 8),
            "maxiter": optimizer_options.get("maxiter", 200),
            "nonnegative": optimizer_options.get("nonnegative", False),
            "method": "L-BFGS-B",
        },
        return_best_only=return_best_only,
    )
    rec_point = candidates.detach().cpu()
    if isinstance(acq_function, FixedFeatureAcquisitionFunction):
        rec_point = acq_function._construct_X_full(rec_point)
    if return_best_only:
        rec_point = rec_point.view(-1)
    return rec_point


def _get_model(
    X: Tensor,
    Y: Tensor,
    Yvar: Tensor,
    task_feature: Optional[int] = None,
    fidelity_features: Optional[List[int]] = None,
    **kwargs: Any,
) -> GPyTorchModel:
    """Instantiate a model of type depending on the input data.

    Args:
        X: A `n x d` tensor of input features.
        Y: A `n x m` tensor of input observations.
        Yvar: A `n x m` tensor of input variances (NaN if unobserved).
        task_feature: The index of the column pertaining to the task feature
            (if present).
        fidelity_features: List of columns of X that are fidelity parameters.

    Returns:
        A GPyTorchModel (unfitted).
    """
    Yvar = Yvar.clamp_min_(MIN_OBSERVED_NOISE_LEVEL)
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
    if fidelity_features is None:
        fidelity_features = []
    if len(fidelity_features) == 0:
        # only pass linear_truncated arg if there are fidelities
        kwargs = {k: v for k, v in kwargs.items() if k != "linear_truncated"}
    if len(fidelity_features) > 0:
        if task_feature:
            raise NotImplementedError(  # pragma: no cover
                "multi-task multi-fidelity models not yet available"
            )
        # at this point we can assume that there is only a single fidelity parameter
        gp = SingleTaskMultiFidelityGP(
            train_X=X, train_Y=Y, data_fidelity=fidelity_features[0], **kwargs
        )
    elif task_feature is None and all_nan_Yvar:
        gp = SingleTaskGP(train_X=X, train_Y=Y, **kwargs)
    elif task_feature is None:
        gp = FixedNoiseGP(train_X=X, train_Y=Y, train_Yvar=Yvar, **kwargs)
    else:
        # instantiate multitask GP
        all_tasks, _, _ = MultiTaskGP.get_all_tasks(X, task_feature)
        num_tasks = len(all_tasks)
        prior_dict = kwargs.get("prior")
        prior = None
        if prior_dict is not None:
            prior_type = prior_dict.get("type", None)
            if issubclass(prior_type, LKJCovariancePrior):
                sd_prior = prior_dict.get("sd_prior", GammaPrior(1.0, 0.15))
                sd_prior._event_shape = torch.Size([num_tasks])
                eta = prior_dict.get("eta", 0.5)
                if not isinstance(eta, float) and not isinstance(eta, int):
                    raise ValueError(f"eta must be a real number, your eta was {eta}")
                prior = LKJCovariancePrior(num_tasks, eta, sd_prior)

            else:
                raise NotImplementedError(
                    "Currently only LKJ prior is supported,"
                    f"your prior type was {prior_type}."
                )

        if all_nan_Yvar:
            gp = MultiTaskGP(
                train_X=X,
                train_Y=Y,
                task_feature=task_feature,
                rank=kwargs.get("rank"),
                task_covar_prior=prior,
            )
        else:
            gp = FixedNoiseMultiTaskGP(
                train_X=X,
                train_Y=Y,
                train_Yvar=Yvar,
                task_feature=task_feature,
                rank=kwargs.get("rank"),
                task_covar_prior=prior,
            )
    return gp
