from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from botorch import settings
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import Log, OutcomeTransform
from botorch.models.utils import validate_input_scaling
from botorch.sampling.samplers import MCSampler
from botorch.utils.containers import TrainingData
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods.gaussian_likelihood import (
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood
)
from gpytorch.priors import NormalPrior
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.models.exact_gp import ExactGP
from gpytorch.module import Module
from torch import Tensor
from gpytorch.constraints.constraints import GreaterThan

from ..utils import *
from ..test_cases.case_base import CaseBase
from ..config import *
from ..solvers.solver_config import SC

MIN_INFERRED_NOISE_LEVEL = 1e-7


class GPModel(BatchedMultiOutputGPyTorchModel, ExactGP):
    r"""A single-task exact GP model using fixed noise levels.

    A single-task exact GP that uses fixed observation noise levels. This model
    also uses relatively strong priors on the Kernel hyperparameters, which work
    best when covariates are normalized to the unit cube and outcomes are
    standardized (zero mean, unit variance).

    This model works in batch mode (each batch having its own hyperparameters).
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor,
        case: CaseBase,
        covar_module: Optional[Module] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
        initial_model: bool = False,
        **kwargs: Any,
    ) -> None:
        r"""A single-task exact GP model using fixed noise levels.

        Args:
            train_X: A `batch_shape x n x d` tensor of training features.
            train_Y: A `batch_shape x n x m` tensor of training observations.
            train_Yvar: A `batch_shape x n x m` tensor of observed measurement
                noise.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the `Posterior` obtained by calling
                `.posterior` on the model will be on the original scale).
            input_transform: An input transfrom that is applied in the model's
                forward pass.

        Example:
            >>> train_X = torch.rand(20, 2)
            >>> train_Y = torch.sin(train_X).sum(dim=1, keepdim=True)
            >>> train_Yvar = torch.full_like(train_Y, 0.2)
            >>> model = GPModel(train_X, train_Y, train_Yvar)
        """
        self.case = case
        self.name = case.name
        if input_transform is not None:
            input_transform.to(train_X)
        transformed_X = self.transform_inputs(
            X=train_X, input_transform=input_transform
        )
        if outcome_transform is not None:
            train_Y, train_Yvar = outcome_transform(train_Y, train_Yvar)
        self._validate_tensor_args(X=transformed_X, Y=train_Y, Yvar=train_Yvar)
        validate_input_scaling(
            train_X=transformed_X, train_Y=train_Y, train_Yvar=train_Yvar
        )
        self._set_dimensions(train_X=train_X, train_Y=train_Y)
        train_X, train_Y, train_Yvar = self._transform_tensor_args(
            X=train_X, Y=train_Y, Yvar=train_Yvar
        )
        # if self.case.sensitivity_control == 'fixed_noise_level' or self.case.sensitivity_control == 'prior_distribution':
        #     likelihood = FixedNoiseGaussianLikelihood(
        #         noise=train_Yvar, batch_shape=self._aug_batch_shape
        #     )
        # else:
        #     likelihood = GaussianLikelihood(
        #         noise_prior=NormalPrior(loc=0.0, scale=gaussian_noise_level[SC.Riemann][self.case.name])
        #     )
        noise_prior = GammaPrior(1.1, 0.05)
        noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
        likelihood = GaussianLikelihood(
            noise_prior=noise_prior,
            batch_shape=self._aug_batch_shape,
            noise_constraint=GreaterThan(
                MIN_INFERRED_NOISE_LEVEL,
                transform=None,
                initial_value=noise_prior_mode,
            ),
        )
        ExactGP.__init__(
            self, train_inputs=train_X, train_targets=train_Y, likelihood=likelihood
        )

        # if case.mean_prior is None:
        self.mean_module = ConstantMean(batch_shape=self._aug_batch_shape)
        # else:
        #     self.mean_module = ConstantMean(prior=NormalPrior(
        #         loc=case.highest_error_from_initial, scale=case.highest_error_from_initial*0.05
        #     ),
        #                                     batch_shape=self._aug_batch_shape)
        if case.lengthscale_prior:
            lengthscales_prior = read_lengthscales_prior('lengthscale_prior.json', key=self.case.name)
        else:
            lengthscales_prior = GammaPrior(3.0, 6.0)
        if covar_module is None:
            self.covar_module = ScaleKernel(
                base_kernel=MaternKernel(
                    nu=2.5,
                    ard_num_dims=transformed_X.shape[-1],
                    batch_shape=self._aug_batch_shape,
                    lengthscale_prior=lengthscales_prior,
                ),
                batch_shape=self._aug_batch_shape,
                outputscale_prior=GammaPrior(2.0, 0.15),
            )
            self._subset_batch_dict = {
                "mean_module.constant": -2,
                "covar_module.raw_outputscale": -1,
                "covar_module.base_kernel.raw_lengthscale": -3,
            }

        else:
            self.covar_module = covar_module
        # TODO: Allow subsetting of other covar modules
        if input_transform is not None:
            self.input_transform = input_transform
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform

        self.to(train_X)

    def fantasize(
        self,
        X: Tensor,
        sampler: MCSampler,
        observation_noise: Union[bool, Tensor] = True,
        **kwargs: Any,
    ):
        r"""Construct a fantasy model.

        Constructs a fantasy model in the following fashion:
        (1) compute the model posterior at `X` (if `observation_noise=True`,
        this includes observation noise taken as the mean across the observation
        noise in the training data. If `observation_noise` is a Tensor, use
        it directly as the observation noise to add).
        (2) sample from this posterior (using `sampler`) to generate "fake"
        observations.
        (3) condition the model on the new fake observations.

        Args:
            X: A `batch_shape x n' x d`-dim Tensor, where `d` is the dimension of
                the feature space, `n'` is the number of points per batch, and
                `batch_shape` is the batch shape (must be compatible with the
                batch shape of the model).
            sampler: The sampler used for sampling from the posterior at `X`.
            observation_noise: If True, include the mean across the observation
                noise in the training data as observation noise in the posterior
                from which the samples are drawn. If a Tensor, use it directly
                as the specified measurement noise.

        Returns:
            The constructed fantasy model.
        """
        propagate_grads = kwargs.pop("propagate_grads", False)
        with settings.propagate_grads(propagate_grads):
            post_X = self.posterior(X, observation_noise=observation_noise, **kwargs)
        Y_fantasized = sampler(post_X)  # num_fantasies x batch_shape x n' x m
        # Use the mean of the previous noise values (TODO: be smarter here).
        # noise should be batch_shape x q x m when X is batch_shape x q x d, and
        # Y_fantasized is num_fantasies x batch_shape x q x m.
        noise_shape = Y_fantasized.shape[1:]
        noise = self.likelihood.noise.mean().expand(noise_shape)
        return self.condition_on_observations(X=X, Y=Y_fantasized, noise=noise)

    def forward(self, x: Tensor) -> MultivariateNormal:
        x = self.transform_inputs(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def subset_output(self, idcs: List[int]) -> BatchedMultiOutputGPyTorchModel:
        r"""Subset the model along the output dimension.

        Args:
            idcs: The output indices to subset the model to.

        Returns:
            The current model, subset to the specified output indices.
        """
        new_model = super().subset_output(idcs=idcs)
        full_noise = new_model.likelihood.noise_covar.noise
        new_noise = full_noise[..., idcs if len(idcs) > 1 else idcs[0], :]
        new_model.likelihood.noise_covar.noise = new_noise
        return new_model

    @classmethod
    def construct_inputs(
        cls, training_data: TrainingData, **kwargs: Any
    ) -> Dict[str, Any]:
        r"""Construct kwargs for the `Model` from `TrainingData` and other options.

        Args:
            training_data: `TrainingData` container with data for single outcome
                or for multiple outcomes for batched multi-output case.
            **kwargs: None expected for this class.
        """
        if training_data.Yvar is None:
            raise ValueError(f"Yvar required for {cls.__name__}.")
        return {
            "train_X": training_data.X,
            "train_Y": training_data.Y,
            "train_Yvar": training_data.Yvar,
        }
