#!/usr/bin/env python3

from typing import Any, Dict, Optional, Tuple
import numpy as np
from ax.core.observation import ObservationFeatures
from ax.modelbridge.base import ModelBridge
from ax.plot.base import AxPlotConfig, AxPlotTypes, PlotData
from ax.plot.color import BLUE_SCALE, GREEN_PINK_SCALE, GREEN_SCALE
from ax.plot.helper import (
    TNullableGeneratorRunsDict,
    contour_config_to_trace,
)

from ax.plot.contour import _get_contour_predictions


def get_datadic(
        model: ModelBridge,
        param_x: str,
        param_y: str,
        metric_name: str,
        generator_runs_dict: TNullableGeneratorRunsDict = None,
        relative: bool = False,
        density: int = 50,
        slice_values: Optional[Dict[str, Any]] = None,
        lower_is_better: bool = False,
        fixed_features: Optional[ObservationFeatures] = None,
        trial_index: Optional[int] = None,
) -> Dict[str, Any]:
    """A customized function that is used to extract data from a real time fitted model.

    Args:
        model: ModelBridge that contains model for predictions
        param_x: Name of parameter that will be sliced on x-axis
        param_y: Name of parameter that will be sliced on y-axis
        metric_name: Name of metric to plot
        generator_runs_dict: A dictionary {name: generator run} of generator runs
            whose arms will be plotted, if they lie in the slice.
        relative: Predictions relative to status quo
        density: Number of points along slice to evaluate predictions.
        slice_values: A dictionary {name: val} for the fixed values of the
            other parameters. If not provided, then the status quo values will
            be used if there is a status quo, otherwise the mean of numeric
            parameters or the mode of choice parameters.
        lower_is_better: Lower values for metric are better.
        fixed_features: An ObservationFeatures object containing the values of
            features (including non-parameter features like context) to be set
            in the slice.
    """
    if param_x == param_y:
        raise ValueError("Please select different parameters for x- and y-dimensions.")

    if trial_index is not None:
        if slice_values is None:
            slice_values = {}
        slice_values["TRIAL_PARAM"] = str(trial_index)

    data, f_plt, sd_plt, grid_x, grid_y, scales = _get_contour_predictions(
        model=model,
        x_param_name=param_x,
        y_param_name=param_y,
        metric=metric_name,
        generator_runs_dict=generator_runs_dict,
        density=density,
        slice_values=slice_values,
    )
    config = {
        "arm_data": data,
        "blue_scale": BLUE_SCALE,
        "density": density,
        "f": f_plt,
        "green_scale": GREEN_SCALE,
        "green_pink_scale": GREEN_PINK_SCALE,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "lower_is_better": lower_is_better,
        "metric": metric_name,
        "rel": relative,
        "sd": sd_plt,
        "xvar": param_x,
        "yvar": param_y,
        "x_is_log": scales["x"],
        "y_is_log": scales["y"],
    }

    config = AxPlotConfig(config, plot_type=AxPlotTypes.GENERIC).data
    traces = contour_config_to_trace(config)

    data_mean = {'x': traces[0]['x'],
                 'y': traces[0]['y'],
                 'z': traces[0]['z'],
                 'zmin': traces[0]['zmin']}

    data_var = {'x': traces[1]['x'],
                'y': traces[1]['y'],
                'z': traces[1]['z']}

    data = {'mean': data_mean, 'var': data_var}
    return data
