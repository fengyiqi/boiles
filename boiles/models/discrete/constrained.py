from ...utils import read_from_csv, log
from ...config.opt_config import OC
from ...models.tree.tree2d import *
from botorch.acquisition import UpperConfidenceBound
import torch
from botorch.optim.optimize import optimize_acqf
import plotly.graph_objects as go
from ...analytical.functions import solutions
from scipy.optimize import minimize
from .tree_based import get_tree_search_cand
from .utils import get_normalized_inputs, get_inputs, cast_through_int, to_int, get_gpdata

display = True


def get_constrained_cand(model, acq_cand):
    opt_bounds = OC.opt_bounds
    fun_bounds = OC.fun_bounds
    # here we want to find how large the smallest grid cell should be
    radiuses = []
    for i, increment_ in enumerate(OC.increment):
        radiuses.append(increment_ / (fun_bounds[i][1] - fun_bounds[i][0]) / 2.0)
    radiuses = np.array(radiuses)
    smallest_radius_index = radiuses.argmin()
    radius = radiuses[smallest_radius_index]
    # denomitor used for the square shape function
    factors = radiuses / radius

    plot = OC.discrete["plot"]
    #TODO adapt this code for multiobjective optimization
    raw_inputs = get_inputs(f"discrete/data.csv")
    acq_x_float = cast_through_int(acq_cand, opt_bounds)
    acq_x_int = to_int(acq_cand, opt_bounds)
    if plot and OC.dim_inputs == 2:
        plot_2Dcontour(model, acq_x_float, levels=np.arange(-1.0, 6, 0.2), title="AF Proposed Candidate")
    if plot and OC.dim_inputs == 3:
        plot_plotly(model, acq_x_float, title="AF Proposed Candidate")

    n_iter = 0
    regenerate = False
    constraints_ = []
    while n_iter < 10:
        if acq_x_int in raw_inputs.astype(int).tolist():
            regenerate = True
            n_iter += 1
            log(f"--- Constrained minimizing started! iteration {n_iter}---", print_to_terminal=display)

            if OC.dim_inputs == 2:
                cons = {'type': 'ineq',
                        'fun': lambda x: ((x[0] - acq_x_float.squeeze()[0]) / factors[0]) ** 4 +
                                         ((x[1] - acq_x_float.squeeze()[1]) / factors[1]) ** 4 - radius ** 4}
            elif OC.dim_inputs == 3:
                cons = {'type': 'ineq',
                        'fun': lambda x: ((x[0] - acq_x_float.squeeze()[0]) / factors[0]) ** 4 +
                                         ((x[1] - acq_x_float.squeeze()[1]) / factors[1]) ** 4 +
                                         ((x[2] - acq_x_float.squeeze()[2]) / factors[2]) ** 4 - radius ** 4}
            else:
                raise Exception(f"Such dimension is invalid.")

            constraints_.append(cons)
            res = multistart_minimize(fun, model, constraints_)
            res = cast_through_int(res, opt_bounds)

            res = np.unique(res, axis=0)
            for tree_cand in res:
                log(f"\tevaluating {tree_cand}", print_to_terminal=display)
                satisfied = []
                for cons in constraints_:
                    satisfied.append(cons['fun'](tree_cand) > 0)
                if False in satisfied:
                    continue
                acq_x_int = to_int(tree_cand, opt_bounds)
                acq_x_float = cast_through_int(np.expand_dims(tree_cand, 0), opt_bounds)

                if acq_x_int not in raw_inputs.astype(int).tolist():
                    acq_cand = torch.tensor(acq_x_float)
                    if plot and OC.dim_inputs == 2:
                        plot_2Dcontour(model, acq_x_float, levels=np.arange(-1.0, 6, 0.2), title="Tree Proposed Candidate")
                    if plot and OC.dim_inputs == 3:
                        plot_plotly(model, acq_x_float, title="Tree Proposed Candidate")
                    return acq_cand
        else:
            break
    # If the constrained optimization can't provide candidate (regenerate is True but function doesn't return), we use
    # the tree-based search to generate a sample. This doesn't happen very often but we have to provide a solution.
    if regenerate:
        log("Using tree-based search.")
        acq_cand = get_tree_search_cand(model, acq_cand)

    return acq_cand


def multistart_minimize(fun, model, constraints_):
    n = 50 * OC.dim_inputs
    starts = np.random.rand(n).reshape(-1, OC.dim_inputs)
    x_array, fun_array = [], []
    for x0 in starts:
        res = minimize(fun, x0, args=(model), method='SLSQP', bounds=[(0, 1)] * OC.dim_inputs,
                       constraints=constraints_, options={'disp': False})
        if res['success']:
            x_array.append(res['x'])
            fun_array.append(res['fun'])
    sort_index = np.array(fun_array).argsort()
    return np.array(x_array)[sort_index]


# Upper Confidence Bound  acquisition function
def fun(x, model):
    X = torch.tensor([x])
    posterior = model.posterior(X=X)
    mean = posterior.mean.detach().numpy()
    variance = posterior.variance.detach().numpy()
    return (mean - 0.1 * variance).squeeze()


def plot_2Dcontour(model, acq_cand, levels, title: str):
    inputs = get_normalized_inputs(f"discrete/data.csv")

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

    x1, x2, pred = get_gpdata(model)
    con = ax.contourf(x1, x2, pred, levels=levels)
    label_fontsize = 18
    tick_fontsize = 16
    ax.scatter(inputs[:, 0], inputs[:, 1], s=20, marker="x", c='white', alpha=0.5)

    ax.scatter(acq_cand.squeeze()[0], acq_cand.squeeze()[1], s=20, marker="x", c='red')
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$x1$', fontsize=label_fontsize)
    ax.set_ylabel(r'$x2$', fontsize=label_fontsize)
    ax.tick_params(labelsize=tick_fontsize)

    cbar = fig.colorbar(con, ax=ax)
    cbar.ax.tick_params(labelsize=tick_fontsize)
    cbar.formatter.set_powerlimits((-2, 2))
    cbar.update_ticks()
    fig.tight_layout()
    plt.savefig(f"discrete/{title[:2]}_{OC.discrete['counter']}.jpg")
    plt.close()


def plot_plotly(model, acq_cand, title=""):
    x, y, z, pred = get_gpdata(model)
    fig_list = []
    # plot 4D volume
    volume = go.Volume(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=pred.flatten(),
        isomin=pred.min(),
        isomax=abs(pred.min()) * 0.2 + pred.min(),
        opacity=0.3,  # needs to be small to see through all surfaces
        surface_count=10,  # needs to be a large number for good volume rendering
        colorscale="Viridis"
    )
    fig_list += [volume]
    # plot existing samples
    inputs = get_normalized_inputs(f"discrete/data.csv")
    scatter = go.Scatter3d(
        x=inputs[:, 0].flatten(),
        y=inputs[:, 1].flatten(),
        z=inputs[:, 2].flatten(),
        mode="markers",
        marker=dict(
            color="black",
            symbol="x",
            size=4
        )
    )
    fig_list += [scatter]
    # plot acq_cand
    cand = acq_cand.squeeze()
    scatter_acq = go.Scatter3d(
        x=[cand[0]],
        y=[cand[1]],
        z=[cand[2]],
        mode="markers",
        marker=dict(
            color="red",
            symbol="diamond",
            size=8
        )
    )
    fig_list += [scatter_acq]

    fig = go.Figure()
    for trace in fig_list:
        fig.add_trace(trace)

    fig.update_layout(
        autosize=False,
        scene=dict(
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            zaxis=dict(range=[0, 1])
        ),
        width=1200,
        height=1200,
        title=title,
        showlegend=False,
    )
    fig.write_html(f"discrete/{title[:2]}_{OC.discrete['counter']}.html", auto_open=False)

