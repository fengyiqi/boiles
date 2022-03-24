from ...utils import log
from ..tree.tree import Tree
from ..tree.utils import get_node_list, update_id, get_max_level
from ...config.opt_config import OC
from ...models.tree.tree2d import Tree2D
from ...models.tree.tree3d import Tree3D
from botorch.acquisition import UpperConfidenceBound
import torch
from botorch.optim.optimize import optimize_acqf
import numpy as np
import plotly.graph_objects as go
from ...analytical.functions import solutions
from .utils import get_inputs, get_normalized_inputs, get_bestf, cast_through_int, to_int, get_gpdata, normalize
import matplotlib.pyplot as plt
from botorch.sampling.samplers import IIDNormalSampler, MCSampler, SobolQMCNormalSampler
from botorch.acquisition.objective import IdentityMCObjective
from ax.models.random.sobol import SobolGenerator

if OC.discrete["activate"]:
    if OC.dim_inputs == 2:
        tree = Tree2D
    elif OC.dim_inputs == 3:
        tree = Tree3D
    else:
        raise Exception(f"Not implemented for {OC.dim_inputs}D input space.")


def get_tree_search_cand(model, acq_cand):
    level = OC.discrete["max_level"]
    bounds = OC.opt_bounds
    plot = OC.discrete["plot"]

    inputs = get_normalized_inputs(f"discrete/data.csv")
    raw_inputs = get_inputs(f"discrete/data.csv")
    y_min = get_bestf(f"discrete/data.csv")

    ori_tree = tree.build_tree(inputs.shape[0], inputs)
    ori_tree_node_list = get_node_list(ori_tree)
    update_id(ori_tree)

    acq_x_float = cast_through_int(acq_cand, bounds)
    acq_x_int = to_int(acq_cand, bounds)

    all_samples = np.vstack((inputs, np.array(acq_x_float)))
    new_tree = tree.build_tree(all_samples.shape[0], all_samples)
    new_tree_node_list = get_node_list(new_tree)

    max_level = get_max_level(new_tree)

    if plot:
        if OC.dim_inputs == 2:
            plot_contour_with_tree(model, new_tree, acq_x_float, levels=np.arange(-1.0, 6, 0.2),
                                   title="AF Proposed Candidate")
        elif OC.dim_inputs == 3:
            plot_plotly(model, new_tree, acq_x_float, title="AF Proposed Candidate")
        else:
            raise Exception(f"Such dimension is invalid.")

    del_node = []
    if max_level > level or acq_x_int in raw_inputs.astype(int).tolist():
        log("--- Tree search started! ---")
        # find the node due to the continuous id sequence by comparing the level
        for i, node in enumerate(ori_tree_node_list):
            if node.level != new_tree_node_list[i].level:
                print(f"del_id: {node.id}")
                del_node.append(node)
                break
            else:
                continue
        # remove the refined node or node with duplicated sample from a copy of original tree where there is no acq_cand
        temp_tree_node_list = ori_tree_node_list.copy()
        for node in del_node:
            temp_tree_node_list.remove(node)
        sorted_node = get_min_node(model, temp_tree_node_list)
        iteration = 0
        for i, min_node in enumerate(sorted_node):
            # determine if skip one node to accelerate the iteration e.g. 0, 2, 4, 6, ...
            # if i != 2*iteration:
            #     continue

            log(f"\titeration: {iteration}, select node: {min_node.id}")
            iteration += 1

            # print(f"old_len: {len(old_tree_node_list)} | id: {var_node.id} | var: {node.var}")
            tree_cand = generate_sample(model=model, node=min_node, best_f=y_min)
            acq_x_int = to_int(tree_cand, bounds)
            if acq_x_int in raw_inputs.astype(int).tolist():
                log("\t\tFind duplicated candidate, go to next node.")
                continue

            tree_cand = cast_through_int(np.expand_dims(tree_cand, 0), bounds)

            new_samples = np.vstack((inputs, tree_cand))
            new_tree = tree.build_tree(new_samples.shape[0], new_samples)

            new_tree_node_list = get_node_list(new_tree)
            max_level = get_max_level(new_tree)

            if max_level <= level and acq_x_int not in raw_inputs.astype(int).tolist():
                acq_cand = torch.tensor(tree_cand)
                if plot:
                    if OC.dim_inputs == 2:
                        plot_contour_with_tree(model, new_tree, tree_cand,
                                               levels=np.arange(-1.0, 6, 0.2),
                                               title="Tree Proposed Candidate")
                    elif OC.dim_inputs == 3:
                        plot_plotly(model, new_tree, tree_cand, title="Tree Proposed Candidate")
                    else:
                        raise Exception(f"Such dimension is invalid.")
                return acq_cand

    return acq_cand


def generate_sample(model, node: Tree, best_f):
    sobol = SobolGenerator()
    if OC.dim_inputs == 2:
        samples = sobol.gen(100, bounds=[(0, 1), (0, 1)])[0]
        samples[:, 0] = samples[:, 0] * node.s + node.cord_x
        samples[:, 1] = samples[:, 1] * node.s + node.cord_y
    elif OC.dim_inputs == 3:
        samples = sobol.gen(100, bounds=[(0, 1), (0, 1), (0, 1)])[0]
        samples[:, 0] = samples[:, 0] * node.s + node.cord_x
        samples[:, 1] = samples[:, 1] * node.s + node.cord_y
        samples[:, 2] = samples[:, 2] * node.s + node.cord_z
    else:
        raise Exception(f"Such dimension is invalid.")
    # print(samples)
    acq_value = []
    for sample in samples:
        X = torch.tensor([sample])
        posterior = model.posterior(X)
        sampler = SobolQMCNormalSampler(num_samples=512, collapse_batch_dims=True)
        samples_ = sampler(posterior)
        objective = IdentityMCObjective()
        obj = objective(samples_, X=X)
        obj = (best_f - obj).clamp_min(0)
        q_ei = obj.max(dim=-1)[0].mean(dim=0).detach().numpy().tolist()

        acq_value.append(q_ei)

    acq_value = np.array(acq_value)
    return samples[acq_value.argmax()]


# model is a GP model
def get_min_node(model, node_list):
    pred_list = []
    for i, node in enumerate(node_list):
        UCB = UpperConfidenceBound(model, beta=0.1, maximize=False)
        if OC.dim_inputs == 2:
            bounds = torch.stack(
                [
                    torch.tensor([node.cord_x, node.cord_y]),
                    torch.tensor([node.cord_x + node.s, node.cord_y + node.s])
                ]
            )
        elif OC.dim_inputs == 3:
            bounds = torch.stack(
                [
                    torch.tensor([node.cord_x, node.cord_y, node.cord_z]),
                    torch.tensor([node.cord_x + node.s, node.cord_y + node.s, node.cord_z + node.s])
                ]
            )
        else:
            raise Exception(f"Such dimension is invalid.")

        candidate, acq_value = optimize_acqf(
            UCB,
            bounds=bounds,
            q=1,
            num_restarts=5,
            raw_samples=512,
        )
        # multiplying area or volume gives better results
        node.value = acq_value * (node.s ** OC.dim_inputs)
        pred_list.append(node.value)

    # because argsort() is only valid for ascending, multipy -1 to avoid reversing the list, saving time.
    pred_array = np.array(pred_list) * -1
    sorted_index = pred_array.argsort().tolist()
    sorted_node_list = []
    for i in sorted_index:
        sorted_node_list.append(node_list[i])

    return sorted_node_list


def plot_contour_with_tree(model, tree: Tree2D, acq_cand, levels, title: str, id: bool = True):
    inputs = get_normalized_inputs(f"discrete/data.csv")

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    tree.print_tree(plot=True)

    x1, x2, pred = get_gpdata(model)
    con = ax.contourf(x1, x2, pred, levels=levels)
    label_fontsize = 18
    tick_fontsize = 16
    ax.scatter(inputs[:, 0], inputs[:, 1], s=20, marker="x", c='white', alpha=0.5)
    if id:
        update_id(tree)
        node_list = get_node_list(tree)
        for node in node_list:
            ax.text(node.cord_x + node.s / 2, node.cord_y + node.s / 2, str(node.id), fontsize=5)

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


def plot_plotly(model, tree: Tree3D, acq_cand, title=""):
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
    # plot cells
    tree.print_tree(plotly=True, fig_list=fig_list)

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
