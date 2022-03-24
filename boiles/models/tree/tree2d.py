import numpy as np
import matplotlib.pyplot as plt
from ax.models.torch.utils import predict_from_model
import gpytorch
import torch
import math
import random
from botorch.sampling.samplers import IIDNormalSampler, MCSampler, SobolQMCNormalSampler
from botorch.acquisition.objective import IdentityMCObjective
from ax.models.random.sobol import SobolGenerator
import plotly.graph_objects as go
from ...utils import log
from .tree import Tree


class Tree2D(Tree):
    def __init__(self, x, y, s, root, father_id=None):  # set empty to True if initializing an empty node
        if root:
            self.sample_x, self.sample_y, self.cord_x, self.cord_y = None, None, x, y
            self.s = s
            self.center = [self.cord_x + self.s/2, self.cord_y + self.s/2]
            self.n = 0
            self.level = round(math.log(1 / self.s, 2))
        else:
            self.sample_x, self.sample_y = x, y
            self.cord_x, self.cord_y = None, None
            self.center = None
            self.s = None
            self.n = None
        self.value = None
        self.id = None
        self.father_id = father_id
        self.children = [None, None, None, None]
        self.candidate = None
        self.acq_value = None


    def add(self, body):
        if self.__in_node__([body.sample_x, body.sample_y]):
            if self.n == 0:  # node empty; base case
                #                self.__set_COM__(body.__get_COM__(), body.__get_mass__())
                self.n += 1
                self.sample_x = body.sample_x
                self.sample_y = body.sample_y

                return True
            elif self.children[0] != None:  # internal node
                #                 self.__update_COM__(body.__get_COM__(), body.__get_mass__())
                return self.children[0].add(body) or self.children[1].add(body) or self.children[2].add(body) or \
                       self.children[3].add(body)
            else:  # external node; have to divide
                old = Tree2D(self.sample_x, self.sample_y, None, False)
                loc = (self.cord_x, self.cord_y)
                s = self.s
                ll, lr = Tree2D(loc[0], loc[1], s / 2, True, father_id=self.id), Tree2D(loc[0] + s / 2, loc[1], s / 2, True, father_id=self.id)
                ul, ur = Tree2D(loc[0], loc[1] + s / 2, s / 2, True, father_id=self.id), Tree2D(loc[0] + s / 2, loc[1] + s / 2, s / 2, True, father_id=self.id)

                self.__set_neighbors__(ul, ur, ll, lr)
                self.add(old) and self.add(body)
                #                 self.__update_COM_in__()
                return True
        return False

    def __set_neighbors__(self, ul, ur, ll, lr):
        self.children = [ul, ur, ll, lr]

    def __in_node__(self, pos):
        return pos[0] >= self.cord_x and pos[0] <= self.cord_x + self.s and pos[1] >= self.cord_y and pos[
            1] <= self.cord_y + self.s

    # outputs a pre traversal or whatever it's called
    def print_tree(self, text=False, plot=False, plotly=False, data_list=None):
        if self.n == 0:  # none empty
            if text:
                log("Empty node! Square information: " + str(self.sample_x) + " " + str(self.sample_y) + " " + str(
                    self.s))
            return True
        elif self.children[0] != None:  # internal node
            if text:
                log("Internal node! Center of mass information: " + str(self.sample_x) + " " + str(
                    self.sample_y) + " " + str(self.n))
            if plot:
                plt.vlines(self.cord_x + self.s / 2, self.cord_y, self.cord_y + self.s, colors='k', alpha=1, linewidth=0.6)
                plt.hlines(self.cord_y + self.s / 2, self.cord_x, self.cord_x + self.s, colors='k', alpha=1, linewidth=0.6)
            if plotly:
                if data_list is None:
                    raise RuntimeError("You need to give a data list.")
                linex = go.Scatter(x=[self.cord_x, self.cord_x + self.s],
                                   y=[self.cord_y + self.s / 2, self.cord_y + self.s / 2],
                                   mode="lines",
                                   line=dict(color="black", width=1))
                liney = go.Scatter(x=[self.cord_x + self.s / 2, self.cord_x + self.s / 2],
                                   y=[self.cord_y, self.cord_y + self.s],
                                   mode="lines",
                                   line=dict(color="black", width=1))
                data_list.append(linex)
                data_list.append(liney)
            # plt.scatter(self.sample_x, self.sample_y, s=10, marker="x", c="black")
            return self.children[0].print_tree(text, plot, plotly, data_list) and \
                   self.children[1].print_tree(text, plot, plotly, data_list) and \
                   self.children[2].print_tree(text, plot, plotly, data_list) and \
                   self.children[3].print_tree(text, plot, plotly, data_list)
        else:  # external node
            if text:
                print(
                    "External node! Particle information: " + str(self.sample_x) + " " + str(self.sample_y) + " " + str(
                        self.n))
            return True

    # builds tree from positions
    def build_tree(n, pos):
        head = Tree2D(0.0 - 1e-6, 0.0 + 1e-6, 1.0 + 2e-6, True)
        for i in range(n):
            head.add(Tree2D(pos[i, 0], pos[i, 1], None, False))
        return head


def unit(x, bounds):
    return (x-bounds[0]) / (bounds[1] - bounds[0])


# only for lower_is_better
def generate_sample(model, tree: Tree2D, best_f):

    sobol = SobolGenerator()

    samples = sobol.gen(100, bounds=[(0, 1), (0, 1)])[0]
    samples[:, 0] = samples[:, 0] * tree.s + tree.cord_x
    samples[:, 1] = samples[:, 1] * tree.s + tree.cord_y
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


def get_max_level(tree: Tree2D):
    node_list = get_node_list(tree)
    level = 0
    for i, node in enumerate(node_list):
        if node.level > level:
            level = node.level
    return level