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
# from mytools.models.tree import Tree
from ...utils import log
from .tree import Tree


class Tree3D(Tree):

    def __init__(self, x, y, z, s, root, father_id=None):  # set empty to True if initializing an empty node
        if root:
            self.sample_x, self.sample_y, self.sample_z, self.cord_x, self.cord_y, self.cord_z = None, None, None, x, y, z
            self.s = s
            self.center = [self.cord_x + self.s / 2, self.cord_y + self.s / 2, self.cord_z + self.s / 2]
            self.n = 0
            self.level = math.log(1 / self.s, 2)
        else:
            self.sample_x, self.sample_y, self.sample_z = x, y, z
            self.cord_x, self.cord_y, self.cord_z = None, None, None
            self.center = None
            self.s = None
            self.n = None
        self.value = None
        self.id = None
        self.father_id = father_id
        self.children = [None, None, None, None, None, None, None, None]
        self.candidate = None
        self.acq_value = None

    def add(self, body):
        if self.__in_node__([body.sample_x, body.sample_y, body.sample_z]):
            if self.n == 0:  # node empty; base case
                #                self.__set_COM__(body.__get_COM__(), body.__get_mass__())
                self.n += 1
                self.sample_x = body.sample_x
                self.sample_y = body.sample_y
                self.sample_z = body.sample_z
                return True
            elif self.children[0] != None:  # internal node
                #                 self.__update_COM__(body.__get_COM__(), body.__get_mass__())
                return self.children[0].add(body) or \
                       self.children[1].add(body) or \
                       self.children[2].add(body) or \
                       self.children[3].add(body) or \
                       self.children[4].add(body) or \
                       self.children[5].add(body) or \
                       self.children[6].add(body) or \
                       self.children[7].add(body)

            else:  # external node; have to divide
                old = Tree3D(self.sample_x, self.sample_y, self.sample_z, None, False)
                loc = (self.cord_x, self.cord_y, self.cord_z)
                s2 = self.s / 2

                # llf = Tree3D(loc[0], loc[1], loc[2], s2, True, father_id=self.id)
                # lrf = Tree3D(loc[0]+s2, loc[1], loc[2], s2, True, father_id=self.id)
                # ulf = Tree3D(loc[0], loc[1], loc[2]+s2, s2, True, father_id=self.id)
                # urf = Tree3D(loc[0]+s2, loc[1], loc[2]+s2, s2, True, father_id=self.id)
                #
                # llb = Tree3D(loc[0], loc[1]+s2, loc[2], s2, True, father_id=self.id)
                # lrb = Tree3D(loc[0]+s2, loc[1]+s2, loc[2], s2, True, father_id=self.id)
                # ulb = Tree3D(loc[0], loc[1]+s2, loc[2]+s2, s2, True, father_id=self.id)
                # urb = Tree3D(loc[0]+s2, loc[1]+s2, loc[2]+s2, s2, True, father_id=self.id)

                llf = Tree3D(loc[0], loc[1], loc[2], s2, True, father_id=self.id)
                lrf = Tree3D(loc[0] + s2, loc[1], loc[2], s2, True, father_id=self.id)
                ulf = Tree3D(loc[0], loc[1] + s2, loc[2], s2, True, father_id=self.id)
                urf = Tree3D(loc[0] + s2, loc[1] + s2, loc[2], s2, True, father_id=self.id)

                llb = Tree3D(loc[0], loc[1], loc[2] + s2, s2, True, father_id=self.id)
                lrb = Tree3D(loc[0] + s2, loc[1], loc[2] + s2, s2, True, father_id=self.id)
                ulb = Tree3D(loc[0], loc[1] + s2, loc[2] + s2, s2, True, father_id=self.id)
                urb = Tree3D(loc[0] + s2, loc[1] + s2, loc[2] + s2, s2, True, father_id=self.id)

                self.__set_neighbors__(llf, lrf, ulf, urf, llb, lrb, ulb, urb)
                self.add(old) and self.add(body)
                #                 self.__update_COM_in__()
                return True
        return False

    def __set_neighbors__(self, llf, lrf, ulf, urf, llb, lrb, ulb, urb):
        self.children = [llf, lrf, ulf, urf, llb, lrb, ulb, urb]

    def __in_node__(self, pos):
        return self.cord_x <= pos[0] <= self.cord_x + self.s and \
               self.cord_y <= pos[1] <= self.cord_y + self.s and \
               self.cord_z <= pos[2] <= self.cord_z + self.s

    # outputs a pre traversal or whatever it's called
    def print_tree(self, text=False, plot=False, plotly=False, fig_list=None):
        if self.n == 0:  # none empty
            if text:
                log("Empty node! Cube information: " + str(self.sample_x) + " " + str(self.sample_y) + " " +
                    str(self.sample_z) + " " + str(self.s))
            return True
        elif self.children[0] != None:  # internal node
            if text:
                log("Internal node! Center of Cube information: " + str(self.sample_x) + " " + str(self.sample_y) +
                    " " + str(self.sample_z) + " " + str(self.s))
            # if plot:
            #     plt.vlines(self.cord_x + self.s / 2, self.cord_y, self.cord_y + self.s, colors='k', alpha=1, linewidth=0.6)
            #     plt.hlines(self.cord_y + self.s / 2, self.cord_x, self.cord_x + self.s, colors='k', alpha=1, linewidth=0.6)
            if plotly:
                if fig_list is None:
                    raise RuntimeError("You need to give a figure list.")
                linex = go.Scatter3d(x=[self.cord_x, self.cord_x + self.s],
                                     y=[self.cord_y + self.s / 2, self.cord_y + self.s / 2],
                                     z=[self.cord_z + self.s / 2, self.cord_z + self.s / 2],
                                     mode="lines",
                                     line=dict(color="red", width=4),
                                     opacity=0.5)
                liney = go.Scatter3d(x=[self.cord_x + self.s / 2, self.cord_x + self.s / 2],
                                     y=[self.cord_y, self.cord_y + self.s],
                                     z=[self.cord_z + self.s / 2, self.cord_z + self.s / 2],
                                     mode="lines",
                                     line=dict(color="green", width=4),
                                     opacity=0.5)
                linez = go.Scatter3d(x=[self.cord_x + self.s / 2, self.cord_x + self.s / 2],
                                     y=[self.cord_y + self.s / 2, self.cord_y + self.s / 2],
                                     z=[self.cord_z, self.cord_z + self.s],
                                     mode="lines",
                                     line=dict(color="blue", width=4),
                                     opacity=0.5)
                n = 2
                mycolorscale = [[0.0, 'rgb(200,200,200)'], [1.0, 'rgb(200,200,200)']]
                x = np.linspace(self.cord_x, self.cord_x + self.s, n)
                y = np.linspace(self.cord_y, self.cord_y + self.s, n)
                z = np.linspace(self.cord_z, self.cord_z + self.s, n)

                slicex = (self.cord_x + self.s / 2) * np.ones(n ** 2).reshape(n, n)
                slicey = (self.cord_y + self.s / 2) * np.ones(n ** 2).reshape(n, n)
                slicez = (self.cord_z + self.s / 2) * np.ones(n ** 2).reshape(n, n)

                y_, z_ = np.meshgrid(y, z)
                planex = go.Surface(x=slicex, y=y_, z=z_, colorscale=mycolorscale, showscale=False, opacity=0.2)

                x_, z_ = np.meshgrid(x, z)
                planey = go.Surface(x=x_, y=slicey, z=z_, colorscale=mycolorscale, showscale=False, opacity=0.2)

                x_, y_ = np.meshgrid(x, y)
                planez = go.Surface(x=x_, y=y_, z=slicez, colorscale=mycolorscale, showscale=False, opacity=0.2)

                fig_list.append(linex)
                fig_list.append(liney)
                fig_list.append(linez)
                fig_list.append(planex)
                fig_list.append(planey)
                fig_list.append(planez)
            # plt.scatter(self.sample_x, self.sample_y, s=10, marker="x", c="black")
            return self.children[0].print_tree(text, plot, plotly, fig_list) and \
                   self.children[1].print_tree(text, plot, plotly, fig_list) and \
                   self.children[2].print_tree(text, plot, plotly, fig_list) and \
                   self.children[3].print_tree(text, plot, plotly, fig_list) and \
                   self.children[4].print_tree(text, plot, plotly, fig_list) and \
                   self.children[5].print_tree(text, plot, plotly, fig_list) and \
                   self.children[6].print_tree(text, plot, plotly, fig_list) and \
                   self.children[7].print_tree(text, plot, plotly, fig_list)
        else:  # external node
            if text:
                log(
                    "External node! Particle information: " + str(self.sample_x) + " " + str(self.sample_y) + " " + str(
                        self.n))
            return True

    # builds tree from positions
    @staticmethod
    def build_tree(n, pos):
        head = Tree3D(0.0 - 1e-4, 0.0 - 1e-4, 0.0 - 1e-4, 1.0 + 2e-4, True)
        for i in range(n):
            head.add(Tree3D(pos[i, 0], pos[i, 1], pos[i, 2], None, False))
        return head
