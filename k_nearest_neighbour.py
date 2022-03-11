#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Name:      Della Balda, Vincente
   Email:     vincente.dellabalda@uzh.ch
   Date:      07 March, 2022
   Kurs:      ESC202
   Semester:  FS22
   Week:      3
   Thema:     k Nearest Neighbours
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from heapq import heappush, heappop, heapreplace


class Particle:
    def __init__(self, r):
        self.r = r  #  position
        self.rho = 0.0

    def __repr__(self):
        return str(self.r)


class Cell:
    def __init__(self, rLow, rHigh, iLower, iUpper):
        self.rLow = rLow  # lower left corner
        self.rHigh = rHigh  # upper right corner
        self.center = (rLow + rHigh)/2
        self.iLower = iLower  # index into global array A
        self.iUpper = iUpper  # index into global array A
        self.pLower = None  # next left cell
        self.pUpper = None  # next right cell

    def __repr__(self):
        return str(self.rLow) + str(self.rHigh)

    def isleaf(self):
        """
        Check if cell is a leaf cell.
        :return: bool
        """
        if self.pLower is None and self.pUpper is None:
            return True

    def distance_to_cell2(self, r):
        """
        Calculate the minimal distance between a cell and point.
        :param r: point
        :return: float, squared distance between cell and point
        """
        dist_2 = 0
        for d in range(2):
            t = abs(self.center[d] - r[d]) - (self.rHigh[d] - self.center[d])
            if t > 0:
                dist_2 += t**2
        return dist_2

    def distance_to_cell2_offset(self, r, offset):
        """
        Calculate the minimal distance between a cell and point including periodic boundary conditions.
        :param r: point
        :param offset: offset
        :return: float, squared distance between cell and point
        """

        dist_2 = 0
        for d in range(2):
            t = abs(self.center[d] - (r[d] + offset[d])) - (self.rHigh[d] - self.center[d])
            if t > 0:
                dist_2 += t ** 2
        return dist_2


# Functions_____________________________________________________________________

def partition(particles, i, j, split_value, dim):
    """
    Partitions array in one dimension according to a split value and returns the partition index.

    :param particles: list of particles
    :param i: index of left-most entry in data array
    :param j: index of right-most entry in data array
    :param split_value: array of N values to partition by
    :param dim: int specifying dimension to partition in
    :return: None if data empty, else partition index
    """
    #  test if array empty
    if not particles:
        return None

    while i < j:
        # The list of particles is scanned from both sides to find a pair of particles in the wrong partition.
        if particles[i].r[dim] < split_value:
            i += 1
        elif particles[j].r[dim] >= split_value:
            j -= 1
        # If an incorrect pair of particles is found, they are swapped into the right partition.
        else:
            tmp = particles[i]
            particles[i] = particles[j]
            particles[j] = tmp

    #  assign partition index
    if particles[j].r[dim] < split_value:
        return i + 1
    else:
        return i


def tree_build(A, root, dim):
    """
    Builds a binary tree from a given root cell by partitioning a global list of particles.
    :param A: global list of particles
    :param root: initial cell containing all particles
    :param dim: dimension to partition by
    :return:
    """
    v = 0.5 * (root.rLow[dim] + root.rHigh[dim])
    s = partition(A, root.iLower, root.iUpper, v, dim)

    # New cell bounds are set depending on the dimension.
    if dim == 0:
        rLow_Lower = root.rLow
        rHigh_Lower = np.array([v, root.rHigh[1]])
        rLow_Upper = np.array([v, root.rLow[1]])
        rHigh_Upper = root.rHigh
    else:
        rLow_Lower = root.rLow
        rHigh_Lower = np.array([root.rHigh[0], v])
        rLow_Upper = np.array([root.rLow[0], v])
        rHigh_Upper = root.rHigh

    # The left cell is generated if a left partition exists and the branching continued.
    if s > root.iLower:
        cLow = Cell(rLow_Lower, rHigh_Lower, root.iLower, s-1)
        root.pLower = cLow
        if len(A[root.iLower:s]) > 8:
            tree_build(A, cLow, 1-dim)

    # The right cell is generated if a right partition exists and the branching continued.
    if s <= root.iUpper:
        cHigh = Cell(rLow_Upper, rHigh_Upper, s, root.iUpper)
        root.pUpper = cHigh
        if len(A[s:root.iUpper+1]) > 8:
            tree_build(A, cHigh, 1-dim)


def plot_leafs(A, node, cmap=cm.get_cmap('prism')):
    """
    Walks the tree starting from a node and generates a scatter plot of all the particles
    color-coded depending on their leaf cell.

    :param A: global list of particles
    :param node: node of which all leaf cells shall be considered
    :param cmap: colormap to be used for the scatter plot
    :return:
    """
    if node:
        plot_leafs(A, node.pLower)
        if node.isleaf():
            # plot a rectangle illustrating the cell
            plt.plot([node.rLow[0], node.rHigh[0], node.rHigh[0], node.rLow[0], node.rLow[0]],
                     [node.rLow[1], node.rLow[1], node.rHigh[1], node.rHigh[1], node.rLow[1]],
                     linewidth=0.5, color='black')
            # only plot particles if there are particles in the cell
            if not (node.iUpper == node.iLower):
                particles = np.array([x.r for x in A[node.iLower:node.iUpper+1]])
                x_coord = [x[0] for x in particles]
                y_coord = [x[1] for x in particles]
                center_mean = np.mean(node.center)
                plt.scatter(x_coord, y_coord, s=2, color=cmap(center_mean))
        plot_leafs(A, node.pUpper)


def ballwalk(A, cell, r, rmax2):
    """
    Using a partitioned list of particles and a binary tree,
    recursively count the number of particles inside a ball with radius rmax around a chosen point r.

    :param A: list of particles
    :param cell: root cell of the binary tree
    :param r: midpoint of the search ball
    :param rmax2: squared maximum search radius around r
    :return : number of particles
    """
    counter = 0  # particle counter

    # If the cell is a leaf node its leaf particles which are within the ball are counted.
    if cell.isleaf():
        for particle in A[cell.iLower:cell.iUpper+1]:
            if np.linalg.norm(r-particle.r)**2 < rmax2:
                counter += 1

    # If the cell is not a leaf node the distance between its children and the ball is checked.
    # In case the child is within the ball, it is inspected further with the ballwalk function.
    else:
        if cell.pLower and cell.pLower.distance_to_cell2(r) < rmax2:
            counter += ballwalk(A, cell.pLower, r, rmax2)
        if cell.pUpper and cell.pUpper.distance_to_cell2(r) < rmax2:
            counter += ballwalk(A, cell.pUpper, r, rmax2)

    return counter


def replace_furthest(heap, dist2, particle):
    heapreplace(heap, (-dist2, particle))


def max2(heap):
    new_max = -heap[0][0]
    return new_max


def k_nearest_neighbour(A, r, search_radius2, cell_heap, particle_heap, counter):
    if not cell_heap:
        return particle_heap

    cell = heappop(cell_heap)[2]
    if cell.isleaf():
        for particle in A[cell.iLower:cell.iUpper+1]:
            dist2 = np.linalg.norm(r - particle.r)**2
            if dist2 < search_radius2:
                replace_furthest(particle_heap, dist2, particle)
                search_radius2 = max2(particle_heap)

        particle_heap = k_nearest_neighbour(A, r, search_radius2, cell_heap, particle_heap, counter)

    else:
        if cell.pLower:
            lower_dist2 = cell.pLower.distance_to_cell2(r)
            if lower_dist2 < search_radius2:
                heappush(cell_heap, (lower_dist2, counter, cell.pLower))
                counter += 1
        if cell.pUpper:
            upper_dist2 = cell.pUpper.distance_to_cell2(r)
            if upper_dist2 < search_radius2:
                heappush(cell_heap, (upper_dist2, counter, cell.pUpper))
                counter += 1
        particle_heap = k_nearest_neighbour(A, r, search_radius2, cell_heap, particle_heap, counter)

    return particle_heap


def k_nearest_neighbour_2(A, r, search_radius2, cell_heap, particle_heap, counter, offset):
    if not cell_heap:
        return particle_heap

    cell = heappop(cell_heap)[2]
    if cell.isleaf():
        for particle in A[cell.iLower:cell.iUpper+1]:
            dist2 = np.linalg.norm(r - particle.r + offset)**2
            if dist2 < search_radius2:
                replace_furthest(particle_heap, dist2, particle)
                search_radius2 = max2(particle_heap)

        particle_heap = k_nearest_neighbour_2(A, r, search_radius2, cell_heap, particle_heap, counter, offset)

    else:
        if cell.pLower:
            lower_dist2 = cell.pLower.distance_to_cell2_offset(r, offset)
            if lower_dist2 < search_radius2:
                heappush(cell_heap, (lower_dist2, counter, cell.pLower))
                counter += 1
        if cell.pUpper:
            upper_dist2 = cell.pUpper.distance_to_cell2_offset(r, offset)
            if upper_dist2 < search_radius2:
                heappush(cell_heap, (upper_dist2, counter, cell.pUpper))
                counter += 1
        particle_heap = k_nearest_neighbour_2(A, r, search_radius2, cell_heap, particle_heap, counter, offset)

    return particle_heap


def top_hat_density(A, root, particle, k=32):
    cell_heap = []
    particle_heap = []
    sentinel = (-1, None)
    search_radius2 = 0.5
    r = particle.r
    for i in range(k):
        heappush(particle_heap, sentinel)
    heappush(cell_heap, (0, 0, root))
    offsets = [np.array([-1, 1]), np.array([0, 1]), np.array([1, 1]),
               np.array([-1, 0]), np.array([0, 0]), np.array([1, 0]),
               np.array([-1, -1]), np.array([0, -1]), np.array([1, -1])]
    for offset in offsets:
        particle_heap = k_nearest_neighbour(A, r+offset, search_radius2, cell_heap, particle_heap, 0)
        cell_heap = []
        heappush(cell_heap, (0, 0, root))
        search_radius2 = max2(particle_heap)
    density = np.sqrt(max2(particle_heap))
    return density


def call_treebuild(A, root, dim):
    print(f"Initiating tree with {len(A)} particles.")
    tree_build(A, root, dim)
    print("Tree build accomplished.")


def generate_plot(A, node, ball_midpoint, rmax, cmap=cm.get_cmap('prism')):
    fig, ax = plt.subplots()
    plot_leafs(A, node, cmap)
    ax.add_artist(plt.Circle(tuple(ball_midpoint), rmax, fill=False))
    ax.add_artist(plt.Circle((0, 0), rmax, fill=False))
    ax.add_artist(plt.Circle((0, 1), rmax, fill=False))
    ax.add_artist(plt.Circle((1, 0), rmax, fill=False))

    plt.axis('square')
    plt.title("Nearest neighbour search algorithm")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plt.show()


def call_ballwalk(A, cell, r, rmax):
    print(f"Counting particles in ball around ({round(r[0],3)}, {round(r[1], 3)}) within a radius of {round(rmax, 3)}.")
    neighbours = ballwalk(A, cell, r, rmax**2)
    print(f"{neighbours} particles were counted.")
    return neighbours


def main():

    A = []
    n = 1000
    particles = np.random.rand(n, 2)
    for i in range(n):
        A.append(Particle(particles[i]))
    rLow = np.array([0, 0])
    rHigh = np.array([1, 1])
    iLower = 0
    iUpper = len(A)-1
    root = Cell(rLow, rHigh, iLower, iUpper)
    dim = 0

    tree_build(A, root, dim)

    x_coord = []
    #y_coord_offset = []
    y_coord = []
    densities = []

    for particle in A:
        particle.rho = top_hat_density(A, root, particle, k=32)
        x_coord.append(particle.r[0])
        #y_coord_offset.append(particle.r[1]+1)
        y_coord.append(particle.r[1])
        densities.append(particle.rho)

    fig, ax = plt.subplots()
    plt.axis('square')
    plt.title("Nearest neighbour search algorithm")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    #ax.set_facecolor((0.0, 0.0, 0.0))
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.scatter(x_coord, y_coord, s=2, c=densities, cmap="autumn")
    plt.colorbar()
    #plt.scatter(x_coord, y_coord_offset, s=2, c=densities, cmap="coolwarm_r")
    plt.show()

    """
    NN = []
    d2NN = []
    for i in range(k):
        d2min = float('inf')
        for q in A:
            if q not in NN:
                d2 = np.linalg.norm(r - q.r)**2
                if d2 < d2min:
                    d2min = d2
                    qmin = q
        NN.append(qmin)
        d2NN.append(d2min)
    """

    # uncomment following line to generate plot of the partitioned cells and the ball search
    #generate_plot(A, root, ball_midpoint, np.sqrt(search_radius2), cmap=cm.get_cmap('prism'))


# Main__________________________________________________________________________

if __name__ == '__main__':
    main()


