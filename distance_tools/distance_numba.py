from __future__ import print_function, absolute_import
from numba import cuda
import numpy
import math
import numba as nb
import numpy as np
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

size = -1
k = -1
kplus1 = -1


# CUDA kplus1ernel : compute distance matrix between two set of solutions for each pop
@cuda.jit
def computeMatrixDistance_Hamming_cluster(nb_clusters, size1, size2, matrixDistance, tColor1, tColor2):
    d = cuda.grid(1)

    if (d < size1 * size2 * nb_clusters):

        num_pop = d // (size1 * size2)
        idx_in_pop = d % (size1 * size2)

        idx1 = int(idx_in_pop // size2)
        idx2 = int(idx_in_pop % size2)

        start_pop1 = num_pop * size1
        start_pop2 = num_pop * size2

        dist = 0

        for x in range(size):

            if (tColor1[int(start_pop1 + idx1), x] != tColor2[int(start_pop2 + idx2), x]):
                dist += 1

        matrixDistance[int(num_pop), int(idx1), int(idx2)] = dist



# CUDA kplus1ernel : compute distance matrix between two set of solutions for each pop
@cuda.jit
def computeMatrixDistance_Hamming_cluster_v2(nb_clusters, size1, size2, matrixDistance, tColor1, tColor2):
    d = cuda.grid(1)

    if (d < size1 * size2 * nb_clusters):

        num_pop = d // (size1 * size2)
        idx_in_pop = d % (size1 * size2)

        idx1 = int(idx_in_pop // size2)
        idx2 = int(idx_in_pop % size2)

        start_pop1 = num_pop * size1

        dist = 0

        for x in range(size):

            if (tColor1[int(start_pop1 + idx1), x] != tColor2[int(num_pop), idx2 , x]):
                dist += 1

        matrixDistance[int(num_pop), int(idx1), int(idx2)] = dist



@cuda.jit
def computeSymmetricMatrixDistance_Hamming_cluster(nb_clusters, size_cluster, matrixDistance, tColor):
    d = cuda.grid(1)

    if (d < (size_cluster * (size_cluster - 1) / 2 * nb_clusters)):

        num_pop = d // (size_cluster * (size_cluster - 1) / 2)
        idx_in_pop = d % (size_cluster * (size_cluster - 1) / 2)

        # Get upper triangular matrix indices from thread index !
        idx1 = int(size_cluster - 2 - int(
            math.sqrt(-8.0 * idx_in_pop + 4.0 * size_cluster * (size_cluster - 1) - 7) / 2.0 - 0.5))
        idx2 = int(idx_in_pop + idx1 + 1 - size_cluster * (size_cluster - 1) / 2 + (size_cluster - idx1) * (
                    (size_cluster - idx1) - 1) / 2)

        start_pop = num_pop * size_cluster

        dist = 0

        for x in range(size):

            if (tColor[int(start_pop + idx1), x] != tColor[int(start_pop + idx2), x]):
                dist += 1

        matrixDistance[int(num_pop), int(idx1), int(idx2)] = dist
        matrixDistance[int(num_pop), int(idx2), int(idx1)] = dist





@cuda.jit
def computeSymmetricMatrixDistance_Hamming_cluster_v2(nb_clusters, size_cluster, matrixDistance, tColor):
    d = cuda.grid(1)

    if (d < (size_cluster * (size_cluster - 1) / 2 * nb_clusters)):

        num_pop = d // (size_cluster * (size_cluster - 1) / 2)
        idx_in_pop = d % (size_cluster * (size_cluster - 1) / 2)

        # Get upper triangular matrix indices from thread index !
        idx1 = int(size_cluster - 2 - int(
            math.sqrt(-8.0 * idx_in_pop + 4.0 * size_cluster * (size_cluster - 1) - 7) / 2.0 - 0.5))
        idx2 = int(idx_in_pop + idx1 + 1 - size_cluster * (size_cluster - 1) / 2 + (size_cluster - idx1) * (
                    (size_cluster - idx1) - 1) / 2)

        dist = 0

        for x in range(size):
            if (tColor[int(num_pop), int(idx1), x] != tColor[int(num_pop), int(idx2), x]):
                dist += 1

        matrixDistance[int(num_pop), int(idx1), int(idx2)] = dist
        matrixDistance[int(num_pop), int(idx2), int(idx1)] = dist
