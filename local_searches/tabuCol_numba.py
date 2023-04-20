from __future__ import print_function, absolute_import
from numba import cuda
import numpy
import math
import numba as nb
import numpy as np
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

size = -1
sizeOneFlip = -1
sizeTwoFlip = -1
size2 = -1
k = -1
E = -1
kplus1 = -1


@cuda.jit
def tabuUBQP_oneFlip(rng_states, D, max_iter, A, tColor, vect_fit, alpha, vect_adj, vect_nb_adj):
    d = cuda.grid(1)

    if (d < D):

        f = 0

        tColor_local = nb.cuda.local.array((size), nb.int8)
        tBest_local = nb.cuda.local.array((size), nb.int8)


        gamma = nb.cuda.local.array((size), nb.int16)
        tabuTenure = nb.cuda.local.array((size), nb.uint16)

        for x in range(size):
            gamma[x] = 0
            tabuTenure[x] = 0

        for x in range(size):
            tColor_local[x] = int(tColor[d, x])

        for x in range(size):

            gamma[x] += A[x, x]

            nb_adj = int(vect_nb_adj[x])

            for j in range(nb_adj):
                y = int(vect_adj[x, j])
                gamma[x] += A[x, y] * tColor_local[y]
            if (tColor_local[x] == 1):
                gamma[x] = -gamma[x]

        f = 0
        for x in range(size):
            for y in range(x + 1):
                f += A[x, y] * tColor_local[x] * tColor_local[y]

        f_best = f

        for iter in range(max_iter):

            best_delta = 9999
            best_x = -1
            trouve = 1

            for x in range(size):

                if ((tabuTenure[x] <= iter) or (gamma[x] + f < f_best)):

                    if (gamma[x] < best_delta):
                        best_x = x
                        best_delta = gamma[x]
                        trouve = 1

                    elif (gamma[x] == best_delta):

                        trouve += 1

                        if (int(trouve * xoroshiro128p_uniform_float32(rng_states, d)) == 0):
                            best_x = x

            f += best_delta

            gamma[best_x] = - gamma[best_x]

            old_value = tColor_local[best_x]

            nb_adj = int(vect_nb_adj[best_x])

            for j in range(nb_adj):
                y = int(vect_adj[best_x, j])
                if (old_value == tColor_local[y]):
                    gamma[y] += A[best_x, y]
                else:
                    gamma[y] -= A[best_x, y]

            tColor_local[best_x] = 1 - tColor_local[best_x]

            tabuTenure[best_x] = int(10 * xoroshiro128p_uniform_float32(rng_states, d)) + int(alpha) + iter

            if (f < f_best):
                f_best = f
                for a in range(size):
                    tBest_local[a] = tColor_local[a]

        for a in range(size):
            tColor[d, a] = tBest_local[a]

        vect_fit[d] = f_best


@cuda.jit
def tabuUBQP_swap(rng_states,  D, max_iter, A, tColor,  vect_fit,  alpha, vect_adj, vect_nb_adj, vect_pairwise_adj, vect_pairwise_nb_adj, vect_correspondance):


    d = cuda.grid(1)

    if (d < D):

        tColor_local = nb.cuda.local.array((size), nb.int8)
        tBest_local = nb.cuda.local.array((size), nb.int8)

        gamma = nb.cuda.local.array((size), nb.int8)
        omicron = nb.cuda.local.array((size2), nb.int8)

        tabuTenure = nb.cuda.local.array((size2), nb.uint16)

        for x in range(size):   
            gamma[x] = 0


        for x in range(size2):   
            tabuTenure[x] = 0
            
            
        for x in range(size):
            tColor_local[x] = int(tColor[d, x])

        for x in range(size):
            
            gamma[x] += A[x, x]
            
            nb_adj  = int(vect_nb_adj[x])
            
            for j in range(nb_adj):
                y = int(vect_adj[x,j])
                gamma[x] += A[x, y]*tColor_local[y]
            
            if(tColor_local[x] == 1):
                gamma[x] = -gamma[x]

        for x in range(size2):
            
            v = vect_correspondance[x]
            
            i = v//size
            j = v%size
            
            if(i==j):
                omicron[x] = gamma[i]
            else:
                omicron[x] = gamma[i] + gamma[j]
                omicron[x] += A[i, j] * (1 - 2*tColor_local[i]  - 2*tColor_local[j] + 4*tColor_local[i]*tColor_local[j])

        f = 0
        for x in range(size):
            for y in range(x+1):
                f += A[x,y]*tColor_local[x]*tColor_local[y]

        f_best = f

        for iter in range(max_iter):

            best_delta = 9999
            best_x = -1

            trouve = 1
            
            for x in range(size2):

                if ((tabuTenure[x] <= iter) or (omicron[x] + f < f_best)):

                    if (omicron[x] < best_delta):
                        best_x = x
                        best_delta = omicron[x]
                        trouve  = 1

                    elif(omicron[x] == best_delta):

                        trouve += 1

                        if(int(trouve * xoroshiro128p_uniform_float32(rng_states, d)) == 0):

                            best_x = x



            f += best_delta

            omicron[best_x] = - omicron[best_x]
            
            best_v = vect_correspondance[best_x]
            
            best_i = best_v//size
            best_j = best_v%size

            old_value_i = tColor_local[best_i]
            old_value_j = tColor_local[best_j]

            gamma[best_i] = - gamma[best_i]

            nb_i  = int(vect_nb_adj[best_i])
            
            for k in range(nb_i):
                
                y = int(vect_adj[best_i,k])
                if (old_value_i == tColor_local[y]):           
                    gamma[y] += A[best_i,y]
                else: 
                    gamma[y] -= A[best_i,y]

            tColor_local[best_i] = 1 - tColor_local[best_i]

            if (best_j != best_i):
                
                gamma[best_j] = - gamma[best_j]
                nb_j = int(vect_nb_adj[best_j])
                
                for k in range(nb_j):
                    
                    y = int(vect_adj[best_j, k])
                    if (old_value_j == tColor_local[y]):
                        gamma[y] += A[best_j, y]
                    else:
                        gamma[y] -= A[best_j, y]

                tColor_local[best_j] = 1 - tColor_local[best_j]

            nb_adj_x = int(vect_pairwise_nb_adj[best_x])

            for k in range(nb_adj_x):

                w = int(vect_pairwise_adj[best_x, k])

                node_w = vect_correspondance[w]

                i2 = node_w// size
                j2 = node_w % size

                if(i2 == j2):
                    omicron[w] = gamma[i2]
                else:
                    omicron[w] = gamma[i2] + gamma[j2]
                    omicron[w] += A[i2, j2] * (1 - 2 * tColor_local[i2] - 2 * tColor_local[j2] + 4 * tColor_local[i2] * tColor_local[j2])
                  
            tabuTenure[best_x] = int(10 * xoroshiro128p_uniform_float32(rng_states, d)) + int(alpha) + iter

            if (f < f_best):

                f_best = f
                for a in range(size):
                    tBest_local[a] = tColor_local[a]

        for a in range(size):
            tColor[d, a] = tBest_local[a]


        vect_fit[d] = f_best
        




@cuda.jit
def tabuUBQP_oneFlip_v2(rng_states, D, max_iter, A, tColor, vect_fit, alpha, vect_adj, vect_nb_adj):
    d = cuda.grid(1)

    if (d < D):

        f = 0

        tColor_local = nb.cuda.local.array((size), nb.int8)
        tBest_local = nb.cuda.local.array((size), nb.int8)


        gamma = nb.cuda.local.array((size), nb.int32)
        tabuTenure = nb.cuda.local.array((size), nb.uint16)

        for x in range(size):
            gamma[x] = 0
            tabuTenure[x] = 0

        for x in range(size):
            tColor_local[x] = int(tColor[d, x])

        for x in range(size):

            gamma[x] += A[x, x]

            nb_adj = int(vect_nb_adj[x])

            for j in range(nb_adj):
                y = int(vect_adj[x, j])
                gamma[x] += A[x, y] * tColor_local[y]
            if (tColor_local[x] == 1):
                gamma[x] = -gamma[x]

        f = 0
        for x in range(size):
            for y in range(x + 1):
                f += A[x, y] * tColor_local[x] * tColor_local[y]

        f_best = f

        for iter in range(max_iter):

            best_delta = 9999
            best_x = -1
            trouve = 1

            for x in range(size):

                if ((tabuTenure[x] <= iter) or (gamma[x] + f < f_best)):

                    if (gamma[x] < best_delta):
                        best_x = x
                        best_delta = gamma[x]
                        trouve = 1

                    elif (gamma[x] == best_delta):

                        trouve += 1

                        if (int(trouve * xoroshiro128p_uniform_float32(rng_states, d)) == 0):
                            best_x = x

            f += best_delta

            gamma[best_x] = - gamma[best_x]

            old_value = tColor_local[best_x]

            nb_adj = int(vect_nb_adj[best_x])

            for j in range(nb_adj):
                y = int(vect_adj[best_x, j])
                if (old_value == tColor_local[y]):
                    gamma[y] += A[best_x, y]
                else:
                    gamma[y] -= A[best_x, y]

            tColor_local[best_x] = 1 - tColor_local[best_x]

            tabuTenure[best_x] = int(10 * xoroshiro128p_uniform_float32(rng_states, d)) + int(alpha) + iter

            if (f < f_best):
                f_best = f
                for a in range(size):
                    tBest_local[a] = tColor_local[a]

        for a in range(size):
            tColor[d, a] = tBest_local[a]

        vect_fit[d] = f_best
