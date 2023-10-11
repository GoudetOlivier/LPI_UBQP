from __future__ import print_function, absolute_import
from numba import cuda
import numpy
import math
import numba as nb
import numpy as np
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

size = -1
size2 = -1
k = -1
E = -1


@cuda.jit
def compute_nearest_neighbor_partition_crossover(rng_states, size_pop, size_sub_pop, tColor, fils, closest_individuals,  vect_adj, vect_nb_adj, A, vect_nb_components, matrix_component_id, size_components ):
    
    d = cuda.grid(1)

    if (d < size_pop):

        idx_in_pop = d%size_sub_pop
        num_pop = d//size_sub_pop

        e = int(num_pop * size_sub_pop + closest_individuals[num_pop,idx_in_pop,0])


        G1 = nb.cuda.local.array((size), nb.int32)
        G2 = nb.cuda.local.array((size), nb.int32)
        
        
        
        parent1 = nb.cuda.local.array((size), nb.int8)
        parent2 = nb.cuda.local.array((size), nb.int8)  

        for j in range(size):
            
            parent1[j] = tColor[d, j]
            parent2[j] = tColor[e, j]

        component_id = nb.cuda.local.array((size), nb.int16)
    

        for x in range(size):
            if(parent1[x] != parent2[x]):
                component_id[x] = 0    
            else:
                component_id[x] = -1  
                fils[d, x] = parent1[x]
                
                
        

        component_number = 0;
        S = nb.cuda.local.array((size), nb.int16)
        sizeS = -1
        
        
        for x in range(size):
            

            
            if(component_id[x] == 0):
                
                component_number += 1
                component_id[x] = component_number
                
                sizeS += 1
                S[sizeS] = x
                    
                        
                        
            while(sizeS > -1):
                

                j = int(S[sizeS])
                sizeS -= 1

                
                nb_adj = int(vect_nb_adj[j])
                
                for k in range(nb_adj):
                    
                    b = int(vect_adj[j, k])
                    

                        
                    if(component_id[b] == 0):
                        
                        component_id[b] = component_number
                        
                        sizeS += 1
                        S[sizeS] = b
                           

                            
        vect_nb_components[d] = component_number


                            

        for l in range(component_number):
            
            G1[l] = 0
            G2[l] = 0
            
            
        for x in range(size):
                
            if(component_id[x] > 0 ):
                
                l = component_id[x]
                    
                G1[l-1] += A[x, x] * parent1[x]
                G2[l-1] += A[x, x] * parent2[x]
                        

                nb_adj = int(vect_nb_adj[x])
                
                for k in range(nb_adj):
                        
                    y = int(vect_adj[x, k])
                    
                    if component_id[y] == l :

                        
                        G1[l-1] += A[x, y] * parent1[x]*parent1[y]/2
                        G2[l-1] += A[x, y] * parent2[x]*parent2[y]/2
                        
                    if component_id[y] == -1 :
                            
                        G1[l-1] += A[x, y] * parent1[x]*parent1[y]
                        G2[l-1] += A[x, y] * parent2[x]*parent2[y]                        
            
            
                
        for x in range(size):
                    
            l = component_id[x]
            
            if(l > 0):
                
                size_components[d, l-1] += 1
                
                if(G1[l-1] < G2[l-1]):
                    
                    fils[d, x] = parent1[x]
                    
                else:
                    
                    fils[d, x] = parent2[x]
                    
            matrix_component_id[d,x] = component_id[x]
                    

            
                

                    
                    
                    
                    
                
                
        
    
    
    
    
    

@cuda.jit
def compute_nearest_neighbor_crossovers_UX_cluster(rng_states, size_pop, size_sub_pop, tColor, fils, indices):
    d = cuda.grid(1)

    if (d < size_pop):

        idx_in_pop = d % size_sub_pop
        num_pop = d // size_sub_pop

        idx1 = int(d)
        idx2 = int(num_pop * size_sub_pop + indices[num_pop, idx_in_pop, 0])

        proba = 0.5

        for x in range(size):

            if (tColor[idx1, x] == tColor[idx2, x]):

                fils[d, x] = tColor[idx1, x]

            else:

                r = xoroshiro128p_uniform_float32(rng_states, d)

                if (r < proba):

                    fils[d, x] = tColor[idx1, x]

                else:

                    fils[d, x] = tColor[idx2, x]
                    
                    

@cuda.jit
def compute_nearest_neighbor_crossovers_MX_cluster(rng_states, size_pop, size_sub_pop, tColor, fils, indices):
    d = cuda.grid(1)

    if (d < size_pop):

        idx_in_pop = d % size_sub_pop
        num_pop = d // size_sub_pop

        idx1 = int(d)
        idx2 = int(num_pop * size_sub_pop + indices[num_pop, idx_in_pop, 0])


        diff = nb.cuda.local.array((size), nb.int8)

        HD = 0
        for x in range(size):

            fils[d, x] = tColor[idx1, x]

            if (tColor[idx1, x] != tColor[idx2, x]):
                diff[x] = 1
                HD += 1
            else:
                diff[x] = 0
                
        
        start = int(size * xoroshiro128p_uniform_float32(rng_states, d))
        
        cpt = 0
        
        for w in range(size):

            x = (start + w)%size

            if(diff[x] == 1 and cpt < HD/2):
                
                fils[d, x] = tColor[idx2, x]  
                
                cpt = cpt + 1
                
                


@cuda.jit
def compute_cluster_pathRelinking(rng_states, size_pop, size_sub_pop, tColor, fils, closest_individuals, gamma1, gamma2,
                                  A, vect_adj, vect_nb_adj):
    d = cuda.grid(1)

    bestFit = 9999

    if (d < size_pop):

        idx_in_pop = d % size_sub_pop
        num_pop = d // size_sub_pop

        e = int(num_pop * size_sub_pop + closest_individuals[num_pop, idx_in_pop, 0])

        current_child = nb.cuda.local.array((size), nb.int8)
        end = nb.cuda.local.array((size), nb.int8)

        for j in range(size):
            current_child[j] = tColor[d, j]
            end[j] = tColor[e, j]

        diff = nb.cuda.local.array((size), nb.int8)

        HD = 0
        for x in range(size):

            if (current_child[x] != end[x]):
                diff[x] = 1
                HD += 1
            else:
                diff[x] = 0

        gamma = nb.cuda.local.array((size), nb.int8)

        for x in range(size):
            gamma[x] = 0

        for x in range(size):

            gamma[x] += A[x, x]

            nb_adj = int(vect_nb_adj[x])

            for j in range(nb_adj):
                y = int(vect_adj[x, j])

                gamma[x] += A[x, y] * current_child[y]

            if (current_child[x] == 1):
                gamma[x] = -gamma[x]

        f = 0
        for x in range(size):
            for y in range(x + 1):
                f += A[x, y] * current_child[x] * current_child[y]

        f_best = f

        for i in range(int(gamma2 * HD)):

            trouve = 1
            best_delta = 9999
            best_x = -1

            for x in range(size):

                if (diff[x] == 1):

                    delt = gamma[x]

                    if (delt < best_delta):

                        best_x = x
                        best_delta = delt

                        trouve = 1

                    elif (delt == best_delta):

                        trouve += 1

                        if (int(trouve * xoroshiro128p_uniform_float32(rng_states, d)) == 0):
                            best_x = x

            diff[best_x] = 0
            f += best_delta

            gamma[best_x] = - gamma[best_x]

            old_value = current_child[best_x]

            nb_adj = int(vect_nb_adj[best_x])

            for j in range(nb_adj):

                y = int(vect_adj[best_x, j])

                if (old_value == current_child[y]):

                    gamma[y] += A[best_x, y]
                else:
                    gamma[y] -= A[best_x, y]

            current_child[best_x] = 1 - current_child[best_x]

            if (i == int(gamma1 * HD) or (f < f_best and i > int(gamma1 * HD))):

                f_best = f

                for j in range(size):
                    fils[d, j] = current_child[j]




@cuda.jit
def compute_nearest_neighbor_crossovers_XT7_cluster(rng_states, size_pop, size_sub_pop, max_iter, beta, A, tColor, fils, closest_individuals, alpha_div, vect_adj, vect_nb_adj, vect_pairwise_adj, vect_pairwise_nb_adj, vect_correspondance):
    
    
    d = cuda.grid(1)

    if (d < size_pop):

        idx_in_pop = d%size_sub_pop
        num_pop = d//size_sub_pop

        e = int(num_pop * size_sub_pop + closest_individuals[num_pop,idx_in_pop,0])

        tColor_local = nb.cuda.local.array((size), nb.int8)
        end = nb.cuda.local.array((size), nb.int8)  

        for j in range(size):
            
            tColor_local[j] = tColor[d, j]
            end[j] = tColor[ e, j]

        diff = nb.cuda.local.array((size), nb.int8)
             
        HD = 0
        
        distFirstParent = 0

        for x in range(size):
            if(tColor_local[x] != end[x]):
                diff[x] = 1    
                HD += 1
            else:
                diff[x] = 0

        gamma = nb.cuda.local.array((size), nb.int8)
        omicron = nb.cuda.local.array((size2), nb.int8)
        tabuTenure = nb.cuda.local.array((size2), nb.uint16)

        for x in range(size):
            fils[d, x] = tColor_local[x]  
            gamma[x] = 0

        for x in range(size2):   
            tabuTenure[x] = 0        

        for x in range(size):

            gamma[x] += A[x, x]
            nb_adj  = int(vect_nb_adj[x])
        
            for j in range(nb_adj):
                y = int(vect_adj[x,j])
                gamma[x] += A[x, y]*tColor_local[y]
            
            if(tColor_local[x] == 1):
                gamma[x] = -gamma[x]

        sizeTT = 0

        for x in range(size2):
            
            v = vect_correspondance[x]
            
            i = v//size
            j = v%size
            
            if(diff[i] == 1 and diff[j] == 1):
                
                sizeTT += 1

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

                v = vect_correspondance[x]
                
                x1 = v//size
                x2 = v%size

                if(diff[x1] == 1 and diff[x2] == 1):

                    bonusDiff = 0
                        
                    if((tColor_local[x1] == tColor[d, x1] and distFirstParent < HD/2) or (tColor_local[x1] != tColor[d, x1] and distFirstParent > HD/2)):
                        bonusDiff = bonusDiff  - beta
                    else:
                        bonusDiff = bonusDiff  + beta
                   
                    if(x1 != x2):
                        if((tColor_local[x2] == tColor[d, x2] and distFirstParent < HD/2) or (tColor_local[x2] != tColor[d, x2] and distFirstParent > HD/2)):
                            bonusDiff = bonusDiff  - beta
                        else:
                            bonusDiff = bonusDiff  + beta

                    delt = omicron[x] + bonusDiff
        
                    if ((tabuTenure[x] <= iter) or (delt + f < f_best)):

                        if (delt < best_delta):
                            best_x = x
                            best_delta = delt
                            trouve  = 1

                        elif(delt == best_delta):

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
            
            if(tColor_local[best_i] == tColor[d, best_i]):
                distFirstParent = distFirstParent + 1
            else:
                distFirstParent = distFirstParent - 1            

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

                if(tColor_local[best_j] == tColor[d, best_j]):
                    distFirstParent = distFirstParent + 1
                else:
                    distFirstParent = distFirstParent - 1        
                    
                    
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

            tabuTenure[best_x] = int(10 * xoroshiro128p_uniform_float32(rng_states, d)) + int(sizeTT/alpha_div) + iter

            if (f < f_best):
                f_best = f
                for a in range(size):
                    fils[d, a] = tColor_local[a]


@cuda.jit
def compute_nearest_neighbor_crossovers_XT8_cluster(rng_states, size_pop, size_sub_pop, max_iter, beta, A, tColor, fils,
                                                    closest_individuals, alpha_div, vect_adj, vect_nb_adj):
    d = cuda.grid(1)

    if (d < size_pop):

        idx_in_pop = d % size_sub_pop
        num_pop = d // size_sub_pop

        e = int(num_pop * size_sub_pop + closest_individuals[num_pop, idx_in_pop, 0])

        tColor_local = nb.cuda.local.array((size), nb.int8)
        tBest_local = nb.cuda.local.array((size), nb.int8)

        end = nb.cuda.local.array((size), nb.int8)

        for j in range(size):
            tColor_local[j] = tColor[d, j]
            end[j] = tColor[e, j]

        diff = nb.cuda.local.array((size), nb.int8)

        HD = 0

        distFirstParent = 0

        for x in range(size):

            if (tColor_local[x] != end[x]):
                diff[x] = 1
                HD += 1
            else:
                diff[x] = 0

        gamma = nb.cuda.local.array((size), nb.int16)
        tabuTenure = nb.cuda.local.array((size), nb.uint16)

        for x in range(size):
            fils[d, x] = tColor_local[x]
            gamma[x] = 0
            tabuTenure[x] = 0

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

        real_fit = f

        f_best = f

        for iter in range(max_iter):

            best_delta = 9999
            best_x = -1
            trouve = 1

            for x in range(size):

                if (diff[x] == 1):
                    if ((tColor_local[x] != end[x] and distFirstParent < HD / 2) or (
                            tColor_local[x] == end[x] and distFirstParent > HD / 2)):
                        delt = gamma[x] - beta
                    else:
                        delt = gamma[x] + beta

                    if ((tabuTenure[x] <= iter) or (delt + f < f_best)):

                        if (delt < best_delta):
                            best_x = x
                            best_delta = delt
                            trouve = 1

                        elif (delt == best_delta):
                            trouve += 1
                            if (int(trouve * xoroshiro128p_uniform_float32(rng_states, d)) == 0):
                                best_x = x

            f += best_delta
            real_fit += gamma[best_x]
            old_value_x = tColor_local[best_x]
            gamma[best_x] = - gamma[best_x]
            nb_i = int(vect_nb_adj[best_x])

            for k in range(nb_i):
                y = int(vect_adj[best_x, k])
                if (old_value_x == tColor_local[y]):
                    gamma[y] += A[best_x, y]
                else:
                    gamma[y] -= A[best_x, y]

            if (tColor_local[best_x] == tColor[d, best_x]):
                distFirstParent = distFirstParent + 1
            else:
                distFirstParent = distFirstParent - 1

            tColor_local[best_x] = 1 - tColor_local[best_x]

            tabuTenure[best_x] = int(10 * xoroshiro128p_uniform_float32(rng_states, d)) + int(HD / alpha_div) + iter

            if (f < f_best):
                f_best = f
                for a in range(size):
                    tBest_local[a] = tColor_local[a]

        for a in range(size):
            fils[d, a] = tBest_local[a]




@cuda.jit
def compute_nearest_neighbor_crossovers_XT8_cluster_v2(rng_states, size_pop, size_sub_pop, max_iter, beta, A, tColor, fils,
                                                    closest_individuals, alpha_div, vect_adj, vect_nb_adj):
    d = cuda.grid(1)

    if (d < size_pop):

        idx_in_pop = d % size_sub_pop
        num_pop = d // size_sub_pop

        e = int(num_pop * size_sub_pop + closest_individuals[num_pop, idx_in_pop, 0])

        tColor_local = nb.cuda.local.array((size), nb.int8)
        tBest_local = nb.cuda.local.array((size), nb.int8)

        end = nb.cuda.local.array((size), nb.int8)

        for j in range(size):
            tColor_local[j] = tColor[d, j]
            end[j] = tColor[e, j]

        diff = nb.cuda.local.array((size), nb.int8)

        HD = 0

        distFirstParent = 0

        for x in range(size):

            if (tColor_local[x] != end[x]):
                diff[x] = 1
                HD += 1
            else:
                diff[x] = 0

        gamma = nb.cuda.local.array((size), nb.int32)
        tabuTenure = nb.cuda.local.array((size), nb.uint16)

        for x in range(size):
            fils[d, x] = tColor_local[x]
            gamma[x] = 0
            tabuTenure[x] = 0

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

        real_fit = f

        f_best = f

        for iter in range(max_iter):

            best_delta = 9999
            best_x = -1
            trouve = 1

            for x in range(size):

                if (diff[x] == 1):
                    if ((tColor_local[x] != end[x] and distFirstParent < HD / 2) or (
                            tColor_local[x] == end[x] and distFirstParent > HD / 2)):
                        delt = gamma[x] - beta
                    else:
                        delt = gamma[x] + beta

                    if ((tabuTenure[x] <= iter) or (delt + f < f_best)):

                        if (delt < best_delta):
                            best_x = x
                            best_delta = delt
                            trouve = 1

                        elif (delt == best_delta):
                            trouve += 1
                            if (int(trouve * xoroshiro128p_uniform_float32(rng_states, d)) == 0):
                                best_x = x

            f += best_delta
            real_fit += gamma[best_x]
            old_value_x = tColor_local[best_x]
            gamma[best_x] = - gamma[best_x]
            nb_i = int(vect_nb_adj[best_x])

            for k in range(nb_i):
                y = int(vect_adj[best_x, k])
                if (old_value_x == tColor_local[y]):
                    gamma[y] += A[best_x, y]
                else:
                    gamma[y] -= A[best_x, y]

            if (tColor_local[best_x] == tColor[d, best_x]):
                distFirstParent = distFirstParent + 1
            else:
                distFirstParent = distFirstParent - 1

            tColor_local[best_x] = 1 - tColor_local[best_x]

            tabuTenure[best_x] = int(10 * xoroshiro128p_uniform_float32(rng_states, d)) + int(HD / alpha_div) + iter

            if (f < f_best):
                f_best = f
                for a in range(size):
                    tBest_local[a] = tColor_local[a]

        for a in range(size):
            fils[d, a] = tBest_local[a]
            

@cuda.jit
def compute_nearest_neighbor_crossovers_XT11_cluster(rng_states, size_pop, size_sub_pop, max_iter, beta, A, tColor, fils, closest_individuals, alpha_div, vect_adj, vect_nb_adj, vect_pairwise_adj, vect_pairwise_nb_adj, vect_correspondance):
    
    
    d = cuda.grid(1)

    if (d < size_pop):

        idx_in_pop = d%size_sub_pop
        num_pop = d//size_sub_pop

        e = int(num_pop * size_sub_pop + closest_individuals[num_pop,idx_in_pop,0])

        tColor_local = nb.cuda.local.array((size), nb.int8)
        tBest_local = nb.cuda.local.array((size), nb.int8)   
        end = nb.cuda.local.array((size), nb.int8)  

        for j in range(size):
            
            tColor_local[j] = tColor[d, j]
            end[j] = tColor[ e, j]

        diff = nb.cuda.local.array((size), nb.int8)
             
        HD = 0
        
        distFirstParent = 0

        for x in range(size):
            if(tColor_local[x] != end[x]):
                diff[x] = 1    
                HD += 1
            else:
                diff[x] = 0

        gamma = nb.cuda.local.array((size), nb.int8)
        omicron = nb.cuda.local.array((size2), nb.int8)
        tabuTenure = nb.cuda.local.array((size2), nb.uint16)

        for x in range(size):
            fils[d, x] = tColor_local[x]  
            gamma[x] = 0

        for x in range(size2):   
            tabuTenure[x] = 0        

        for x in range(size):

            gamma[x] += A[x, x]
            nb_adj  = int(vect_nb_adj[x])
        
            for j in range(nb_adj):
                y = int(vect_adj[x,j])
                gamma[x] += A[x, y]*tColor_local[y]
            
            if(tColor_local[x] == 1):
                gamma[x] = -gamma[x]

        sizeTT = 0

        for x in range(size2):
            
            v = vect_correspondance[x]
            
            i = v//size
            j = v%size
            
            if(diff[i] == 1 and diff[j] == 1):
                
                sizeTT += 1

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

                v = vect_correspondance[x]
                
                x1 = v//size
                x2 = v%size

                if(diff[x1] == 1 and diff[x2] == 1):

                    bonusDiff = 0
                        
                    if((tColor_local[x1] != end[x1] and distFirstParent < HD/2) or (tColor_local[x1] == end[x1] and distFirstParent > HD/2)):
                        bonusDiff = bonusDiff  - beta
                    else:
                        bonusDiff = bonusDiff  + beta
                   
                    if(x1 != x2):
                        if((tColor_local[x2] != end[x2] and distFirstParent < HD/2) or (tColor_local[x2] == end[x2] and distFirstParent > HD/2)):
                            bonusDiff = bonusDiff  - beta
                        else:
                            bonusDiff = bonusDiff  + beta

                    delt = omicron[x] + bonusDiff
        
                    if ((tabuTenure[x] <= iter) or (delt + f < f_best)):

                        if (delt < best_delta):
                            best_x = x
                            best_delta = delt
                            trouve  = 1

                        elif(delt == best_delta):

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
            
            if(tColor_local[best_i] == tColor[d, best_i]):
                distFirstParent = distFirstParent + 1
            else:
                distFirstParent = distFirstParent - 1            

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

                if(tColor_local[best_j] == tColor[d, best_j]):
                    distFirstParent = distFirstParent + 1
                else:
                    distFirstParent = distFirstParent - 1        
                    
                    
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

            tabuTenure[best_x] = int(10 * xoroshiro128p_uniform_float32(rng_states, d)) + int(sizeTT/alpha_div) + iter

            if (f < f_best):
                f_best = f
                for a in range(size):
                    tBest_local[a] = tColor_local[a]


        for a in range(size):
            fils[d, a] = tBest_local[a]



