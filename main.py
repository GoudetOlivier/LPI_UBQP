from __future__ import print_function, absolute_import
from numba import cuda
import numba as nb
import numpy as np
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import datetime
import argparse
import logging
from time import time
import crossovers.crossovers_numba
import distance_tools.distance_numba
import local_searches.tabuCol_numba
import utils.tools_numba

from distance_tools.insertion import insertion_pop
from joblib import Parallel, delayed
from numba import jit
from tqdm import tqdm


def test_python_partition_crossover(size, parent1, parent2, vect_nb_adj, vect_adj):
    


    component_id = np.zeros(size)


    for x in range(size):
        if(parent1[x] != parent2[x]):
            component_id[x] = 0    
        else:
            component_id[x] = -1            
            
    
    print(component_id)
    component_number = 0;
    S = np.zeros(size)
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

            
            #print("j : " + str(j))
            nb_adj = int(vect_nb_adj[j])
            #print("nb_adj : " + str(nb_adj))

            
            for k in range(nb_adj):
                
                b = int(vect_adj[j, k])
                
            
                if(component_id[b] == 0):
                        
                    component_id[b] = component_number
                        
                    sizeS += 1
                    S[sizeS] = b 

            


    return component_id
                                
                                

def verificationPython(component_id, vect_adj, nb_adj):
    
    maxComponent = np.max(component_id)
    print("maxComponent " + str(maxComponent))
    
    pb= False
    for i in range(component_id.shape[0]):
        
        if(component_id[i] != -1):
            
            #print("i :" + str(i))
        
            str_ = "neigh :"
        
            for j in range(int(nb_adj[i])):
            
                neigh = int(vect_adj[i,j])
            
                if(component_id[neigh] != -1):
                    str_ += str(neigh) + "-" + str(component_id[neigh]) + "__"
            
                    if(component_id[neigh] != component_id[i]):
                
                        pb = True
                        
    if pb:
        print("pb extraction components")
        



def verification_generation_child(component_id, child, parent1, parent2, vect_adj, nb_adj, size, Q):
    
    new_child = np.zeros((size))
    maxComponent = np.max(component_id)
    
    for x in range(size):
        if(parent1[x] == parent2[x]):
            new_child[x] = parent1[x]
            if(component_id[x] != -1):
                
                print("pbpbpb component")

    
    G1 = np.zeros((size))
    G2 = np.zeros((size))  
    
    for l in range(maxComponent):
        
        
        for x in range(size):
            
            if(component_id[x] == -1 or component_id[x] == l+1):
                
                
                for y in range(x+1):
                    
                    if(component_id[y] == -1 or component_id[y] == l+1):
                        
                        G1[l] += Q[x, y] * parent1[x] * parent1[y]
                        G2[l] += Q[x, y] * parent2[x] * parent2[y]
    
    for x in range(size):
        
        l = component_id[x] 
        
        if(l > 0):
            if(G1[l-1] < G2[l-1]):
            
                new_child[x] = parent1[x]
            
            else:
            
                new_child[x] = parent2[x]
            
    pb = False
    for x in range(size):

        if(int(new_child[x])  !=  int(child[x])):
            
            pb = True
            
    
    if(pb):
        print("pbpbpb Child")
                        

        
        
        

@jit(nopython=True)
def get_vect_pairwise_adj(size2,Q,list_correspondance):

    vect_pairwise_adj = np.ones((size2, size2)) * (-1)
    nb_pairwise_adj = np.zeros((size2))

    for x1 in range(size2):

        cpt = 0
        v1 = list_correspondance[x1]

        i1 = v1//size
        j1 = v1%size

        for x2 in range(size2):

            v2 = list_correspondance[x2]
            i2 = v2 // size
            j2 = v2 % size

            if (x1 != x2 and (Q[i1, i2] != 0 or  Q[i1, j2] != 0 or  Q[j1, i2] != 0 or Q[j1, j2] != 0)):

                nb_pairwise_adj[x1] += 1
                vect_pairwise_adj[x1, cpt] = x2
                cpt += 1

    max_nb_pairwise_adj = int(np.max(nb_pairwise_adj))
    vect_pairwise_adj = vect_pairwise_adj[:, :max_nb_pairwise_adj]

    return  nb_pairwise_adj, vect_pairwise_adj


@jit(nopython=True)
def get_list_correspondance(size,Q):

    size2 = 0

    list_correspondance = []

    for i in range(size):

        for j in range(i+1):

            if (Q[i, j] != 0):

                size2 += 1

                list_correspondance.append(size*i + j)

    return list_correspondance, size2


@jit(nopython=True)
def get_nb_adj(size,Q):

    vect_adj = np.ones((size, size))*(-1)
    nb_adj = np.zeros((size))
    
    for i in range(size):
        
        cpt = 0
        
        for j in range(size):
            
            if(i!=j and Q[i,j] != 0):
                
                nb_adj[i] += 1
                
                vect_adj[i,cpt] = j
                
                cpt += 1
            
    max_nb_adj = int(np.max(nb_adj))
    vect_adj = vect_adj[:,:max_nb_adj]

    return vect_adj, nb_adj
                



if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('nameGraph', metavar='t', type=str, help='nameGraph')
    parser.add_argument('typeInstance', metavar='t', type=str, help='maxcut or ubqp')
    parser.add_argument('--id_gpu', metavar='t', type=int, help='id_gpu', default=0)
    parser.add_argument('--seed', metavar='t', type=int, help='seed', default=0)
    parser.add_argument('--budget', metavar='t', type=str, help='budget', default=999)
    parser.add_argument('--alpha_div', metavar='t', type=float, help='alpha_div', default=25)
    parser.add_argument('--gamma', metavar='t', type=float, help='gamma', default=20)
    parser.add_argument('--nb_neighbors', metavar='t', type=int, help='nb_neighbors', default=48)
    parser.add_argument('--factor_iter', metavar='t', type=int, help='factor_iter', default=2)
    parser.add_argument('--threadsperblock', metavar='t', type=int, help='threadsperblock', default=64)
    parser.add_argument('--nb_migrants', metavar='t', type=int, help='nb_migrants', default=10)
    parser.add_argument('--topology', metavar='t', type=str, help='topology', default='ring')
    parser.add_argument('--size_cross_factor', metavar='t', type=float, help='size_cross_factor',default=0.5)
    parser.add_argument('--beta', metavar='t', type=float, help='beta', default=1)
    parser.add_argument('--test', metavar='t', type=str, help='test', default='NoTest')
    parser.add_argument('--with_logs', metavar='t', type=bool, help='test', default=False)
    parser.add_argument('--score', metavar='t', type=int, help='score', default=-1)
    parser.add_argument('--typeCrossover', metavar='t', type=str, help='typeCrossover', default='XT')

       
    args = parser.parse_args()

    ######## Init gpu devices
    id_gpu = int(args.id_gpu)
    nb.cuda.select_device(id_gpu)
    device = "cuda:" + str(id_gpu)
    print(device)

    factor_iter = int(args.factor_iter)

    nb_migrants = int(args.nb_migrants)
    
    
    score = int(args.score)
    
    
    topology = args.topology
    
    
    typeCrossover = args.typeCrossover
    

    print("typeCrossover")
    print(typeCrossover)
    
    
    budget = float(args.budget)
    budget_time_total = budget*3600

    with_logs = args.with_logs
    print("with_logs : " + str(with_logs))
    
    ######## Load graph
    nameGraph = args.nameGraph
    typeInstance = args.typeInstance
    
    if(typeInstance == "maxcut"):
        filename = "benchmark/" + nameGraph
    elif(typeInstance == "ubqp"): 
        filename = "benchmark/" + nameGraph
        
        
    parallelInsert = False
    

                
    f = open(filename, "r")
    cpt = 0
    
    for line in f:
    
        line = line.rstrip("\n")
    
        x = line.split(sep=" ")
    
        if(cpt == 0):
            size = int(x[0])
            Q = np.zeros((size, size))
            cpt = 1
        else:
            if(typeInstance == "maxcut"):
                Q[int(x[0]),int(x[1]) ] = int(x[2])
                Q[int(x[1]),int(x[0]) ] = int(x[2])
            elif(typeInstance == "ubqp"):
                if(int(x[0]) != int(x[1])):
                    Q[int(x[0])-1,int(x[1])-1 ] = -2*int(x[2])
                    Q[int(x[1])-1,int(x[0])-1 ] = -2*int(x[2])
                else:
                    Q[int(x[0])-1,int(x[1])-1 ] = -int(x[2])
                
    beginTime = time()

    nonZeroQ = np.count_nonzero(Q)
    ratio = nonZeroQ/size

    vect_adj, nb_adj  = get_nb_adj(size,Q)
    vect_adj_global_mem = cuda.to_device(np.ascontiguousarray(vect_adj))
    nb_adj_global_mem = cuda.to_device(nb_adj)
    
    size2 = 0

    print(ratio)

    if(ratio < 8 ):
        
        list_correspondance, size2 = get_list_correspondance(size,Q)
        typeTabu = "doubleMove"
        
        if(typeCrossover != "PR" and typeCrossover != "UX" and typeCrossover != "MX" and typeCrossover != "PC"):
            typeCrossover = "XT11"
        vect_correspondance = np.array(list_correspondance)
        vect_correspondance_global_mem = cuda.to_device(np.ascontiguousarray(vect_correspondance))
        nb_pairwise_adj, vect_pairwise_adj = get_vect_pairwise_adj(size2,Q,list_correspondance)
        vect_pairwise_adj_global_mem = cuda.to_device(np.ascontiguousarray(vect_pairwise_adj))
        nb_pairwise_adj_global_mem = cuda.to_device(np.ascontiguousarray(nb_pairwise_adj))
    else:
        typeTabu = "singleMove"
        
        if(typeCrossover != "PR" and typeCrossover != "UX" and typeCrossover != "MX" and typeCrossover != "PC"):
            typeCrossover = "XT8"




    Q_global_mem = cuda.to_device(Q) # load adjacency matrix to device

    print("size : " + str(size))
    print("size 2 : " + str(size2))
    
    gamma = args.gamma

    ######## Parameters
    min_dist_insertion = size/gamma

    beta = float(args.beta)  
    

    nb_neighbors = int(args.nb_neighbors)

    nb_iter_tabu = int(size*factor_iter) # Number of local search iteration with TabuCol algorithm
    
    alpha_div = args.alpha_div
    
    if(typeTabu == "singleMove"):
        alpha = size/args.alpha_div
    else:
        alpha = size2/args.alpha_div
        
  
    print("alpha")
    print(alpha)

    test = args.test
    
    if(test == "test"):
        modeTest = True
    else:
        modeTest = False

    nb_iter_cross = int(size*args.size_cross_factor)

    size_pop = min(64000,int(32*10000000/size/1000) * 1000)
    
    print("size pop")
    print(size_pop)

    nb_clusters = int(size_pop/1000)

    print("nb_clusters")
    print(nb_clusters)

    if(modeTest):
        print("TEST")
        nb_iter_tabu = 10
        size_pop = 40
        nb_neighbors = 3
        nb_clusters = 4
        nb_migrants = nb_clusters
        nb_iter_cross = 10

    size_cluster = int(size_pop/nb_clusters)

    nb_neighbors = min(nb_neighbors,size_cluster)


    # Numba parameters
    threadsperblock = int(args.threadsperblock)

    best_score = 999999

    ######### Init logs
    date = datetime.datetime.now()

    name_expe = "Island_UBQP" + "_" + nameGraph + "_nb_clusters_" + str(nb_clusters) + "_topology_" + topology + "_size_pop_" + str(size_pop)   + "_typeTabu_" + typeTabu + "_typeCrossover_" + typeCrossover + "_gamma_" + str(gamma)  + "_nb_iter_" + str(nb_iter_tabu) + "_nb_iter_cross_" + str(nb_iter_cross)   + "_alpha_div_" + str(alpha_div) + "_nb_neighbors_" + str(nb_neighbors) + "_nb_migrants_" + str(nb_migrants) + "_beta_" + str(beta) + "_seed_" + str(args.seed) + "_" + str(date) + ".txt"
    if(with_logs):
        print("OKOKOKOK")
        logging.basicConfig(filename= "logs/" + name_expe + ".log",level=logging.INFO)
    #########

    ### Init tables ######

    offsprings_pop = np.zeros((size_pop, size), dtype=np.int32) # new colors generated after offspring
    fitness_pop = np.ones((size_pop), dtype=np.int32)*9999 # vector of fitness of the population
    fitness_offsprings = np.zeros((size_pop),dtype=np.int32) # vector of fitness of the offsprings

    vect_nb_components  = np.zeros((size_pop), dtype=np.int32)
    size_components = np.zeros((size_pop, size), dtype=np.int16)
    matrix_component_id = np.zeros((size_pop, size), dtype=np.int16)

    
    
    ####TEST
    #component_id = np.zeros((size_pop, size), dtype=np.int32) # new colors generated after offspring
    #component_id_gpu_memory = cuda.to_device(component_id)

    # Big Distance matrix with all individuals in pop and all offsprings at each generation
    matrixDistanceAll = np.zeros((nb_clusters, 2 * size_cluster, 2 * size_cluster), dtype=np.int16)

    for i in range(nb_clusters):
        matrixDistanceAll[i , :size_cluster, :size_cluster] = np.ones(( size_cluster, size_cluster), dtype=np.int16)*size

    matrixDistance1 = np.zeros((nb_clusters, size_cluster, size_cluster), dtype=np.int16) # Matrix with ditances between individuals in sub pop and offsprings for each sub pop
    matrixDistance2 = np.zeros((nb_clusters, size_cluster, size_cluster), dtype=np.int16) # Matrix with ditances between all offsprings for each sub pop
    
    matrixDistance1_gpu_memory = cuda.to_device(matrixDistance1)
    matrixDistance2_gpu_memory = cuda.to_device(matrixDistance2)


    matrice_crossovers_already_tested = np.zeros((nb_clusters, size_cluster, size_cluster), dtype=np.uint8)

    
    offsprings_pop_gpu_memory = cuda.to_device(offsprings_pop)
    fitness_pop_gpu_memory = cuda.to_device(fitness_pop)
    fitness_offsprings_gpu_memory = cuda.to_device(fitness_offsprings)
    
    vect_nb_components_gpu_memory  = cuda.to_device(vect_nb_components)
    matrix_component_id_gpu_memory  = cuda.to_device(matrix_component_id)
    size_components_gpu_memory  = cuda.to_device(size_components)

 
 
    # data for migration
    swap_pop_iter = 1
    start_migrations = 1

    size_sub_pop = size_cluster
    nb_pop = nb_clusters
    vector_already_sent = np.zeros((nb_pop, size_sub_pop), dtype=np.int16)

    if(topology == "fullyConnected"):
        nb_migrants = nb_clusters
    else:
        nb_migrants = int(args.nb_migrants)

    # Big Distance matrix with all individuals in pop and all offsprings at each generation
    matrixDistanceAll_migrants = np.zeros((nb_clusters, size_cluster + nb_migrants, size_cluster +  nb_migrants), dtype=np.int16)

    if(nb_migrants > 0):
        
        best_migrants = np.zeros((nb_pop, nb_migrants, size), dtype=np.int16)
        color_insertion = np.zeros((nb_pop * nb_migrants, size), dtype=np.int16)
        fitness_insertion = np.zeros((nb_pop, nb_migrants), dtype=np.int32)
        best_migrants_global_mem = cuda.to_device(best_migrants)
        best_migrants_fits = np.ones((nb_pop, nb_migrants)) * 9999

        matrix_migrants1 = np.zeros((nb_pop, size_sub_pop,  nb_migrants), dtype=np.int16)
        matrix_migrants1_global_mem = cuda.to_device(matrix_migrants1)

        matrix_migrants2 = np.zeros((nb_pop, nb_migrants,  nb_migrants), dtype=np.int16)
        matrix_migrants2_global_mem = cuda.to_device(matrix_migrants2)

    

    crossovers.crossovers_numba.size = size
    crossovers.crossovers_numba.size2 = size2
    distance_tools.distance_numba.size = size
    local_searches.tabuCol_numba.size = size
    local_searches.tabuCol_numba.size2 = size2
    utils.tools_numba.size = size
    utils.tools_numba.size_pop = size_pop
     
    #################################
    blockspergrid1 = (size_pop + (threadsperblock - 1)) // threadsperblock
    rng_states = create_xoroshiro128p_states(threadsperblock * blockspergrid1, seed=int(args.seed))  ## Init numba random generator
    np.random.seed(int(args.seed))

    time_start = time()

    ### Init pop ####
    utils.tools_numba.initPop_UBQP[blockspergrid1, threadsperblock](rng_states, size_pop, offsprings_pop_gpu_memory)

    colors_pop = offsprings_pop_gpu_memory.copy_to_host()


    #component_id = test_python_partition_crossover(size, colors_pop[0], colors_pop[1], nb_adj, vect_adj)

    #print("test component_id")
    #print(component_id)
    
    #verificationPython(component_id, vect_adj, nb_adj)
    
    for epoch in range(100000):



        ####### First step : local search  ################################################################################
        ########################################################################################################

        #### Start tabu
        if (with_logs):
            logging.info("############################")
            logging.info("Start Tabu")
            logging.info("############################")

        startEpoch = time()

        print("start tabu")
        start = time()

        if(typeTabu == "singleMove"):
            
            if(typeInstance == "ubqp"):
                local_searches.tabuCol_numba.tabuUBQP_oneFlip_v2[blockspergrid1, threadsperblock](rng_states,  size_pop, nb_iter_tabu, Q_global_mem,  offsprings_pop_gpu_memory,  fitness_offsprings_gpu_memory, alpha, vect_adj_global_mem, nb_adj_global_mem)
            else:
                local_searches.tabuCol_numba.tabuUBQP_oneFlip[blockspergrid1, threadsperblock](rng_states,  size_pop, nb_iter_tabu, Q_global_mem,  offsprings_pop_gpu_memory,  fitness_offsprings_gpu_memory, alpha, vect_adj_global_mem, nb_adj_global_mem)                
        else:
            local_searches.tabuCol_numba.tabuUBQP_swap[blockspergrid1, threadsperblock](rng_states,  size_pop, nb_iter_tabu, Q_global_mem,  offsprings_pop_gpu_memory,  fitness_offsprings_gpu_memory, alpha, vect_adj_global_mem, nb_adj_global_mem, vect_pairwise_adj_global_mem, nb_pairwise_adj_global_mem, vect_correspondance_global_mem)

        nb.cuda.synchronize()
        
        
        offsprings_pop_after_tabu = offsprings_pop_gpu_memory.copy_to_host()
        fitness_offsprings_after_tabu = fitness_offsprings_gpu_memory.copy_to_host()      
    
        print("offsprings_pop_after_tabu")
        print(offsprings_pop_after_tabu)

        if(with_logs):
            logging.info("fitness_offsprings_after_tabu")
            #logging.info(fitness_offsprings_after_tabu)
            logging.info("Tabucol duration : " + str(time() - start))

        print("end tabu")


        min_ubqp_score = np.min(fitness_offsprings_after_tabu)

        print("min_ubqp_score")
        print(min_ubqp_score )

        print("avg_ubqp_score")
        print(np.mean(fitness_offsprings_after_tabu))

        if (with_logs):
            logging.info("avg_ubqp_score")
            logging.info(np.mean(fitness_offsprings_after_tabu))


        if (min_ubqp_score < best_score):
            best_score = min_ubqp_score
            UBQP_solution = offsprings_pop_after_tabu[np.argmin(fitness_offsprings_after_tabu)]
            np.savetxt("legal_solutions/solution_ubqp_" + nameGraph + "_score_" + str(best_score) + ".csv", UBQP_solution)

        if (epoch % 1 == 0):

            fichier = open("evol/" + name_expe, "a")
            fichier.write("\n" + str(best_score ) + "," + str(
                    min_ubqp_score ) + "," + str(epoch) + ","  + str(time() - beginTime))
            fichier.close()

        if(time() - time_start >budget_time_total or (score != -1 and best_score == -score) ):
            break


        ####### Second step : insertion of offsprings in pop according to diversity/fit criterion ###############
        ########################################################################################################
        if (with_logs):
            logging.info("Keep best with diversity/fit tradeoff")
            logging.info("start matrix distance")

        print("start matrix distance")
        start = time()
        colors_pop_gpu_memory = cuda.to_device(colors_pop)

        blockspergrid_new = ((size_cluster * size_cluster)*nb_clusters + (threadsperblock - 1)) // threadsperblock

        distance_tools.distance_numba.computeMatrixDistance_Hamming_cluster[blockspergrid_new, threadsperblock](nb_clusters,
                                                                                                     size_cluster,
                                                                                                     size_cluster,
                                                                                                     matrixDistance1_gpu_memory,
                                                                                                     colors_pop_gpu_memory,
                                                                                                     offsprings_pop_gpu_memory)


        matrixDistance1 = matrixDistance1_gpu_memory.copy_to_host()


        blockspergrid2 = ((size_cluster * (size_cluster-1)//2*nb_clusters) + (threadsperblock - 1)) // threadsperblock
        distance_tools.distance_numba.computeSymmetricMatrixDistance_Hamming_cluster[blockspergrid2, threadsperblock](nb_clusters, size_cluster, matrixDistance2_gpu_memory, offsprings_pop_gpu_memory)

        matrixDistance2 = matrixDistance2_gpu_memory.copy_to_host()

        # Aggregate all the matrix in order to obtain a full 2*size_pop matrix with all the distances between individuals in pop and in offspring
        matrixDistanceAll[:, :size_cluster, size_cluster:] = matrixDistance1
        matrixDistanceAll[:, size_cluster:, :size_cluster] = matrixDistance1.transpose(0, 2, 1)
        matrixDistanceAll[:, size_cluster:, size_cluster:] = matrixDistance2

        if (with_logs):
            logging.info("Matrix distance duration : " + str(time() - start))

        print("end  matrix distance")
        #####################################


        print("start insertion in pop")
        start = time()

        if(parallelInsert):
            
            results = Parallel(n_jobs=nb_clusters)(delayed(insertion_pop)(size_cluster, 
                                                                        size_cluster,
                                                                    matrixDistanceAll[num_cluster], 
                                                                    colors_pop[num_cluster * size_cluster:(num_cluster + 1) * size_cluster],
                                                                    offsprings_pop_after_tabu[num_cluster * size_cluster:(num_cluster + 1) * size_cluster],
                                                                    fitness_pop[num_cluster * size_cluster:(num_cluster + 1) * size_cluster],
                                                                    fitness_offsprings_after_tabu[num_cluster * size_cluster:(num_cluster + 1) * size_cluster],
                                                                    matrice_crossovers_already_tested[num_cluster],
                                                                    vector_already_sent[num_cluster],
                                                                    min_dist_insertion) for num_cluster in range(nb_clusters))

            nb_insertion = 0
            
            for num_cluster in range(nb_clusters):
                matrixDistanceAll[num_cluster,:size_cluster,: size_cluster] = results[num_cluster][0]
                fitness_pop[num_cluster * size_cluster:(num_cluster + 1) * size_cluster] = results[num_cluster][1]
                colors_pop[num_cluster * size_cluster:(num_cluster + 1) * size_cluster] = results[num_cluster][2]
                matrice_crossovers_already_tested[num_cluster] = results[num_cluster][3]
                vector_already_sent[num_cluster] = results[num_cluster][4]
                nb_insertion += results[num_cluster][5]

        else:
                    
                nb_insertion = 0
                
                for num_cluster in range(nb_clusters):
                    
                    matrixDistanceAll[num_cluster,:size_cluster,: size_cluster], fitness_pop[num_cluster * size_cluster:(num_cluster + 1) * size_cluster], colors_pop[num_cluster * size_cluster:(num_cluster + 1) * size_cluster], matrice_crossovers_already_tested[num_cluster], vector_already_sent[num_cluster], nb_insertion_cluster = insertion_pop(size_cluster, 
                                                                        size_cluster,
                                                                    matrixDistanceAll[num_cluster], 
                                                                    colors_pop[num_cluster * size_cluster:(num_cluster + 1) * size_cluster],
                                                                    offsprings_pop_after_tabu[num_cluster * size_cluster:(num_cluster + 1) * size_cluster],
                                                                    fitness_pop[num_cluster * size_cluster:(num_cluster + 1) * size_cluster],
                                                                    fitness_offsprings_after_tabu[num_cluster * size_cluster:(num_cluster + 1) * size_cluster],
                                                                    matrice_crossovers_already_tested[num_cluster],
                                                                    vector_already_sent[num_cluster],
                                                                    min_dist_insertion)
                    
                    nb_insertion += nb_insertion_cluster

        if (with_logs):
            logging.info("Nb insertion pop : " + str(nb_insertion))
            logging.info("Insertion in pop : " + str(time() - start))

        print("end insertion in pop")

        if (with_logs):

            logging.info("After keep best info")

            for num_cluster in range(nb_clusters):

                best_score_pop = np.min(fitness_pop[num_cluster * size_cluster:(num_cluster + 1) * size_cluster])
                worst_score_pop = np.max(fitness_pop[num_cluster * size_cluster:(num_cluster + 1) * size_cluster])
                avg_score_pop = np.mean(fitness_pop[num_cluster * size_cluster:(num_cluster + 1) * size_cluster])

                #logging.info("Pop " + str(num_cluster) + "_best : " + str(best_score_pop) + "_worst : " + str(worst_score_pop) + "_avg : " + str(avg_score_pop))
                #logging.info(fitness_pop[num_cluster * size_cluster:(num_cluster + 1) * size_cluster])


                matrix_distance_pop = matrixDistanceAll[num_cluster,:size_cluster, :size_cluster]

                max_dist = np.max(matrix_distance_pop)
                min_dist = np.min(matrix_distance_pop + np.eye(size_cluster) * 9999)
                avg_dist = np.sum(matrix_distance_pop) / (size_cluster * (size_cluster - 1))

                logging.info("Avg dist : " + str(avg_dist) + " min dist : " + str(min_dist) + " max dist : " + str(max_dist))




    

        ####### Third step : migration  #########################
        ########################################################################################################

        if (epoch > start_migrations and epoch % swap_pop_iter == 0 and nb_migrants > 0):

            if (with_logs):
                logging.info("############################")
                logging.info("migration")
                logging.info("############################")

            print("start migration")
            start = time()


            if(topology == "ring"):

                for num_pop in range(nb_pop):
                
                    all_potential_migrants_fits = np.where(vector_already_sent[num_pop] < 1, fitness_pop[num_pop * size_sub_pop:(num_pop + 1) * size_sub_pop], 9999)
                    idx_best_migrants = np.argsort(all_potential_migrants_fits)[:nb_migrants]
                    all_individuals_pop = colors_pop[num_pop * size_sub_pop:(num_pop + 1) * size_sub_pop]
                    best_migrants[num_pop,:,:] = all_individuals_pop[idx_best_migrants,:]
                    all_individuals_fit = fitness_pop[num_pop * size_sub_pop:(num_pop + 1) * size_sub_pop]
                    best_migrants_fits[num_pop,:] = all_individuals_fit[idx_best_migrants]

                    if (with_logs):
                        logging.info("fit best_migrants")
                        logging.info(best_migrants_fits[num_pop,:])

                    vector_already_sent[num_pop,idx_best_migrants] = 1


                for num_pop in range(nb_pop):
                
                    fitness_insertion[num_pop, :nb_migrants] = best_migrants_fits[(num_pop - 1) % nb_pop]
                    color_insertion[(num_pop  * nb_migrants):(num_pop  * nb_migrants  + nb_migrants)] = best_migrants[(num_pop - 1) % nb_pop]
                

            elif(topology == "fullyConnected"):

                for num_pop in range(nb_pop):
                
                    all_potential_migrants_fits = np.where(vector_already_sent[num_pop] < 1, fitness_pop[num_pop * size_sub_pop:(num_pop + 1) * size_sub_pop], 9999)
                    idx_best_migrants = np.argsort(all_potential_migrants_fits)[:1]
                    all_individuals_pop = colors_pop[num_pop * size_sub_pop:(num_pop + 1) * size_sub_pop]

                    if(num_pop == 0):
                        best_migrants = all_individuals_pop[idx_best_migrants,:]
                    else:
                        best_migrants = np.concatenate([best_migrants, all_individuals_pop[idx_best_migrants,:]], axis = 0)

                    all_individuals_fit = fitness_pop[num_pop * size_sub_pop:(num_pop + 1) * size_sub_pop]

                    if(num_pop == 0):
                        best_migrants_fits = all_individuals_fit[idx_best_migrants]
                    else:
                        best_migrants_fits = np.concatenate([best_migrants_fits, all_individuals_fit[idx_best_migrants]], axis = 0)


                    vector_already_sent[num_pop,idx_best_migrants] = 1


                for num_pop in range(nb_pop):
                
                    fitness_insertion[num_pop, :nb_migrants] = best_migrants_fits
                    color_insertion[(num_pop  * nb_migrants):(num_pop  * nb_migrants  + nb_migrants)] = best_migrants
                


            for num_cluster in range(nb_clusters):
                matrixDistanceAll_migrants[num_cluster , :size_cluster, :size_cluster] = matrixDistanceAll[num_cluster, :size_cluster, :size_cluster]

            colors_pop_gpu_memory = cuda.to_device(colors_pop)
            color_insertion_global_mem = cuda.to_device(color_insertion)

            blockspergrid_new1 = ((size_cluster * nb_migrants)*nb_clusters + (threadsperblock - 1)) // threadsperblock

            distance_tools.distance_numba.computeMatrixDistance_Hamming_cluster[blockspergrid_new1, threadsperblock](nb_clusters,
                                                                                                        size_cluster,
                                                                                                        nb_migrants,
                                                                                                        matrix_migrants1_global_mem,
                                                                                                        colors_pop_gpu_memory,
                                                                                                        color_insertion_global_mem)

            matrix_migrants1 = matrix_migrants1_global_mem.copy_to_host()

            matrixDistanceAll_migrants[:, :size_cluster, size_cluster:] = matrix_migrants1
            matrixDistanceAll_migrants[:, size_cluster:, :size_cluster] = matrix_migrants1.transpose(0, 2, 1)

            if(nb_migrants > 1):

                blockspergrid_new2 = int(((nb_migrants * (nb_migrants-1)//2*nb_clusters) + (threadsperblock - 1)) // threadsperblock)
                distance_tools.distance_numba.computeSymmetricMatrixDistance_Hamming_cluster[blockspergrid_new2, threadsperblock](nb_clusters, nb_migrants, matrix_migrants2_global_mem, color_insertion_global_mem)
                matrix_migrants2 = matrix_migrants2_global_mem.copy_to_host()
                matrixDistanceAll_migrants[:, size_cluster:, size_cluster:] = matrix_migrants2



            print("start insertion in pop")
            start = time()

            
            if(parallelInsert):
                results = Parallel(n_jobs=nb_clusters)(delayed(insertion_pop)(size_cluster, 
                                                                            nb_migrants,                              
                                                                            matrixDistanceAll_migrants[num_cluster], 
                                                                            colors_pop[num_cluster * size_cluster:(num_cluster + 1) * size_cluster],                                 
                                                                            color_insertion[num_cluster * nb_migrants:(num_cluster + 1)  * nb_migrants],
                                                                            fitness_pop[num_cluster * size_cluster:(num_cluster + 1) * size_cluster],
                                                                            fitness_insertion[num_cluster],                                      
                                                                            matrice_crossovers_already_tested[num_cluster],
                                                                            vector_already_sent[num_cluster],
                                                                            min_dist_insertion) for num_cluster in range(nb_clusters))

                

                nb_insertion = 0
                    
                for num_cluster in range(nb_clusters):

                    matrixDistanceAll[num_cluster,:size_cluster,: size_cluster] = results[num_cluster][0]
                    fitness_pop[num_cluster * size_cluster:(num_cluster + 1) * size_cluster] = results[num_cluster][1]
                    colors_pop[num_cluster * size_cluster:(num_cluster + 1) * size_cluster] = results[num_cluster][2]
                    matrice_crossovers_already_tested[num_cluster] = results[num_cluster][3]
                    vector_already_sent[num_cluster] = results[num_cluster][4]          
                    nb_insertion += results[num_cluster][5]
            
            else:
                
                nb_insertion = 0
                
                for num_cluster in range(nb_clusters):
                    
       
                   results = insertion_pop(size_cluster,  nb_migrants,                              
                                                                            matrixDistanceAll_migrants[num_cluster], 
                                                                            colors_pop[num_cluster * size_cluster:(num_cluster + 1) * size_cluster],                                 
                                                                            color_insertion[num_cluster * nb_migrants:(num_cluster + 1)  * nb_migrants],
                                                                            fitness_pop[num_cluster * size_cluster:(num_cluster + 1) * size_cluster],
                                                                            fitness_insertion[num_cluster],                                      
                                                                            matrice_crossovers_already_tested[num_cluster],
                                                                            vector_already_sent[num_cluster],
                                                                            min_dist_insertion)
                   
                   matrixDistanceAll[num_cluster,:size_cluster,: size_cluster] = results[0]
                   fitness_pop[num_cluster * size_cluster:(num_cluster + 1) * size_cluster] = results[1]
                   colors_pop[num_cluster * size_cluster:(num_cluster + 1) * size_cluster] = results[2]
                   matrice_crossovers_already_tested[num_cluster] = results[3]
                   vector_already_sent[num_cluster] = results[4]          
                   nb_insertion += results[5]

            if (with_logs):
                logging.info("Nb migrants insertion pop : " + str(nb_insertion))
                logging.info("migrants Insertion in pop : " + str(time() - start))

            print("end migrants insertion in pop")

            if (with_logs):
                logging.info("After migration info")

                for num_cluster in range(nb_clusters):

                    best_score_pop = np.min(fitness_pop[num_cluster * size_cluster:(num_cluster + 1) * size_cluster])
                    worst_score_pop = np.max(fitness_pop[num_cluster * size_cluster:(num_cluster + 1) * size_cluster])
                    avg_score_pop = np.mean(fitness_pop[num_cluster * size_cluster:(num_cluster + 1) * size_cluster])

                    #logging.info("Pop " + str(num_cluster) + "_best : " + str(best_score_pop) + "_worst : " + str(worst_score_pop) + "_avg : " + str(avg_score_pop))
                    #logging.info(fitness_pop[num_cluster * size_cluster:(num_cluster + 1) * size_cluster])


                    matrix_distance_pop = matrixDistanceAll[num_cluster,:size_cluster, :size_cluster]

                    max_dist = np.max(matrix_distance_pop)
                    min_dist = np.min(matrix_distance_pop + np.eye(size_cluster) * 9999)
                    avg_dist = np.sum(matrix_distance_pop) / (size_cluster * (size_cluster - 1))

                    logging.info("Avg dist : " + str(avg_dist) + " min dist : " + str(min_dist) + " max dist : " + str(max_dist))

                logging.info("Time migration : " + str(time() - start))
            
            
            

        
  
            
            
        ####### Third step : selection of best crossovers to generate new offsprings  #########################
        ########################################################################################################
        if (with_logs):
            logging.info("############################")
            logging.info("start crossover")
            logging.info("############################")

        print("start crossover")

        start = time()
        

        bestColor_global_mem = cuda.to_device(colors_pop)


        all_expected_fit = []
        all_crossover = []

        pbar = range(size_pop)


        dist_neighbors = np.where(matrice_crossovers_already_tested == 1, 99999,
                matrixDistanceAll[:,:size_cluster, :size_cluster])

        dist_neighbors = np.where(dist_neighbors == 0, 99999, dist_neighbors)

        select_idx = np.random.randint(nb_neighbors, size=(nb_clusters, size_cluster,1))

        closest_individuals = np.argsort(dist_neighbors,axis =2)[:,:,:nb_neighbors]


        rng_closest_individuals = np.take_along_axis(closest_individuals, select_idx, 2)
        

        best_expected_fit = np.ones((size_pop))*99999
        best_solutions = np.zeros((size_pop, size))

        rng_closest_individuals_gpu_memory = cuda.to_device(np.ascontiguousarray(rng_closest_individuals))

        if(typeCrossover == "PR"):
            
            gamma1 = 0.3
            gamma2 = 0.7
              
            crossovers.crossovers_numba.compute_cluster_pathRelinking[blockspergrid1, threadsperblock](rng_states, size_pop, size_cluster, bestColor_global_mem, offsprings_pop_gpu_memory , rng_closest_individuals_gpu_memory, gamma1, gamma2,  Q_global_mem, vect_adj_global_mem, nb_adj_global_mem)

        elif (typeCrossover == "UX"):

            crossovers.crossovers_numba.compute_nearest_neighbor_crossovers_UX_cluster[blockspergrid1, threadsperblock](
                rng_states, size_pop, size_cluster, bestColor_global_mem, offsprings_pop_gpu_memory,
                rng_closest_individuals_gpu_memory)


        elif (typeCrossover == "MX"):

            crossovers.crossovers_numba.compute_nearest_neighbor_crossovers_MX_cluster[blockspergrid1, threadsperblock](
                rng_states, size_pop, size_cluster, bestColor_global_mem, offsprings_pop_gpu_memory,
                rng_closest_individuals_gpu_memory)
            
            
        elif (typeCrossover == "XT7"):

            crossovers.crossovers_numba.compute_nearest_neighbor_crossovers_XT7_cluster[blockspergrid1, threadsperblock](rng_states, size_pop, size_cluster,  nb_iter_cross, beta,  Q_global_mem, bestColor_global_mem, offsprings_pop_gpu_memory , rng_closest_individuals_gpu_memory, int(args.alpha_div) , vect_adj_global_mem, nb_adj_global_mem, vect_pairwise_adj_global_mem, nb_pairwise_adj_global_mem, vect_correspondance_global_mem)


           
        elif(typeCrossover == "XT8"):
            

            if(typeInstance == "ubqp"):
                crossovers.crossovers_numba.compute_nearest_neighbor_crossovers_XT8_cluster_v2[blockspergrid1, threadsperblock](rng_states, size_pop, size_cluster,  nb_iter_cross, beta,  Q_global_mem, bestColor_global_mem, offsprings_pop_gpu_memory , rng_closest_individuals_gpu_memory, int(args.alpha_div) , vect_adj_global_mem, nb_adj_global_mem)
            else:
                crossovers.crossovers_numba.compute_nearest_neighbor_crossovers_XT8_cluster[blockspergrid1, threadsperblock](rng_states, size_pop, size_cluster,  nb_iter_cross, beta,  Q_global_mem, bestColor_global_mem, offsprings_pop_gpu_memory , rng_closest_individuals_gpu_memory, int(args.alpha_div) , vect_adj_global_mem, nb_adj_global_mem)                
                
        elif (typeCrossover == "XT11"):

            crossovers.crossovers_numba.compute_nearest_neighbor_crossovers_XT11_cluster[blockspergrid1, threadsperblock](rng_states, size_pop, size_cluster,  nb_iter_cross, beta,  Q_global_mem, bestColor_global_mem, offsprings_pop_gpu_memory , rng_closest_individuals_gpu_memory, int(args.alpha_div) , vect_adj_global_mem, nb_adj_global_mem, vect_pairwise_adj_global_mem, nb_pairwise_adj_global_mem, vect_correspondance_global_mem)
            
        elif(typeCrossover == "PC"):
            
            
            
            
            crossovers.crossovers_numba.compute_nearest_neighbor_partition_crossover[blockspergrid1, threadsperblock](rng_states, size_pop, size_cluster, bestColor_global_mem, offsprings_pop_gpu_memory, rng_closest_individuals_gpu_memory,  vect_adj_global_mem, nb_adj_global_mem, Q_global_mem, vect_nb_components_gpu_memory, matrix_component_id_gpu_memory, size_components_gpu_memory)
            
            vect_nb_components = vect_nb_components_gpu_memory.copy_to_host()
            size_components = size_components_gpu_memory.copy_to_host()
            
            
            max_components = np.max(vect_nb_components)
            min_nb_components = np.min(vect_nb_components)
            avg_nb_components = np.mean(vect_nb_components)
            
            logging.info("Partition crossover")
            logging.info("avg_nb_components : " + str(avg_nb_components) + " min : " + str(min_nb_components) + " max : " + str(max_components))
            

            print("Partition crossover")
            print("avg_nb_components : " + str(avg_nb_components) + " min : " + str(min_nb_components) + " max : " + str(max_components))
            

            max_size = np.max(size_components[size_components>0])
            min_size = np.min(size_components[size_components>0])
            avg_size = np.mean(size_components[size_components>0])
            
            logging.info("avg_size_components : " + str(avg_size) + " min : " + str(min_size) + " max : " + str(max_size))
            
            print("avg_size_components : " + str(avg_size) + " min : " + str(min_size) + " max : " + str(max_size))


            #matrix_component_id = matrix_component_id_gpu_memory.copy_to_host()
            
            #offsprings_pop = offsprings_pop_gpu_memory.copy_to_host()
            
            
            #print("verification crossover")
            
            #pbar = tqdm(range(size_pop))
            
            #for i in pbar:
                
                #print("verification components")
                #verificationPython(matrix_component_id[i], vect_adj, nb_adj)
                
                
                #idx1 = i
                
                #idx_in_pop = idx1%size_cluster
                #num_pop = idx1//size_sub_pop
                #idx2 = int(num_pop * size_cluster + rng_closest_individuals[num_pop,idx_in_pop,0])
        
                
                #print("verification child")
                #verification_generation_child(matrix_component_id[i], offsprings_pop[i], colors_pop[idx1], colors_pop[idx2], vect_adj, nb_adj, size, Q)
                
                
                #distanceParent1 = matrixDistance1[i,i]
                
 
 
        blockspergrid_new = ((size_cluster * size_cluster)*nb_clusters + (threadsperblock - 1)) // threadsperblock
            
        distance_tools.distance_numba.computeMatrixDistance_Hamming_cluster[blockspergrid_new, threadsperblock](nb_clusters,
                                                                                                     size_cluster,
                                                                                                     size_cluster,
                                                                                                     matrixDistance1_gpu_memory,
                                                                                                     offsprings_pop_gpu_memory,
                                                                                                     bestColor_global_mem)
            
        matrixDistance1 = matrixDistance1_gpu_memory.copy_to_host()
        
        
        distance_child = np.zeros((size_pop))
        for i in range(size_pop):    

            idx1 = i
            idx_in_pop = idx1%size_cluster
            num_pop = idx1//size_sub_pop
            idx2 = int(num_pop * size_cluster + rng_closest_individuals[num_pop,idx_in_pop,0])
            
            distanceParent1 =  matrixDistance1[num_pop,idx_in_pop, idx_in_pop]
            distanceParent2 =  matrixDistance1[num_pop,idx_in_pop, int( rng_closest_individuals[num_pop,idx_in_pop,0])] 

            minDist = min(distanceParent1,distanceParent2)
            
            distance_child[i] = minDist
           
           
        max_dist = np.max(distance_child)
        min_dist = np.min(distance_child)
        avg_dist = np.mean(distance_child)
        
        logging.info("Distance child")
        logging.info("avg_dist_child : " + str(avg_dist) + " min : " + str(min_dist) + " max : " + str(max_dist))
        

        print("Distance child")
        print("avg_dist_child : " + str(avg_dist) + " min : " + str(min_dist) + " max : " + str(max_dist))           
            
            

        nb.cuda.synchronize()


        for i in range(size_pop):
            idx_in_pop = i%size_cluster
            num_pop = i//size_cluster
            matrice_crossovers_already_tested[num_pop,idx_in_pop,select_idx[num_pop,idx_in_pop,0]] = 1

        if (with_logs):
            logging.info("nb cross already tested in pop : " + str(np.sum(matrice_crossovers_already_tested)))
            logging.info("Crossover duration : " + str(time() - start))

        print("end crossover")

        if (with_logs):
            logging.info("generation duration : " + str(time() - startEpoch ))
        
        print("generation duration : " + str(time() - startEpoch ))          
            
