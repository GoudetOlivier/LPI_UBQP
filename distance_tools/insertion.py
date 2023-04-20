
import numpy as np
from random import shuffle


def insertion_pop(size1, size2, matrixDistanceAll, color1,
                           color2, fitness1, fitness2, matrice_crossovers_already_tested, vector_already_sent, min_dist):


    all_scores = np.hstack((fitness1, fitness2))

    matrice_crossovers_already_tested_new= np.zeros((size1 + size2, size1 + size2), dtype = np.uint8)
    matrice_crossovers_already_tested_new[:size1,:size1] = matrice_crossovers_already_tested

    vector_already_sent_new = np.zeros((size1 + size2), dtype = np.uint8)
    vector_already_sent_new[:size1] = vector_already_sent
    
    
    idx_best = np.argsort(all_scores)

    idx_selected = []

    cpt = 0

    for i in range(0, size1 + size2):

        idx = idx_best[i]

        if(len(idx_selected) > 0):
            dist = np.min(matrixDistanceAll[idx,idx_selected])
        else: 
            dist = 99999
         
        if (dist >= min_dist):

            idx_selected.append(idx)

            if(idx >= size1):
                cpt+=1

        if(len(idx_selected) == size1):
            break;



    if(len(idx_selected) != size1):
        for i in range(0,size1  + size2):
            idx = idx_best[i]

            if(idx not in idx_selected):
                dist = np.min(matrixDistanceAll[idx,idx_selected])
                if (dist >= 0):
                    idx_selected.append(idx)

            if(len(idx_selected) == size1):
                break;



    new_matrix = matrixDistanceAll[idx_selected, :][:,idx_selected]


    stack_all = np.vstack((color1, color2))

    colors_pop_v2 = stack_all[idx_selected]
    fitness_pop_v2 = all_scores[idx_selected]

    matrice_crossovers_already_tested_v2 = matrice_crossovers_already_tested_new[idx_selected, :][:, idx_selected]
    vector_already_sent_v2 = vector_already_sent_new[idx_selected]

    return new_matrix, fitness_pop_v2,   colors_pop_v2, matrice_crossovers_already_tested_v2, vector_already_sent_v2, cpt









