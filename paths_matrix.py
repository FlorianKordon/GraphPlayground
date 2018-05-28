# -*- coding: utf-8 -*-
"""
Created on Sun May 27 22:10:37 2018

@author: floko
"""
import numpy as np
import numpy.linalg as LA
import time

def paths_matrix(A_d, variant):
    if(variant == 1):
        """
        Variante 1): Erreichbarkeitsalgorithmus
        """
        W_d = np.clip(A_d,0,1)
        
        num_vert = A_d.shape[0]
        for k in range(num_vert):
            for i in range(num_vert):
                for j in range(num_vert):
                    if W_d[i,k] and W_d[k,j]:
                        W_d[i,j] = 1
                        
        return W_d
    
    elif(variant == 2):
        """
        Variante 2): Potenzen der Adjazenzmatrix  
        """
        num_vert = A_d.shape[0]
        A_pot = np.eye(num_vert)
        W_d = np.zeros_like(A_d, dtype = np.float64)
        for k in range(num_vert):
            A_pot = np.dot(A_pot, A_d)
            W_d += A_pot
            
        return np.clip(W_d,0,1)
    
    elif(variant == 3):
        """
        Variante 3): Transformation zur Eigenbasis, Potenzen der Adjazenzmatrix
        """        
        num_vert = A_d.shape[0]
        
        # Eigenvektoren und Eigenwerte + Sortierung
        eigen_vals, eigen_vecs = LA.eig(A_d)
        idx_order = np.argsort(eigen_vals)[::-1]
        eigen_vals = eigen_vals[idx_order]
        eigen_vecs = eigen_vecs[:,idx_order]        
        
        
        # Trans. zur Eigenbasis, Ausnutzen der Potenzierung von diagonalen Matrizen
        C = eigen_vecs
        
        if (LA.det(C)==0):
            raise Exception('The eigenbasis matrix is singular/degenerate and can not be inverted. Hence, the conversion to its eigenbasis is not possible')
        C_inv = LA.inv(eigen_vecs)
        D = np.diag(eigen_vals)     
        
        W_d = np.zeros_like(A_d, dtype = np.complex128)
        for k in range (1,num_vert+1):
            T = C @ D**k @ C_inv
            W_d += T
            
        return np.clip(np.real(W_d),0,1)
    
    else:
        raise ValueError('Variant number must be either 1,2 or 3')
            
##########
# Testlauf
        
# Adjazenzmatrix eines Graphen, zu dem die Wegematrix berechnet werden soll.
A_d = np.array([[0,0,1,0],
                [0,0,0,1],
                [0,1,0,0],
                [0,0,1,0]])
#A_d = np.where(np.random.uniform(10, size=(150, 150))>9.5,1,0)       
        
res ={}
for i in range(1,4):
    start_time = time.time()
    res[i] = paths_matrix(A_d,i)
    print("Variant " + str(i) + " took %s seconds" % (time.time() - start_time))















