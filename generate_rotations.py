#!/usr/bin/python3

#----------------------
# script to generate random angle rotations 
# uniformly distributed in a S2
#----------------------



import numpy as np


def normal_vector(u,theta):
    x = np.sqrt(1-u**2)*np.cos(theta)
    y = np.sqrt(1-u**2)*np.sin(theta)
    z = u
    
    return np.array((x,y,z))

def angles(x,y,z):
    beta = np.arcsin(-x)
    alpha = np.arcsin(y/np.cos(beta))
    
    if z<0:
        beta = beta+np.pi
    return alpha, beta


n = []

for i in range(0,1000):
    u = np.random.random()*2-1
    theta = np.random.random()*2*np.pi
    n.append(normal_vector(u,theta))  
    
n = np.array(n)

# generate the angles

RAs = [] 

for i in range(len(n)):
    x = n[i,0]
    y = n[i,1]
    z = n[i,2]
    alpha , beta = angles(x,y,z)
    gamma = np.random.random()*2*np.pi # random north 
    
    alpha = alpha*180/np.pi
    beta = beta*180/np.pi
    gamma = gamma*180/np.pi
    
    RAs.append([alpha,beta,gamma])

np.savetxt('1k_rotations.txt',RAs)


