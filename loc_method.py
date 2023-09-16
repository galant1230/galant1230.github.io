# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 01:19:51 2022

@author: ren tsai
"""
import numpy as np
from scipy import optimize
import sys, collections, time
from scipy.optimize import lsq_linear, root, minimize
import random
# import matplotlib.pyplot as plt
import numpy.matlib 
from itertools import product
from itertools import combinations
from collections import Counter
import numpy as np
# import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import heapq
import random 
from sympy import *
import cmath
import math


def lsq_method(distances_to_anchors, anchor_positions, u):
    # distances_to_anchors, anchor_positions = np.array(distances_to_anchors), np.array(anchor_positions)
    # if not np.all(distances_to_anchors):
    #     raise ValueError('Bad uwb connection. distances_to_anchors must never be zero. ' + str(distances_to_anchors))
    anchor_offset = anchor_positions[0]
    anchor_positions = anchor_positions[1:] - anchor_offset
    K = np.sum(np.square(anchor_positions), axis=1)   #ax=1 列加
    squared_distances_to_anchors = np.square(distances_to_anchors)
    squared_distances_to_anchors = (squared_distances_to_anchors - squared_distances_to_anchors[0])[1:]
    b = (K - squared_distances_to_anchors) / 2.
    det = u.T @ b
    #res = lsq_linear(anchor_positions, b, lsmr_tol='auto', verbose=0)
    #res = np.dot(np.dot(np.linalg.inv(np.dot(anchor_positions.T, anchor_positions)),(anchor_positions.T)), b)
    res = np.linalg.lstsq(anchor_positions, b, rcond=None)[0]
    return res + anchor_offset, det, b

from sympy import symbols, expand
from sympy.parsing.sympy_parser import parse_expr
from collections import OrderedDict

def extract_coefficients(expression):
    z = symbols('z')
    expanded_expr = expand((expression))
    # print('expanded_expr', expanded_expr)
    coefficients = Poly(expanded_expr, z).all_coeffs()
    # print('coefficients', coefficients)

    return coefficients

def Cardano(a,b,c,d):
    
    complex_num = (-1+cmath.sqrt(3)*1j)/2
    complex_num_2 = complex_num**2
    z0=b/3/a
    a2,b2 = a*a,b*b
    p=-b2/3/a2 +c/a
    q=(b/27*(2*b2/a2-9*c/a)+d)/a
    D=-4*p*p*p-27*q*q
    r=cmath.sqrt(-D/27+0j)
    u=((-q-r)/2)**0.33333333333333333333333
    v=((-q+r)/2)**0.33333333333333333333333

    
    z_candidate = [u+v-z0, u*complex_num + v *complex_num_2-z0, u*complex_num_2 + v*complex_num-z0]
    return z_candidate


def cardano_formula(a, b, c, d):
    # 计算中间变量
    p = (3*a*c - b**2) / (3*a**2)
    q = (2*b**3 - 9*a*b*c + 27*a**2*d) / (27*a**3)

    # 计算判别式和虚部修正项
    delta = q**2/4 + p**3/27
    delta_sqrt = cmath.sqrt(delta)

    # 计算两个复数解的基础值
    u1 = -q/2 + delta_sqrt
    u2 = -q/2 - delta_sqrt

    # 计算两个复数解
    x1 = u1**(1/3)
    x2 = u2**(1/3)

    # 计算第一个实数解
    y1 = x1 + x2 - b / (3*a)

    # 计算剩余两个复数解
    j = cmath.exp(2j*np.pi/3)
    y2 = j*x1 + j**2*x2 - b / (3*a)
    y3 = j**2*x1 + j*x2 - b / (3*a)

    return y1, y2, y3


def two_stage(distances_to_anchors, anchor_positions, u):
    tag_pos, det, b = lsq_method(distances_to_anchors, anchor_positions, u)
    
    z = symbols('z') #, real = True
    f = symbols('f', cls = Function)
    f = 0
    sum_delta, b_z, c_z, d_z = 0, 0, 0, 0
    for i in range(anchor_positions.shape[0]):
        delta = distances_to_anchors[i]**2 - ((tag_pos[0]- anchor_positions[i][0])**2 + (tag_pos[1]- anchor_positions[i][1])**2)
        f += 4 * ((z - anchor_positions[i][2]) ** 3 - delta*((z)-anchor_positions[i][2]))
    coeff = extract_coefficients(f)
    
    z_candidate = solve(f,z)
    # z_candidate = Cardano(coeff[0], coeff[1], coeff[2], coeff[3])
    # z_candidate = cardano_formula(coeff[0], coeff[1], coeff[2], coeff[3])
    # print('This is z candidate cardano ', z_candidate)
    


    z_candidate = np.array([complex(item) for item in z_candidate])
    # print('This is z candidate cardano', z_candidate)
    z_candidate = np.round(np.array([abs(z_candidate[0]), abs(z_candidate[1]), abs(z_candidate[2])]),5)


    result = list()
    check_ls = list()
    
        
    for i in range(z_candidate.shape[0]):
        check = abs(distances_to_anchors[0]**2 - (tag_pos[0] - anchor_positions[0][0])**2 - (tag_pos[1] - anchor_positions[0][1]) **2 - (z_candidate[i] - anchor_positions[0][2])**2)
        check_ls.append(check)
    index = check_ls.index(min(check_ls))
    # print('index', index)
    
    two_ans = np.array([tag_pos[0], tag_pos[1], z_candidate[index]])
    # print('two_ans', two_ans)
    result.append(two_ans)
    
    return np.array(result).astype(np.float32), det, b

def costfun_method(distances_to_anchors, anchor_positions, u):
    distances_to_anchors, anchor_positions = np.array(distances_to_anchors), np.array(anchor_positions)
    # tag_pos = Least_square(distances_to_anchors, anchor_positions)
    tag_pos, det, b= lsq_method(distances_to_anchors, anchor_positions, u)
    anc_z_ls_mean = np.mean(np.array([i[2] for i in anchor_positions]) )  
    new_z = (np.array([i[2] for i in anchor_positions]) - anc_z_ls_mean).reshape(4, 1)
    new_anc_pos = np.concatenate((np.delete(anchor_positions, 2, axis = 1), new_z ), axis=1)
    new_disto_anc = np.sqrt(abs(distances_to_anchors[:]**2 - (tag_pos[0] - new_anc_pos[:,0])**2 - (tag_pos[1] - new_anc_pos[:,1])**2))
    new_z = new_z.reshape(4,)

    a = (np.sum(new_disto_anc[:]**2) - 3*np.sum(new_z[:]**2))/len(anchor_positions)
    b = (np.sum((new_disto_anc[:]**2) * (new_z[:])) - np.sum(new_z[:]**3))/len(anchor_positions)
    cost = lambda z: np.sum(((z - new_z[:])**4 - 2*(((new_disto_anc[:])*(z - new_z[:]))**2 ) + new_disto_anc[:]**4))/len(anchor_positions) 

    function = lambda z: z**3 - a*z + b
    derivative = lambda z: 3*z**2 - a

    ranges = (slice(-10, 10, 0.01), )
    resbrute = optimize.brute(cost, ranges, full_output = True, finish = optimize.fmin)
    # print('resbrute: ', resbrute[0][0] + anc_z_ls_mean)
    new_tag_pos = np.array([tag_pos[0], tag_pos[1], abs(resbrute[0][0]) + anc_z_ls_mean])

    # new_tag_pos = np.array([tag_pos[0], tag_pos[1], newton_z + anc_z_ls_mean])

    return np.around(new_tag_pos, 2)

def two_stage_solve(distances_to_anchors, anchor_positions, u):
    tag_pos, det, b = lsq_method(distances_to_anchors, anchor_positions, u)
    
    z = symbols('z') #, real = True
    f = symbols('f', cls = Function)
    f = 0
    sum_delta, b_z, c_z, d_z = 0, 0, 0, 0
    for i in range(anchor_positions.shape[0]):
        delta = distances_to_anchors[i]**2 - ((tag_pos[0]- anchor_positions[i][0])**2 + (tag_pos[1]- anchor_positions[i][1])**2)
        f += 4 * ((z - anchor_positions[i][2]) ** 3 - delta*((z)-anchor_positions[i][2]))
    coeff = extract_coefficients(f)
    
    # z_candidate = solve(f,z)
    z_candidate = solveset(f,z)
    # z_candidate = Cardano(coeff[0], coeff[1], coeff[2], coeff[3])


    z_candidate = np.array([complex(item) for item in z_candidate])
    print('This is z candidate solve', z_candidate)
    z_candidate = np.round(np.array([abs(z_candidate[0]), abs(z_candidate[1]), abs(z_candidate[2])]),5)
    # print('This is z candidate', z_candidate_max)  
    result = list()
    check_ls = list()
    
    for i in range(z_candidate.shape[0]):
        check = abs(distances_to_anchors[0]**2 - (tag_pos[0] - anchor_positions[0][0])**2 - (tag_pos[1] - anchor_positions[0][1]) **2 - (z_candidate[i] - anchor_positions[0][2])**2)
        check_ls.append(check)
    index = check_ls.index(min(check_ls))
    two_ans = np.array([tag_pos[0], tag_pos[1], abs(z_candidate[index])])
    two_ans = np.array([tag_pos[0], tag_pos[1], z_candidate[index]])
    result.append(two_ans)
    
    return np.array(result).astype(np.float32), det, b

