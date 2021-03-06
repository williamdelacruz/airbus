from __future__ import print_function
import math
import numpy as np
import sys
import itertools
import random
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from random import randint
from random import gauss
import pickle
import sqaod as sq
import time
from dwave_qbsolv import QBSolv
import neal



# - - - - - - - - - - - - - - - - - - #
#     Inicio de las definiciones      #
# - - - - - - - - - - - - - - - - - - #


# Funcion de energia 'E_alloc' de la ecuacion (27)
#
def quboAllocation_validation(solution, mass_x, mass_y, mass_z, Npos, offset):
    # Cantidad de contenedores de los tres tipos
    n1 = len(mass_x)
    n2 = len(mass_y)
    n3 = len(mass_z)


    # Ealloc = Sum_j (2 Sum_i x_ij + Sum_i y_ij + 2 Sum_ik z_ijk
    #          + ep_1j + 2ep_2j - 2)**2
    #        = sum_j (2*A1 + A2 + 2*A3 + 2*ep1 - 2)**2

    t_cost1 = 0

    for j in range(Npos):
        # Indexes for vaiables x_ij, y_ij, z_ijk and ep_j{1,2}
        list_x = [ix(i,j,Npos) for i in range(n1)]
        list_y = [iy(i,j,n1,Npos) for i in range(n2)]
        list_z = [iz(i,j,k,n1,n2,n3,Npos) for i in range(n3) for k in range(2)]
        list_slack = [islack(i,j,n1,n2,n3,Npos,0, offset) for i in range(1)]

        len_x = n1
        len_y = n2
        len_z = 2*n3

        # -8 A1
        for x in list_x:
            #Q = updateKey(Q, (x, x), -8)
            t_cost1 = t_cost1 + -8*solution[x]

        # -4 A2
        for y in list_y:
            #Q = updateKey(Q, (y, y), -4)
            t_cost1 = t_cost1 + -4*solution[y]

        # -8 A3
        for z in list_z:
            #Q = updateKey(Q, (z, z), -8)
            t_cost1 = t_cost1 + -8*solution[z]


        # + 4 A1 A2
        for x in list_x:
            for y in list_y:
                #Q = updateKey(Q, ipair(x, y), 4)
                t_cost1 = t_cost1 + 4*solution[x]*solution[y]

        # + 8 A1 A3
        for x in list_x:
            for z in list_z:
                #Q = updateKey(Q, ipair(x, z), 8)
                t_cost1 = t_cost1 + 8*solution[x]*solution[z]

        # + 4 A2 A3
        for y in list_y:
            for z in list_z:
                #Q = updateKey(Q, ipair(y, z), 4)
                t_cost1 = t_cost1 + 4*solution[y]*solution[z]

        # + 4 A1**2
        for x in list_x:
            #Q = updateKey(Q, (x,x), 4)
            t_cost1 = t_cost1 + 4*solution[x]

        for x1 in range(len_x):
            for x2 in range(x1+1,len_x):
                #key = ipair(list_x[x1], list_x[x2])
                #Q = updateKey(Q, key, 8)
                t_cost1 = t_cost1 + 8*solution[list_x[x1]]*solution[list_x[x2]]

        # + A2**2
        for y in list_y:
            #Q = updateKey(Q, (y,y), 1)
            t_cost1 = t_cost1 + solution[y]

        for y1 in range(len_y):
            for y2 in range(y1+1,len_y):
                #key = ipair(list_y[y1], list_y[y2])
                #Q = updateKey(Q, key, 2)
                t_cost1 = t_cost1 + 2*solution[list_y[y1]]*solution[list_y[y2]]

        # + 4 A3**2
        for z in list_z:
            #Q = updateKey(Q, (z,z), 4)
            t_cost1 = t_cost1 + 4*solution[z]

        for z1 in range(len_z):
            for z2 in range(z1+1,len_z):
                #key = ipair(list_z[z1], list_z[z2])
                #Q = updateKey(Q, key, 8)
                t_cost1 = t_cost1 + 8*solution[list_z[z1]]*solution[list_z[z2]]

        # + 8 A1 ep1
        for x in list_x:
            #key = ipair(x, list_slack[0])
            #Q = updateKey(Q, key, 4)
            t_cost1 = t_cost1 + 8*solution[x]*solution[list_slack[0]]
            #key = ipair(x, list_slack[1])
            #Q = updateKey(Q, key, 8)
            #t_cost1 = t_cost1 + 8*solution[x]*solution[list_slack[1]]

        # + 4 A2 ep1
        for y in list_y:
            #key = ipair(y, list_slack[0])
            #Q = updateKey(Q, key, 2)
            t_cost1 = t_cost1 + 4*solution[y]*solution[list_slack[0]]
            #key = ipair(y, list_slack[1])
            #Q = updateKey(Q, key, 4)
            #t_cost1 = t_cost1 + 4*solution[y]*solution[list_slack[1]]


        # + 8 A3 ep1
        for z in list_z:
            #key = ipair(z, list_slack[0])
            #Q = updateKey(Q, key, 4)
            t_cost1 = t_cost1 + 8*solution[z]*solution[list_slack[0]]
            #key = ipair(z, list_slack[1])
            #Q = updateKey(Q, key, 8)
            #t_cost1 = t_cost1 + 8*solution[z]*solution[list_slack[1]]

        # - 4 ep1
        #key = (list_slack[0], list_slack[0])
        #Q = updateKey(Q, key, -3)
        t_cost1 = t_cost1 + -4*solution[list_slack[0]]
        #key = (list_slack[1], list_slack[1])
        #Q = updateKey(Q, key, -4)
        #t_cost1 = t_cost1 + -4*solution[list_slack[1]]
        #key = ipair(list_slack[0], list_slack[1])
        #Q = updateKey(Q, key, 4)
        #t_cost1 = t_cost1 + 4*solution[list_slack[0]]*solution[list_slack[1]]

        #constant += 4
        t_cost1 = t_cost1 + 4



    # Segundo termino ecuacion (27)
    # Sum_i Sum_{j1 < j2} x_ij_1 x_ij_2
    t_cost2 = 0

    for i in range(n1):
        for j1 in range(Npos):
            for j2 in range(j1+1, Npos):
                x1 = ix(i,j1,Npos)
                x2 = ix(i,j2,Npos)
                #key = ipair(x1,x2)
                #Q = updateKey(Q, key, 1)
                t_cost2 = t_cost2 + solution[x1]*solution[x2]

    # Tercer termino ecuacion (27)
    # Sum_i Sum_{j1 < j2} y_ij_1 y_ij_2
    t_cost3 = 0

    for i in range(n2):
        for j1 in range(Npos):
            for j2 in range(j1+1, Npos):
                y1 = iy(i, j1, n1, Npos)
                y2 = iy(i, j2, n1, Npos)
                #key = ipair(y1,y2)
                #Q = updateKey(Q, key, 1)
                t_cost3 = t_cost3 + solution[y1]*solution[y2]

    # Cuarto y quinto termino ecuacion (27)
    # Sum_i Sum_{j1 < j2} z_ij_11 z_ij_21  +
    # Sum_i Sum_{j1 < j2} z_ij_12 z_ij_22
    t_cost4 = 0
    t_cost5 = 0

    for i in range(n3):
        for j1 in range(Npos):
            for j2 in range(j1+1, Npos):
                z1 = iz(i, j1, 0, n1, n2, n3, Npos)
                z2 = iz(i, j2, 0, n1, n2, n3, Npos)
                #Q = updateKey(Q, ipair(z1, z2), 1)
                t_cost4 = t_cost4 + solution[z1]*solution[z2]

                z1 = iz(i, j1, 1, n1, n2, n3, Npos)
                z2 = iz(i, j2, 1, n1, n2, n3, Npos)
                #Q = updateKey(Q, ipair(z1, z2), 1)
                t_cost5 = t_cost5 + solution[z1]*solution[z2]


    # Sexto  termino ecuacion (27)
    # Sum_i Sum_{j=1}^{N-1} [ z_ij1 + z_i(j+1)2 - 2 z_ij1 z_i(j+1)2 ]
    t_cost6 = 0

    for i in range(n3):
        for j in range(Npos-1):
            z1 = iz(i, j, 0, n1, n2, n3, Npos)
            #Q = updateKey(Q, (z1,z1), 1)
            t_cost6 = t_cost6 + solution[z1]

            z2 = iz(i, j+1, 1, n1, n2, n3, Npos)
            #Q = updateKey(Q, (z2,z2), 1)
            t_cost6 = t_cost6 + solution[z2]

            #z12 = ipair(z1, z2)
            #Q = updateKey(Q, z12, -2)
            t_cost6 = t_cost6 + -2*solution[z1]*solution[z2]


    # Septimo termino ecuacion (27)
    # Sum_i (z_iN1 + z_i12)
    t_cost7 = 0

    for i in range(n3):
        z = iz(i, Npos-1, 0, n1, n2, n3, Npos)
        #Q = updateKey(Q, (z,z), 1)
        t_cost7 = t_cost7 + solution[z]

        z = iz(i, 0, 1, n1, n2, n3, Npos)
        #Q = updateKey(Q, (z,z), 1)
        t_cost7 = t_cost7 + solution[z]

    t_cost = [t_cost1, t_cost2, t_cost3, t_cost4, t_cost5, t_cost6, t_cost7]

    return t_cost




# Construccion de la restriccion de maxima carga Ec. 28
#
def quboPayload_validation(solution, mass_x, mass_y, mass_z, Npos, Wp, offset, Weight):
    # We simplify constraint (3) as (F - Wp + Sw)**2
    # whose expand is F**2 + 2*F*Sw + Sw**2 - 2*F*Wp - 2*Sw*Wp + Wp**2

    #Q = Q1

    # Cantidad de contenedores de los tres tipos
    n1 = len(mass_x)
    n2 = len(mass_y)
    n3 = len(mass_z)

    #constant = const;

    # Numero de variables slack para representar Wp
    nbits = math.ceil(math.log2(Wp))

    # Indexes for variables x_ij, y_ij, z_ij1
    list_x = [ix(i,j,Npos) for j in range(Npos) for i in range(n1)]
    list_y = [iy(i,j,n1,Npos) for j in range(Npos) for i in range(n2)]
    list_z = [iz(i,j,0,n1,n2,n3,Npos) for j in range(Npos) for i in range(n3)]


    # List of masses according to the enumeration above
    list_mass_x = []
    list_mass_y = []
    list_mass_z = []

    for k in range(Npos):
        list_mass_x = np.concatenate([list_mass_x, mass_x])
        list_mass_y = np.concatenate([list_mass_y, mass_y])
        list_mass_z = np.concatenate([list_mass_z, mass_z])



    # lista de variables y coeficientes para la expansion binaria de Wp

    list_slack = [islack(i,0,n1,n2,n3,Npos,1, offset) for i in range(nbits)]
    coeff_slack = [2**i for i in range(nbits)]


    #    Concatenate all masses in a one vector [list_mass_x , list_mass_y , list_mass_z]
    list_var = np.concatenate([list_x, list_y, list_z])
    list_mass = np.concatenate([list_mass_x, list_mass_y, list_mass_z])

    t_cost = 0

    # Termino F**2
    len_list = len(list_var)

    for i in range(0,len_list):
        for j in range(i+1,len_list):
            #key = ipair(list_var[i], list_var[j])
            #Q = updateKey(Q, key, 2*list_mass[i]*list_mass[j])
            t_cost = t_cost + 2*list_mass[i]*list_mass[j]*solution[list_var[i]]*solution[list_var[j]]*Weight

    for i in range(len_list):
        #i1 = list_var[i]
        #Q = updateKey(Q, (i1,i1), list_mass[i]**2)
        t_cost = t_cost + Weight*solution[list_var[i]]*list_mass[i]**2


    # Termino 2*F*Sw
    for i in range(len_list):
        for j in range(nbits):
            #key = ipair(list_var[i], list_slack[j])
            #Q = updateKey(Q, key, 2*list_mass[i]*coeff_slack[j])
            t_cost = t_cost + 2*list_mass[i]*coeff_slack[j]*solution[list_var[i]]*solution[list_slack[j]]*Weight


    # Termino Sw**2
    for i in range(nbits):
        for j in range(i+1,nbits):
            #key = ipair(list_slack[i], list_slack[j])
            #Q = updateKey(Q, key, 2*coeff_slack[i]*coeff_slack[j])
            t_cost = t_cost + 2*coeff_slack[i]*coeff_slack[j]*solution[list_slack[i]]*solution[list_slack[j]]*Weight


    for i in range(nbits):
        #i1 = list_slack[i]
        #Q = updateKey(Q, (i1,i1), coeff_slack[i]**2)
        t_cost = t_cost + Weight*solution[list_slack[i]]*coeff_slack[i]**2


    # Termino -2*Wp*F
    for i in range(len_list):
        #k = list_var[i]
        #Q = updateKey(Q, (k,k), -2*list_mass[i]*Wp)
        t_cost = t_cost + -2*list_mass[i]*Wp*solution[list_var[i]]*Weight


    # Termino -2*Wp*Sw
    for i in range(nbits):
        #k = list_slack[i]
        #Q = updateKey(Q, (k,k), -2*coeff_slack[i]*Wp)
        t_cost = t_cost + -2*coeff_slack[i]*Wp*solution[list_slack[i]]*Weight


    # El termino constante se conserva  Wp**2 en la variable 'constant'
    #constant += Wp**2
    #offset1 = offset + nbits

    t_cost = t_cost + Weight*Wp**2

    return t_cost






#Crea el grafico de barras solo con el objetivo 1
def PlotBarContainers(solution, Npos, mass_x, mass_y, mass_z):
    n1 = len(mass_x)
    n2 = len(mass_y)
    n3 = len(mass_z)

    pos = [i for i in range(Npos)]
    masa = [0 for i in range(Npos)]
    masa_top = [0 for i in range(Npos)]
    masa_c = ['red' for i in range(Npos)]

    for j in range(Npos):
        for i in range(n1):
            key = ix(i,j,Npos)
            masa[j] += solution[key]*mass_x[i]

        for i in range(n2):
            key = iy(i, j, n1, Npos)
            if masa[j] == 0:
                masa[j] += solution[key]*mass_y[i]
            else:
                masa_top[j] += solution[key]*mass_y[i]

            if solution[key] == 1:
                masa_c[j] = 'yellow'

        for i in range(n3):
            key0 = iz(i, j, 0, n1, n2, n3, Npos)
            masa[j] += solution[key0]*mass_z[i]
            key1 = iz(i, j, 1, n1, n2, n3, Npos)
            masa[j] += solution[key1]*mass_z[i]

            if solution[key0] == 1 or solution[key1] == 1:
                masa_c[j] = 'blue'

    fig = plt.figure()
    plt.xlabel('Position')
    plt.ylabel('Accumulated Mass')


    plt.bar(pos, masa, color=masa_c, edgecolor='black')
    plt.bar(pos, masa_top, bottom=masa, edgecolor='black', color='yellow')
    caja1 = mpatches.Patch(color='red', label='Type 1')
    caja2 = mpatches.Patch(color='yellow', label='Type 2')
    caja3 = mpatches.Patch(color='blue', label='Type 3')
    plt.legend(handles=[caja1, caja2, caja3], loc='upper center', ncol=3)
    plt.suptitle('Total mass ' + str(sum(masa)+sum(masa_top)))
    #plt.grid()
    ax = fig.gca()
    ax.set_xticks(pos)
    fig.savefig("test.png")
    plt.show()



# Convert a QUBO dictionary to a matrix
#
def dict_to_array(Q, numvars):
    qarray = np.zeros((numvars, numvars))

    for key in Q:
        qarray[key[0]][key[1]] = Q[key]
        #qarray[key[1]][key[0]] = Q[key]

    return qarray


# Get the number of variables in a QUBO problem
#
def problem_size(Q):
    numvars = 0

    for key in Q.keys():
        if key[0] > numvars:
            numvars = key[0]
        if key[1] > numvars:
            numvars = key[1]
    return numvars + 1



# Save to file a qubo function
#
def writequbo(Q, filename, constant):
    diag = {}
    offdiag = {}
    numvars = 0

    for key in Q.keys():
        if key[0]==key[1]:
            diag[key] = Q[key]
        else:
            offdiag[key] = Q[key]

        if key[0] > numvars:
            numvars = key[0]
        if key[1] > numvars:
            numvars = key[1]

    # save data to file
    fid = open(filename+'.qubo', 'w')
    fid.write('c Airbus qubo model\n')
    fid.write('c Constant term {}\n'.format(constant))
    fid.write('p qubo 0 {} {} {}\n'.format(numvars+1, len(diag),len(offdiag)))

    for key in diag:
        fid.write(str(key[0]) + ' ' + str(key[1]) + ' ' + str(diag[key]) + '\n')
    for key in offdiag:
        fid.write(str(key[0]) + ' ' + str(key[1]) + ' ' + str(offdiag[key]) + '\n')

    fid.close()
    return numvars + 1




# Save to file qubo problem in CPLEX format
#
def writequbo_cplex(Q, filename, constant):
    diag = {}
    offdiag = {}
    numvars = 0

    for key in Q.keys():
        if key[0]==key[1]:
            diag[key] = Q[key]
        else:
            offdiag[key] = Q[key]

        if key[0] > numvars:
            numvars = key[0]
        if key[1] > numvars:
            numvars = key[1]

    # File id to create the data
    fid_data = open(filename+'.dat', 'w')

    fid_data.write('constant = {};\n\n'.format(constant))

    fid_data.write('nvars = {};\n\n'.format(numvars+1))

    fid_data.write('dim_h = {};\n\n'.format(len(diag)))

    fid_data.write('dim_J = {};\n\n'.format(len(offdiag)))

    # save coefficients of vector h
    fid_data.write('h = [ ')

    iter1 = 1
    for key in diag:
        if iter1<len(diag):
            fid_data.write('{}, '.format(diag[key]))
        else:
            fid_data.write('{} ];\n\n'.format(diag[key]))

        iter1 = iter1+1



    # save index variable of vector h
    fid_data.write('hi = [ '.format(len(diag)))

    iter1 = 1
    for key in diag:
        if iter1<len(diag):
            fid_data.write('{}, '.format(key[0]+1))
        else:
            fid_data.write('{} ];\n\n'.format(key[0]+1))

        iter1 = iter1+1



    # save coefficient of vector J
    fid_data.write('jc = [ ')

    iter1 = 1
    for key in offdiag:
        if iter1<len(offdiag):
            fid_data.write('{}, '.format(offdiag[key]))
        else:
            fid_data.write('{} ];\n\n'.format(offdiag[key]))

        iter1 = iter1+1

    # save index variable of vector J (row)
    fid_data.write('jy = [ ')

    iter1 = 1
    for key in offdiag:
        if iter1<len(offdiag):
            fid_data.write('{}, '.format(key[0]+1))
        else:
            fid_data.write('{} ];\n\n'.format(key[0]+1))

        iter1 = iter1+1


    # save index variable of vector J (column)
    fid_data.write('jx = [ ')

    iter1 = 1
    for key in offdiag:
        if iter1<len(offdiag):
            fid_data.write('{}, '.format(key[1]+1))
        else:
            fid_data.write('{} ];\n\n'.format(key[1]+1))

        iter1 = iter1+1

    fid_data.close()

    return numvars + 1




# Leer archivo de containers
#
def leer_archivo(archivo):
    f = open(archivo,'r')
    containers = f.readlines()
    f.close()
    return containers


# Carga las masas de los containers
#
def crear_masas(archivo):

    containers = leer_archivo(archivo)

    # Crea una lista de containers con enteros
    datos = []
    for x in containers:
        cont = [int(y) for y in x.split(',')]
        datos.append(cont)

    # Creamos las variables x, y, z
    mass_x=[]
    mass_y=[]
    mass_z=[]

    for c in datos:
        if c[1] == 1:# si el container es de tipo 1, se pone en x
            mass_x.append(c[0])
        elif c[1] == 2: # si el container es de tipo 2, se pone en y
            mass_y.append(c[0])
        else:# si el container es de tipo 3, se pone en z
            mass_z.append(c[0])
    return mass_x, mass_y, mass_z;


#  Distancia de un contenedor al centro de la aeronave
#
def dist(j,N,L):
    return (2*j-N-1)*L/(2*N)


# Fuselaja simetrico
#
def Smax(j, S0, Npos, Len):
    if j<=Npos/2:
        s = 2*S0*dist(j, Npos, Len)/Len+S0
    else:
        s = -2*S0*dist(j, Npos, Len)/Len+S0
    return s



# Indexes for vaiables x's
#
def  ix(i, j, Npos):
    return j + i*Npos


def invfx(p, Npos):
    a = math.floor(p/Npos)
    b = p%Npos
    return [a, b]


# Indexes for vaiables y's
#
def iy(i, j, n1, Npos):
    return j + i*Npos + n1*Npos


def invfy(p, n1, Npos):
    p1 = p - n1*Npos
    a = math.floor(p1/Npos)
    b = p1%Npos
    return [a, b]


# Indexes for vaiables z's
#
def iz(i, j, k, n1, n2, n3, Npos):
    return j + i*Npos + k*n3*Npos + n1*Npos + n2*Npos


def invfz(p, n1, n2, n3, Npos):
    p0 = p - (n1+n2)*Npos
    a = math.floor(p0/(Npos*n3))
    p1 = p0
    if p0 > n3*Npos-1:
        p1 = p0 - n3*Npos

    b = math.floor(p1/Npos)
    c = p1%Npos
    return [b, c, a]


# Indexes for slack vaiables
#
def islack(i, j, n1, n2, n3, Npos, type_var, offset):
    if type_var==0:
        return i + 1*j + (n1 + n2 + 2*n3)*Npos
    elif type_var==1:
        return i + (n1 + n2 + 2*n3)*Npos + offset




# Return an ordered pair (x,y) with x <= y
#
def ipair(x, y):
    if x <= y:
        return (x, y)
    else:
        return (y, x)



# If key exists in Q then Q[key] add value, otherwise, Q[key] = values is defined
#
def updateKey(Q, key, value):
    if key in Q.keys():
        Q[key] = Q[key] + value
    else:
        Q[key] = value
    return Q



# Funcion para validar una solucion devuelta por qbsolv
#
def validationSol(mass_x, mass_y, mass_z, Npos, Wp, solution, constant, verbose):
    n1 = len(mass_x)
    n2 = len(mass_y)
    n3 = len(mass_z)

    if verbose==1:
        print('Solution {}'.format(solution))

    result = 0

    # Allocation constraints eq. 2

    # Restriction eq. 2a
    for j in range(Npos):
        t1 = 0
        t2 = 0
        t3 = 0
        for i in range(n1):
            key = ix(i, j, Npos)
            t1 += solution[key]
        for i in range(n2):
            key = iy(i, j, n1, Npos)
            t2 += solution[key]
        for i in range(n3):
            for k in range(2):
                key = iz(i, j, k, n1, n2, n3, Npos)
                t3 += solution[key]

        if t1 + 0.5*t2 + t3 > 1:
            result = 1
            if verbose==1:
                print('--> Do not satisfies 2a.')


    # Restriction eq. 2b
    for i in range(n1):
        t = 0
        for j in range(Npos):
            key = ix(i,j,Npos)
            t += solution[key]
        if t > 1:
            result = 1
            if verbose==1:
                print('--> Do not satisfies 2b.')

    # Restriction eq. 2c
    for i in range(n2):
        t = 0
        for j in range(Npos):
            key = iy(i,j,n1,Npos)
            t += solution[key]
        if t > 1:
            result = 1
            if verbose==1:
                print('--> Do not satisfies 2c.')


    # Restriction eq. 2d
    for i in range(n3):
        t = 0
        for j in range(Npos):
            key = iz(i,j,0,n1,n2,n3,Npos)
            t += solution[key]
        if t > 1:
            result = 1
            if verbose==1:
                print('--> Do not satisfies 2d.')

    # Restriction eq. 2e
    for i in range(n3):
        t = 0
        for j in range(Npos):
            key = iz(i,j,1,n1,n2,n3,Npos)
            t += solution[key]
        if t > 1:
            result = 1
            if verbose==1:
                print('--> Do not satisfies 2e.')


    # Restriction eq. 2f
    for i in range(n3):
        for j in range(Npos-1):
            key1 = iz(i, j, 0, n1, n2, n3, Npos)
            key2 = iz(i, j+1, 1, n1, n2, n3, Npos)
            if solution[key1] != solution[key2]:
                result = 1
                if verbose==1:
                    print('--> Do not satisfies 2f.')

    # Restrictions eqs. 2g and 2h
    for i in range(n3):
        key1 = iz(i, Npos-1, 0, n1, n2, n3, Npos)
        key2 = iz(i, 0, 1, n1, n2, n3, Npos)
        if solution[key1] != 0:
            result = 1
            if verbose==1:
                print('--> Do not satisfies 2g.')
        if solution[key2] != 0:
            result = 1
            if verbose==1:
                print('--> Do not satisfies 2h.')



    # Objective 1 in eq. 3
    t1 = 0
    t2 = 0
    t3 = 0

    for j in range(Npos):
        for i in range(n1):
            key = ix(i,j,Npos)
            t1 += mass_x[i]*solution[key]
        for i in range(n2):
            key = iy(i,j,n1,Npos)
            t2 += mass_y[i]*solution[key]
        for i in range(n3):
            key = iz(i,j,0,n1,n2,n3,Npos)
            t3 += mass_z[i]*solution[key]

    obj1 = t1 + t2 + t3

    if obj1 > Wp:
        result = 1
        if verbose==1:
            print('--> Do not satisfies 4 (maximum payload {}).'.format(Wp))

    if verbose==1:
        print('==> Payload of the solution: {}\n'.format(obj1))

    return result, obj1




# Comparison with Mathematica
#
def QuboCompare(Q, QUBO):

    print('Q vs QUBO')
    for key in Q.keys():
        if key in QUBO.keys():
            if abs(Q[key]-QUBO[key]) > 0.01:
                print('{:.3f}\t{:.3f}\t{:.3f}'.format(Q[key],QUBO[key], abs(Q[key]-QUBO[key])))
            #else:
            #    print('Coefficients are not equal')
        else:
            print('Key {} does not exists in QUBO. Coefficient {}'.format(key, Q[key]))

    print('\nQUBO vs Q')
    for key in QUBO.keys():
        if key in Q.keys():
            if abs(Q[key]-QUBO[key]) > 0.01:
                print('{:.3f}\t{:.3f}\t{:.3f}'.format(Q[key],QUBO[key], abs(Q[key]-QUBO[key])))
            #else:
            #    print('Coefficients are not equal')
        else:
            print('Key {} does not exists in Q'.format(key))










# Borra los terminos nulos
#
def RemoveZeroKeys(Q):
    Q1=Q
    nullkeys=[]
    for key in Q1.keys():
        if abs(Q1[key]) < 1e-10:
            nullkeys.append(key)

    for key in nullkeys:
        del Q1[key]

    return Q1




# Funcion de energia 'E_alloc' de la ecuacion (27)
#
def quboAllocation(Q1, mass_x, mass_y, mass_z, Npos, offset, const):

    Q = Q1

    # Cantidad de contenedores de los tres tipos
    n1 = len(mass_x)
    n2 = len(mass_y)
    n3 = len(mass_z)


    constant = const;

    # Ealloc = Sum_j (2 Sum_i x_ij + Sum_i y_ij + 2 Sum_ik z_ijk
    #          + ep_1j + 2ep_2j - 2)**2
    #        = sum_j (2*A1 + A2 + 2*A3 + ep1 + 2*ep2 - 2)**2

    for j in range(Npos):
        # Indexes for vaiables x_ij, y_ij, z_ijk and ep_j{1,2}
        list_x = [ix(i,j,Npos) for i in range(n1)]
        list_y = [iy(i,j,n1,Npos) for i in range(n2)]
        list_z = [iz(i,j,k,n1,n2,n3,Npos) for i in range(n3) for k in range(2)]
        list_slack = [islack(i,j,n1,n2,n3,Npos,0, offset) for i in range(1)]

        len_x = n1
        len_y = n2
        len_z = 2*n3

        # -8 A1
        for x in list_x:
            Q = updateKey(Q, (x, x), -8)

        # -4 A2
        for y in list_y:
            Q = updateKey(Q, (y, y), -4)

        # -8 A3
        for z in list_z:
            Q = updateKey(Q, (z, z), -8)


        # + 4 A1 A2
        for x in list_x:
            for y in list_y:
                Q = updateKey(Q, ipair(x, y), 4)

        # + 8 A1 A3
        for x in list_x:
            for z in list_z:
                Q = updateKey(Q, ipair(x, z), 8)

        # + 4 A2 A3
        for y in list_y:
            for z in list_z:
                Q = updateKey(Q, ipair(y, z), 4)

        # + 4 A1**2
        for x in list_x:
            Q = updateKey(Q, (x,x), 4)

        for x1 in range(len_x):
            for x2 in range(x1+1,len_x):
                key = ipair(list_x[x1], list_x[x2])
                Q = updateKey(Q, key, 8)

        # + A2**2
        for y in list_y:
            Q = updateKey(Q, (y,y), 1)

        for y1 in range(len_y):
            for y2 in range(y1+1,len_y):
                key = ipair(list_y[y1], list_y[y2])
                Q = updateKey(Q, key, 2)

        # + 4 A3**2
        for z in list_z:
            Q = updateKey(Q, (z,z), 4)

        for z1 in range(len_z):
            for z2 in range(z1+1,len_z):
                key = ipair(list_z[z1], list_z[z2])
                Q = updateKey(Q, key, 8)

        # + 8 A1 ep1
        for x in list_x:
            key = ipair(x, list_slack[0])
            Q = updateKey(Q, key, 8)


        # + 4 A2 ep1
        for y in list_y:
            key = ipair(y, list_slack[0])
            Q = updateKey(Q, key, 4)



        # + 8 A3 ep1
        for z in list_z:
            key = ipair(z, list_slack[0])
            Q = updateKey(Q, key, 8)


        # - 4 ep2
        key = (list_slack[0], list_slack[0])
        Q = updateKey(Q, key, -4)


        constant += 4

    # Segundo termino ecuacion (27)
    # Sum_i Sum_{j1 < j2} x_ij_1 x_ij_2
    for i in range(n1):
        for j1 in range(Npos):
            for j2 in range(j1+1, Npos):
                x1 = ix(i,j1,Npos)
                x2 = ix(i,j2,Npos)
                key = ipair(x1,x2)
                Q = updateKey(Q, key, 1)

    # Tercer termino ecuacion (27)
    # Sum_i Sum_{j1 < j2} y_ij_1 y_ij_2
    for i in range(n2):
        for j1 in range(Npos):
            for j2 in range(j1+1, Npos):
                y1 = iy(i, j1, n1, Npos)
                y2 = iy(i, j2, n1, Npos)
                key = ipair(y1,y2)
                Q = updateKey(Q, key, 1)

    # Cuarto y quinto termino ecuacion (27)
    # Sum_i Sum_{j1 < j2} z_ij_11 z_ij_21  +
    # Sum_i Sum_{j1 < j2} z_ij_12 z_ij_22
    for i in range(n3):
        for j1 in range(Npos):
            for j2 in range(j1+1, Npos):
                z1 = iz(i, j1, 0, n1, n2, n3, Npos)
                z2 = iz(i, j2, 0, n1, n2, n3, Npos)
                Q = updateKey(Q, ipair(z1, z2), 1)

                z1 = iz(i, j1, 1, n1, n2, n3, Npos)
                z2 = iz(i, j2, 1, n1, n2, n3, Npos)
                Q = updateKey(Q, ipair(z1, z2), 1)


    # Sexto  termino ecuacion (27)
    # Sum_i Sum_{j=1}^{N-1} [ z_ij1 + z_i(j+1)2 - 2 z_ij1 z_i(j+1)2 ]
    for i in range(n3):
        for j in range(Npos-1):
            z1 = iz(i, j, 0, n1, n2, n3, Npos)
            Q = updateKey(Q, (z1,z1), 1)

            z2 = iz(i, j+1, 1, n1, n2, n3, Npos)
            Q = updateKey(Q, (z2,z2), 1)

            z12 = ipair(z1, z2)
            Q = updateKey(Q, z12, -2)


    # Septimo termino ecuacion (27)
    # Sum_i (z_iN1 + z_i12)
    for i in range(n3):
        z = iz(i, Npos-1, 0, n1, n2, n3, Npos)
        Q = updateKey(Q, (z,z), 1)

        z = iz(i, 0, 1, n1, n2, n3, Npos)
        Q = updateKey(Q, (z,z), 1)

    offset1 = offset + Npos

    return Q, constant, offset1






# Construccion de la restriccion de maxima carga Ec. 28
#
def quboPayload(Q1, mass_x, mass_y, mass_z, Npos, Wp, offset, const, Weight):
    # We simplify constraint (3) as (F - Wp + Sw)**2
    # whose expand is F**2 + 2*F*Sw + Sw**2 - 2*F*Wp - 2*Sw*Wp + Wp**2

    Q = Q1

    # Cantidad de contenedores de los tres tipos
    n1 = len(mass_x)
    n2 = len(mass_y)
    n3 = len(mass_z)

    constant = const;

    # Numero de variables slack para representar Wp
    nbits = math.ceil(math.log2(Wp))
    #print('\nBinary expansion of Wp={} is {}'.format(Wp, nbits))


    # Indexes for variables x_ij, y_ij, z_ij1
    list_x = [ix(i,j,Npos) for j in range(Npos) for i in range(n1)]
    list_y = [iy(i,j,n1,Npos) for j in range(Npos) for i in range(n2)]
    list_z = [iz(i,j,0,n1,n2,n3,Npos) for j in range(Npos) for i in range(n3)]


    # List of masses according to the enumeration above
    list_mass_x = []
    list_mass_y = []
    list_mass_z = []

    for k in range(Npos):
        list_mass_x = np.concatenate([list_mass_x, mass_x])
        list_mass_y = np.concatenate([list_mass_y, mass_y])
        list_mass_z = np.concatenate([list_mass_z, mass_z])



    # lista de variables y coeficientes para la expansion binaria de Wp

    list_slack = [islack(i,0,n1,n2,n3,Npos,1, offset) for i in range(nbits)]
    coeff_slack = [2**i for i in range(nbits)]


    #    Concatenate all masses in a one vector [list_mass_x , list_mass_y , list_mass_z]
    list_var = np.concatenate([list_x, list_y, list_z])
    list_mass = np.concatenate([list_mass_x, list_mass_y, list_mass_z])



    # Termino F**2
    len_list = len(list_var)

    for i in range(0,len_list):
        for j in range(i+1,len_list):
            key = ipair(list_var[i], list_var[j])
            Q = updateKey(Q, key, 2*list_mass[i]*list_mass[j]*Weight)

    for i in range(len_list):
        i1 = list_var[i]
        Q = updateKey(Q, (i1,i1), Weight*(list_mass[i]**2))


    # Termino 2*F*Sw
    for i in range(len_list):
        for j in range(nbits):
            key = ipair(list_var[i], list_slack[j])
            Q = updateKey(Q, key, 2*list_mass[i]*coeff_slack[j]*Weight)


    # Termino Sw**2
    for i in range(nbits):
        for j in range(i+1,nbits):
            key = ipair(list_slack[i], list_slack[j])
            Q = updateKey(Q, key, 2*coeff_slack[i]*coeff_slack[j]*Weight)


    for i in range(nbits):
        i1 = list_slack[i]
        Q = updateKey(Q, (i1,i1), Weight*(coeff_slack[i]**2))


    # Termino -2*Wp*F
    for i in range(len_list):
        k = list_var[i]
        Q = updateKey(Q, (k,k), -2*list_mass[i]*Wp*Weight)


    # Termino -2*Wp*Sw
    for i in range(nbits):
        k = list_slack[i]
        Q = updateKey(Q, (k,k), -2*coeff_slack[i]*Wp*Weight)


    # El termino constante se conserva  Wp**2 en la variable 'constant'
    constant += Weight*(Wp**2)
    offset1 = offset + nbits

    return Q, constant, offset1





# Center of gravity constraint (E_cg_min) Eq. 30
#
def quboGravityLeft(Q1, mass_x, mass_y, mass_z, Npos, xcgmin, xcge, We, Len, offset, const):
    # We simplify energy function (30) as (F - Wp + Sw)**2
    # whose expand is F**2 + 2*F*Sw + Sw**2 - 2*F*Wp - 2*Sw*Wp + Wp**2

    Q = Q1

    # Cantidad de contenedores de los tres tipos
    n1 = len(mass_x)
    n2 = len(mass_y)
    n3 = len(mass_z)

    constant = const

    # Indexes for vaiables x_ij, y_ij, z_ij1
    list_x = [ix(i,j,Npos) for j in range(Npos) for i in range(n1)]
    list_y = [iy(i,j,n1,Npos) for j in range(Npos) for i in range(n2)]
    list_z = [iz(i,j,k,n1,n2,n3,Npos) for j in range(Npos) for i in range(n3) for k in range(2)]

    # List of masses according to the enumeration above
    list_mass_x = []
    list_mass_y = []
    list_mass_z = []

    for j in range(Npos):
        mass_x_d = []
        mass_y_d = []
        mass_z_d = []
        dcg = xcgmin - dist(j+1,Npos,Len)

        for i in range(n1):
            mass_x_d.append(mass_x[i]*dcg)
        for i in range(n2):
             mass_y_d.append(mass_y[i]*dcg)
        for i in range(n3):
            mass_z_d.append(0.5*mass_z[i]*dcg)
            mass_z_d.append(0.5*mass_z[i]*dcg)

        list_mass_x = np.concatenate([list_mass_x, mass_x_d])
        list_mass_y = np.concatenate([list_mass_y, mass_y_d])
        list_mass_z = np.concatenate([list_mass_z, mass_z_d])

    # Numero de variables slack para representar We(x_cg_e - x_cg_min)
    Wecgmin = We*(xcge - xcgmin)
    nbits = math.ceil(math.log2(Wecgmin))

    # lista de variables y coeficientes para la expansion binaria de We(x_cg_e - x_cg_min)
    list_slack = [islack(i,0,n1,n2,n3,Npos, 1, offset) for i in range(nbits)]
    coeff_slack = [2**i for i in range(nbits)]


    # Concatenate all masses in a one vector [list_mass_x , list_mass_y , list_mass_z]
    list_var = np.concatenate([list_x, list_y, list_z])
    list_mass = np.concatenate([list_mass_x, list_mass_y, list_mass_z])
    len_list = len(list_var)


    # Termino F**2
    for i in range(len_list):
        for j in range(i+1,len_list):
            key = ipair(list_var[i], list_var[j])
            Q = updateKey(Q, key, 2*list_mass[i]*list_mass[j])


    for i in range(len_list):
        i1 = list_var[i]
        Q = updateKey(Q, (i1,i1), list_mass[i]**2)


    # Termino 2*F*Sw
    for i in range(len_list):
        for j in range(nbits):
            key = ipair(list_var[i], list_slack[j])
            Q = updateKey(Q, key, 2*list_mass[i]*coeff_slack[j])


    # Termino Sw**2
    for i in range(nbits):
        for j in range(i+1,nbits):
            key = ipair(list_slack[i], list_slack[j])
            Q = updateKey(Q, key, 2*coeff_slack[i]*coeff_slack[j])


    for i in range(nbits):
        i1 = list_slack[i]
        Q = updateKey(Q, (i1,i1), coeff_slack[i]**2)


    # Termino -2*We(xcge-xcgmin)*F
    for i in range(len_list):
        k = list_var[i]
        Q = updateKey(Q, (k,k), -2*list_mass[i]*Wecgmin)


    # Termino -2*We(xcge-xcgmin)*Sw
    for i in range(nbits):
        k = list_slack[i]
        Q = updateKey(Q, (k,k), -2*coeff_slack[i]*Wecgmin)


    # El termino constante se conserva  [We(xcge-xcgmin)]**2
    constant += Wecgmin**2

    offset1 = offset + nbits

    return Q, constant, offset1




# Center of gravity constraint (E_cg_max) Eq. 31
#
def quboGravityRight(Q1, mass_x, mass_y, mass_z, Npos, xcgmax, xcge, We, Len, offset, const):
    #    We simplify energy function (31) as (F - Wp + Sw)**2
    #    whose expand is F**2 + 2*F*Sw + Sw**2 - 2*F*Wp - 2*Sw*Wp + Wp**2

    Q = Q1

    # Cantidad de contenedores de los tres tipos
    n1 = len(mass_x)
    n2 = len(mass_y)
    n3 = len(mass_z)

    constant = const

    # Numero de variables slack para representar We(x_cg_max - x_cg_e)
    Wecgmax = We*(xcgmax - xcge)
    nbits = math.ceil(math.log2(Wecgmax))

    # Indexes for vaiables x_ij, y_ij, z_ij1
    list_x = [ix(i,j,Npos) for j in range(Npos) for i in range(n1)]
    list_y = [iy(i,j,n1,Npos) for j in range(Npos) for i in range(n2)]
    list_z = [iz(i,j,k,n1,n2,n3,Npos) for j in range(Npos) for i in range(n3) for k in range(2)]

    # List of masses according to the enumeration above
    list_mass_x = []
    list_mass_y = []
    list_mass_z = []

    for j in range(Npos):
        mass_x_d = []
        mass_y_d = []
        mass_z_d = []
        dcg = dist(j+1,Npos,Len) - xcgmax

        for i in range(n1):
            mass_x_d.append(mass_x[i]*dcg)
        for i in range(n2):
             mass_y_d.append(mass_y[i]*dcg)
        for i in range(n3):
            mass_z_d.append(0.5*mass_z[i]*dcg)
            mass_z_d.append(0.5*mass_z[i]*dcg)

        list_mass_x = np.concatenate([list_mass_x, mass_x_d])
        list_mass_y = np.concatenate([list_mass_y, mass_y_d])
        list_mass_z = np.concatenate([list_mass_z, mass_z_d])

    # lista de variables y coeficientes para la expansion binaria de We(x_cg_max - x_cg_e)

    list_slack = [islack(i,0,n1,n2,n3,Npos, 1, offset) for i in range(nbits)]
    coeff_slack = [2**i for i in range(nbits)]

    # Concatenate all masses in a one vector [list_mass_x , list_mass_y , list_mass_z]
    list_var = np.concatenate([list_x, list_y, list_z])
    list_mass = np.concatenate([list_mass_x, list_mass_y, list_mass_z])
    len_list = len(list_var)


    # Termino F**2
    for i in range(0,len_list):
        for j in range(i+1,len_list):
            key = ipair(list_var[i], list_var[j])
            Q = updateKey(Q, key, 2*list_mass[i]*list_mass[j])


    for i in range(len_list):
        i1 = list_var[i]
        Q = updateKey(Q, (i1,i1), list_mass[i]**2)


    # Termino 2*F*Sw
    for i in range(len_list):
        for j in range(nbits):
            key = ipair(list_var[i], list_slack[j])
            Q = updateKey(Q, key, 2*list_mass[i]*coeff_slack[j])


    # Termino Sw**2
    for i in range(nbits):
        for j in range(i+1,nbits):
            key = ipair(list_slack[i], list_slack[j])
            Q = updateKey(Q, key, 2*coeff_slack[i]*coeff_slack[j])


    for i in range(nbits):
        i1 = list_slack[i]
        Q = updateKey(Q, (i1,i1), coeff_slack[i]**2)


    # Termino -2*We(xcgmax-xcge)*F
    for i in range(len_list):
        k = list_var[i]
        Q = updateKey(Q, (k,k), -2*list_mass[i]*Wecgmax)


    # Termino -2*We(xcgmax-xcge)*Sw
    for i in range(nbits):
        k = list_slack[i]
        Q = updateKey(Q, (k,k), -2*coeff_slack[i]*Wecgmax)


    #    El termino constante se conserva  Wecgmax**2
    constant += Wecgmax**2

    offset1 = offset + nbits

    return Q, constant, offset1





# Objective function 1 (Eq. 3)
#
def quboFunction1(Q1, mass_x, mass_y, mass_z, Npos, offset, const, Wa):

    Q = Q1

    # Cantidad de contenedores de los tres tipos
    n1 = len(mass_x)
    n2 = len(mass_y)
    n3 = len(mass_z)

    constant = const;
    offset1 = offset

    # Indexes for variables x_ij, y_ij, z_ij1
    list_x = [ix(i,j,Npos) for j in range(Npos) for i in range(n1)]
    list_y = [iy(i,j,n1,Npos) for j in range(Npos) for i in range(n2)]
    list_z = [iz(i,j,0,n1,n2,n3,Npos) for j in range(Npos) for i in range(n3)]


    # List of masses according to the enumeration above
    list_mass_x = []
    list_mass_y = []
    list_mass_z = []

    for k in range(Npos):
        list_mass_x = np.concatenate([list_mass_x, mass_x])
        list_mass_y = np.concatenate([list_mass_y, mass_y])
        list_mass_z = np.concatenate([list_mass_z, mass_z])

    #    Concatenate all masses in a one vector [list_mass_x , list_mass_y , list_mass_z]
    list_var = np.concatenate([list_x, list_y, list_z])
    list_mass = np.concatenate([list_mass_x, list_mass_y, list_mass_z])


    # Funcion objetivo 1
    for i in range(len(list_var)):
        k = list_var[i]
        if (k, k) in Q.keys():
            Q[(k,k)] += Wa*list_mass[i]
        else:
            Q[(k,k)] = Wa*list_mass[i]

    return Q, constant, offset1





# Objective function 2 (Eq. 15)
#
def quboFunction2(Q1, mass_x, mass_y, mass_z, Npos, xcgt, Wp, xcge, Len, offset, const):
    # (F+a)**2 = a**2 + 2aF + F**2
    Q = Q1

    # Cantidad de contenedores de los tres tipos
    n1 = len(mass_x)
    n2 = len(mass_y)
    n3 = len(mass_z)

    constant = const;
    offset1 = offset

    # Indexes for variables x_ij, y_ij, z_ij1
    list_x = [ix(i,j,Npos) for j in range(Npos) for i in range(n1)]
    list_y = [iy(i,j,n1,Npos) for j in range(Npos) for i in range(n2)]
    list_z = [iz(i,j,0,n1,n2,n3,Npos) for j in range(Npos) for i in range(n3)]


    # List of masses according to the enumeration above
    list_mass_x = []
    list_mass_y = []
    list_mass_z = []

    for j in range(Npos):
        mass_x_d = []
        mass_y_d = []
        mass_z_d = []
        dcg = dist(j+1,Npos,Len) - xcgt

        for i in range(n1):
            mass_x_d.append(mass_x[i]*dcg)
        for i in range(n2):
             mass_y_d.append(mass_y[i]*dcg)
        for i in range(n3):
            mass_z_d.append(mass_z[i]*dcg)

        list_mass_x = np.concatenate([list_mass_x, mass_x_d])
        list_mass_y = np.concatenate([list_mass_y, mass_y_d])
        list_mass_z = np.concatenate([list_mass_z, mass_z_d])

    #    Concatenate all masses in a one vector [list_mass_x , list_mass_y , list_mass_z]
    list_var = np.concatenate([list_x, list_y, list_z])
    list_mass = np.concatenate([list_mass_x, list_mass_y, list_mass_z])
    len_list = len(list_var)

    # termino independiente Wp(xcge - xcgt)
    Wpcgt = Wp*(xcge-xcgt)


    # Termino F**2
    for i in range(0,len_list):
        for j in range(i+1,len_list):
            key = ipair(list_var[i], list_var[j])
            Q = updateKey(Q, key, 2*list_mass[i]*list_mass[j])


    for i in range(len_list):
        i1 = list_var[i]
        Q = updateKey(Q, (i1,i1), list_mass[i]**2)


    # Termino 2*Wp(xcge-xcgt)*F
    for i in range(len_list):
        k = list_var[i]
        Q = updateKey(Q, (k,k), 2*list_mass[i]*Wpcgt)


    #    El termino constante se conserva Wpcgt**2
    constant += Wpcgt**2

    return Q, constant, offset1





# Left shear penalization (Eq. 32)
#
def quboLeftShear(Q1, mass_x, mass_y, mass_z, Npos, S0, Len, offset, const):

    Q = Q1
    constant = const
    offset1 = offset

    # Cantidad de contenedores de los tres tipos
    n1 = len(mass_x)
    n2 = len(mass_y)
    n3 = len(mass_z)

    for j in range(math.floor(Npos/2)+1):

         # Indexes for vaiables x_ij, y_ij, z_ij1
        list_x = [ix(i,j1,Npos) for j1 in range(j+1) for i in range(n1)]
        list_y = [iy(i,j1,n1,Npos) for j1 in range(j+1) for i in range(n2)]
        list_z = [iz(i,j1,k,n1,n2,n3,Npos) for j1 in range(j+1) for i in range(n3) for k in range(2)]

        # List of masses according to the enumeration above
        list_mass_x = []
        list_mass_y = []
        list_mass_z = []

        for j1 in range(j+1):
            mass_x_d = []
            mass_y_d = []
            mass_z_d = []

            for i in range(n1):
                mass_x_d.append(mass_x[i])
            for i in range(n2):
                 mass_y_d.append(mass_y[i])
            for i in range(n3):
                mass_z_d.append(0.5*mass_z[i])
                mass_z_d.append(0.5*mass_z[i])

            list_mass_x = np.concatenate([list_mass_x, mass_x_d])
            list_mass_y = np.concatenate([list_mass_y, mass_y_d])
            list_mass_z = np.concatenate([list_mass_z, mass_z_d])


        # Numero de variables slack para representar Smax(j)
        smaxj = Smax(j+1, S0, Npos, Len)
        nbits = math.ceil(math.log2(smaxj))

        # lista de variables y coeficientes para la expansion binaria de Smax(j)
        list_slack = [islack(i,0,n1,n2,n3,Npos, 1, offset1) for i in range(nbits)]
        coeff_slack = [2**i for i in range(nbits)]

        # Concatenate all masses in a one vector [list_mass_x , list_mass_y , list_mass_z]
        list_var = np.concatenate([list_x, list_y, list_z])
        list_mass = np.concatenate([list_mass_x, list_mass_y, list_mass_z])
        len_list = len(list_var)


        # Termino F**2
        for i in range(len_list):
            for j1 in range(i+1,len_list):
                key = ipair(list_var[i], list_var[j1])
                Q = updateKey(Q, key, 2*list_mass[i]*list_mass[j1])


        for i in range(len_list):
            i1 = list_var[i]
            Q = updateKey(Q, (i1,i1), list_mass[i]**2)


        # Termino 2*F*Sw
        for i in range(len_list):
            for j1 in range(nbits):
                key = ipair(list_var[i], list_slack[j1])
                Q = updateKey(Q, key, 2*list_mass[i]*coeff_slack[j1])


        # Termino Sw**2
        for i in range(nbits):
            for j1 in range(i+1,nbits):
                key = ipair(list_slack[i], list_slack[j1])
                Q = updateKey(Q, key, 2*coeff_slack[i]*coeff_slack[j1])


        for i in range(nbits):
            i1 = list_slack[i]
            Q = updateKey(Q, (i1,i1), coeff_slack[i]**2)


        # Termino -2*Smax(j)*F
        for i in range(len_list):
            k = list_var[i]
            Q = updateKey(Q, (k,k), -2*list_mass[i]*smaxj)


        # Termino -2*Smax(j)*Sw
        for i in range(nbits):
            k = list_slack[i]
            Q = updateKey(Q, (k,k), -2*coeff_slack[i]*smaxj)


        # El termino constante se conserva  Smax(j)**2 e incrementa offset1
        constant += smaxj**2
        offset1 += nbits

    return Q, constant, offset1





# Right shear penalization (Eq. 33)
#
def quboRightShear(Q1, mass_x, mass_y, mass_z, Npos, S0, Len, offset, const):

    Q = Q1
    constant = const
    offset1 = offset

    # Cantidad de contenedores de los tres tipos
    n1 = len(mass_x)
    n2 = len(mass_y)
    n3 = len(mass_z)

    for j in range(math.floor(Npos/2)+1,Npos):

         # Indexes for vaiables x_ij, y_ij, z_ij1
        list_x = [ix(i,j1,Npos) for j1 in range(Npos-1,j-1,-1) for i in range(n1)]
        list_y = [iy(i,j1,n1,Npos) for j1 in range(Npos-1,j-1,-1) for i in range(n2)]
        list_z = [iz(i,j1,k,n1,n2,n3,Npos) for j1 in range(Npos-1,j-1,-1) for i in range(n3) for k in range(2)]

        # List of masses according to the enumeration above
        list_mass_x = []
        list_mass_y = []
        list_mass_z = []


        for j1 in range(Npos-1,j-1,-1):
            mass_x_d = []
            mass_y_d = []
            mass_z_d = []

            for i in range(n1):
                mass_x_d.append(mass_x[i])
            for i in range(n2):
                 mass_y_d.append(mass_y[i])
            for i in range(n3):
                mass_z_d.append(0.5*mass_z[i])
                mass_z_d.append(0.5*mass_z[i])

            list_mass_x = np.concatenate([list_mass_x, mass_x_d])
            list_mass_y = np.concatenate([list_mass_y, mass_y_d])
            list_mass_z = np.concatenate([list_mass_z, mass_z_d])


        # Numero de variables slack para representar Smax(j)
        smaxj = Smax(j+1, S0, Npos, Len)
        nbits = math.ceil(math.log2(smaxj))

        # lista de variables y coeficientes para la expansion binaria de Smax(j)
        list_slack = [islack(i,0,n1,n2,n3,Npos, 1, offset1) for i in range(nbits)]
        coeff_slack = [2**i for i in range(nbits)]

        # Concatenate all masses in a one vector [list_mass_x , list_mass_y , list_mass_z]
        list_var = np.concatenate([list_x, list_y, list_z])
        list_mass = np.concatenate([list_mass_x, list_mass_y, list_mass_z])
        len_list = len(list_var)


        # Termino F**2
        for i in range(len_list):
            for j1 in range(i+1,len_list):
                key = ipair(list_var[i], list_var[j1])
                Q = updateKey(Q, key, 2*list_mass[i]*list_mass[j1])


        for i in range(len_list):
            i1 = list_var[i]
            Q = updateKey(Q, (i1,i1), list_mass[i]**2)


        # Termino 2*F*Sw
        for i in range(len_list):
            for j1 in range(nbits):
                key = ipair(list_var[i], list_slack[j1])
                Q = updateKey(Q, key, 2*list_mass[i]*coeff_slack[j1])


        # Termino Sw**2
        for i in range(nbits):
            for j1 in range(i+1,nbits):
                key = ipair(list_slack[i], list_slack[j1])
                Q = updateKey(Q, key, 2*coeff_slack[i]*coeff_slack[j1])


        for i in range(nbits):
            i1 = list_slack[i]
            Q = updateKey(Q, (i1,i1), coeff_slack[i]**2)


        # Termino -2*Smax(j)*F
        for i in range(len_list):
            k = list_var[i]
            Q = updateKey(Q, (k,k), -2*list_mass[i]*smaxj)


        # Termino -2*Smax(j)*Sw
        for i in range(nbits):
            k = list_slack[i]
            Q = updateKey(Q, (k,k), -2*coeff_slack[i]*smaxj)


        # El termino constante se conserva  Smax(j)**2 e incrementa offset1
        constant += smaxj**2
        offset1 += nbits

    return Q, constant, offset1





# Scale every term in Q by a factor w
#
def quboScale(Q1, weight, const):

    Q = Q1
    for key in Q:
        Q[key] *= weight

    constant = const*weight
    return Q, constant




# Construccion del modelo QUBO de la seccion 4.2.
#
def quboEtotal(mass_x, mass_y, mass_z, Npos, Wp, xcgmin, xcgmax, xcge, We, xcgt,
                Len, S0, Wa, Weight_alloc, Weight_payload):

    # Definicion del diccionario Q.
    Q = {}

    constant = 0;
    offset = 0

    # Eq. 27
    Q, constant, offset = quboAllocation(Q, mass_x, mass_y, mass_z, Npos, offset, constant)

    # Scale penalty terms
    Q, constant = quboScale(Q, Weight_alloc, constant)

    # Eq. 28
    Q, constant, offset = quboPayload(Q, mass_x, mass_y, mass_z, Npos, Wp, offset, constant, Weight_payload)

    # Eq. 30
    #Q, constant, offset = quboGravityLeft(Q, mass_x, mass_y, mass_z, Npos, xcgmin, xcge, We, Len, offset, constant)

    # Eq. 31
    #Q, constant, offset = quboGravityRight(Q, mass_x, mass_y, mass_z, Npos, xcgmax, xcge, We, Len, offset, constant)

    # Eq. 32
    #Q, constant, offset = quboLeftShear(Q, mass_x, mass_y, mass_z, Npos, S0, Len, offset, constant)

    # Eq. 33
    #Q, constant, offset = quboRightShear(Q, mass_x, mass_y, mass_z, Npos, S0, Len, offset, constant)

    # Eq. 3
    Q, constant, offset = quboFunction1(Q, mass_x, mass_y, mass_z, Npos, offset, constant, Wa)

    # Eq. 15
    #Q, constant, offset = quboFunction2(Q, mass_x, mass_y, mass_z, Npos, xcgt, Wp, xcge, Len, offset, constant)


    # Limpia diccionario
    Q = RemoveZeroKeys(Q)

    return Q, constant, offset





# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  Perform a sampling of the objective and penalty functions
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def sampling(Npos_, Wp_, Len_, We_, S0_, xcgmin_, xcgmax_, xcge_, xcgt_,
            numvars, constant, instance_name_, num_instance):
    # Parametros
    Npos = Npos_
    Wp = Wp_
    Len = Len_
    We = We_
    S0 = S0_
    xcgmin = xcgmin_
    xcgmax = xcgmax_
    xcge = xcge_
    xcgt = xcgt_
    obj = 'min'

    # Path to instances
    instance_name = instance_name_
    path_instances = "instancias/" + instance_name + "/"

    # name of the instances
    filename = "i" + str(num_instance) + "-" + instance_name

    # read the masses
    mass_x, mass_y, mass_z = crear_masas(path_instances+filename)

    print('\nSampling ...')

    num_calls = 10000
    avg_energy = 0
    avg_p1 = 0
    avg_p2 = 0

    for rep in range(num_calls):
        assignment = np.arange(numvars)
        for k in range(numvars):
            assignment[k] = random.randint(0, 1)

        val, energy = validationSol(mass_x, mass_y, mass_z, Npos, Wp, assignment, constant, 0)
        penalty1 = quboAllocation_validation(assignment, mass_x, mass_y, mass_z, Npos, 0)
        penalty2 = quboPayload_validation(assignment, mass_x, mass_y, mass_z, Npos, Wp, Npos, 1)  #1/1000

        avg_energy = avg_energy + energy
        avg_p1 = avg_p1 + np.sum(penalty1)
        avg_p2 = avg_p2 + penalty2

    avg_energy = avg_energy / num_calls
    avg_p1 = avg_p1 / num_calls
    avg_p2 = avg_p2 / num_calls

    print('avg-energy   = {}'.format(avg_energy))
    print('avg-penalty1 = {}'.format(avg_p1))
    print('avg-penalty2 = {}'.format(avg_p2))




#  Solve a qubo problem calling sqaod
#
def sqaod_caller(Q, mass_x, mass_y, mass_z, Npos, numvars, Wp, constant, Wa, Weight_alloc, Weight_payload, num_calls_, obj):

    # - - - - - - - - - - - - - - - -
    #     Calling sqaod solver
    # - - - - - - - - - - - - - - - -

    # 1. set problem.  As an example, a matrix filled with 1. is used.
    W = dict_to_array(Q, numvars)

    # 2. choosing solver .
    sol = sq.cpu # use CPU annealer
    # If you want to use CUDA, check CUDA availability with sq.is_cuda_available().
    if sq.is_cuda_available() :
        import sqaod.cuda
        sol = sqaod.cuda


    # - - - - - - - - - - - - - - - - - - -
    #             Repetitions
    # - - - - - - - - - - - - - - - - - - -

    # Number of calls to SimulatedAnnealingSampler
    num_calls = num_calls_
    count_solutions = 0

    start_time = time.time()

    cost_alloc = 0
    cost_payload = 0
    list_energies = [0 for i in range(num_calls)]

    print('\nCalling sqaod solver:')

    for k in range(num_calls):
        print('#{}'.format(k))

        # 3. instanciate solver
        ann = sol.dense_graph_annealer()

        # 4. (optional) set random seed.
        seed = random.randrange(sys.maxsize)
        rng = random.Random(seed)
        #print("Seed was:", seed)
        ann.seed(seed)

        # 5. Setting problem
        # Setting W and optimize direction (minimize or maxminize)
        # n_trotters is implicitly set to N/4 by default.
        #ann.set_qubo(W, sq.minimize)

        # 6. set preferences,
        # The following line set n_trotters is identical to the dimension of W.
        #ann.set_preferences(n_trotters = W.shape[0])

        # altternative for 4. and 5.
        # W, optimize direction and n_trotters are able to be set on instantiation.

        if obj=='min':
            ann = sol.dense_graph_annealer(W, sq.minimize, n_trotters = W.shape[0])
        else:
            ann = sol.dense_graph_annealer(W, sq.maximize, n_trotters = W.shape[0])


        # 7. get ising model paramters. (optional)
        # When W and optimize dir are set, ising hamiltonian of h, J and c are caluclated.
        # By using get_hamiltonian() to get these values.
        #h, J, c = ann.get_hamiltonian()
        #print('h={}\n'.format(h))
        #print('J={}\n'.format(J))
        #print('c={}\n'.format(c))

        # 8. showing preferences (optional)
        # preferences of solvers are obtained by calling get_preference().
        # preferences is always repeseted as python dictionay object.
        #print('\n{}'.format(ann.get_preferences()))

        # 9. prepare to run anneal. Annealers must be prepared
        #  before calling randomize_spin() and anneal_one_step().
        ann.prepare()

        # 10. randomize or set x(0 or 1) to set the initial state (mandatory)
        ann.randomize_spin()

        # 11. annealing
        Ginit = 5.
        Gfin = 0.01
        beta = 1. / 0.02
        tau = 0.99

        # annealing loop
        G = Ginit
        while Gfin <= G :
            # 11. call anneal_one_step to try flipping spins for (n_bits x n_trotters) times.
            ann.anneal_one_step(G, beta)
            G *= tau
            # 12. you may call get_E() to get a current E value.
            # ann.get_E()

        # 13. some methods to get results
        # - Get list of E for every trogger.
        #E = ann.get_E()
        # - Get annealed q. get_q() returns q matrix as (n_trotters, N)
        #q = ann.get_q()
        # - Get annealed x. get_x() returns x matrix as (n_trotters, N)
        #x = ann.get_x()

        # 14. creating summary object
        summary = sq.make_summary(ann)

        # 15. get the best engergy(for min E for minimizing problem, and max E for maxmizing problem)
        #qubo_energy = summary.E + constant
        #print('\nE={}\n'.format(qubo_energy))

        # 16. show the number of solutions that has the same energy of the best E.
        #print('Number of solutions : {}'.format(len(summary.xlist)))

        # 17. show solutions.
        #for idx in range(len(summary.xlist)):
        for idx in range(1):
            result, energy_sol = validationSol(mass_x, mass_y, mass_z, Npos, Wp, summary.xlist[idx], constant, 0)

            # Best energy found
            qubo_energy = summary.E + constant

            list_energies[k] = qubo_energy

            # if valid solution
            if result==0:
                # if best energy found coincides with the actual payload of the assignment
                if qubo_energy==Wa*energy_sol:
                    print('\n Iteration {}\n'.format(k+1))
                    print (summary.xlist[idx])
                    print('payload     = {}'.format(Wa*energy_sol))
                    print('qubo_energy = {}\n'.format(qubo_energy))

                    print('------------------')
                    print('Energy validation:')
                    print('------------------')

                    print('payload = {}'.format(Wa*energy_sol))

                    cost_list = quboAllocation_validation(summary.xlist[idx], mass_x, mass_y, mass_z, Npos, 0)
                    cost1 = np.sum(cost_list)*Weight_alloc
                    print("cost_alloc = {}".format(cost1))

                    cost2 = quboPayload_validation(summary.xlist[idx], mass_x, mass_y, mass_z, Npos, Wp, Npos, Weight_payload)
                    print("payload_cost = {}".format(cost2))

                    total_cost = Wa*energy_sol + cost1 + cost2
                    print('Total cost {}'.format(total_cost))

                    PlotBarContainers(summary.xlist[idx], Npos, mass_x, mass_y, mass_z)

                    count_solutions = count_solutions + 1

    # Summary
    #
    print('count_solutions = {}, with beta = {}'.format(count_solutions, beta))
    print("--- %s seconds ---" % (time.time() - start_time))

    # Sort energies
    list_energies.sort()

    return list_energies




#  Solve a qubo problem calling simulated annealing
#
def sq_caller(Q, mass_x, mass_y, mass_z, Npos, numvars, Wp, constant, Wa, Weight_alloc, Weight_payload, num_calls_, obj):
    # - - - - - - - - - - - - - - - - - - - - -
    #       calling sq solver
    # - - - - - - - - - - - - - - - - - - - - -


    print('calling sa solver:')

    # Weight for the objective function. -1 for minimization and +1 for maximization
    if obj=='min':
        find_max_val = False
    else:
        find_max_val = True

    # Number of calls to SimulatedAnnealingSampler
    num_calls = num_calls_
    count_solutions = 0

    start_time = time.time()


    # to save the best solution per call
    list_energies = [0 for i in range(num_calls)]

    #  Repetitions
    #
    for k in range(num_calls):
        print('#{} '.format(k+1))
        sampler = neal.SimulatedAnnealingSampler()
        response = QBSolv().sample_qubo(Q, solver=sampler, find_max=find_max_val)
        energies = response.data_vectors['energy'] + constant
        solutions = list(response.samples())

        result, energy_sol = validationSol(mass_x, mass_y, mass_z, Npos, Wp, solutions[0], constant, 0)

        # Best energy found
        qubo_energy = energies[0]
        list_energies[k] = qubo_energy

        # if valid solution
        if result==0:
            # if best energy found coincides with the actual payload of the assignment
            #if qubo_energy==Wa*energy_sol:
            if 1:
                print('\n Iteration {}\n'.format(k+1))
                print(solutions[0])
                print('payload     = {}'.format(Wa*energy_sol))
                print('qubo_energy = {}\n'.format(qubo_energy))

                print('------------------')
                print('Energy validation:')
                print('------------------')

                print('payload = {}'.format(Wa*energy_sol))

                cost_list = quboAllocation_validation(solutions[0], mass_x, mass_y, mass_z, Npos, 0)
                cost1 = np.sum(cost_list)*Weight_alloc
                print("cost_alloc = {}".format(cost1))

                cost2 = quboPayload_validation(solutions[0], mass_x, mass_y, mass_z, Npos, Wp, Npos, Weight_payload)
                print("payload_cost = {}".format(cost2))

                total_cost = Wa*energy_sol + cost1 + cost2
                print('Total cost {}'.format(total_cost))

                PlotBarContainers(solutions[0], Npos, mass_x, mass_y, mass_z)

                count_solutions = count_solutions + 1

    # Summary
    #
    print('count_solutions = {}'.format(count_solutions))
    print("--- %s seconds ---" % (time.time() - start_time))

    # Sort energies
    list_energies.sort()

    return list_energies







#  Solve the problem using simulated quantum annealing (Sqaod)
#
def main():
    # Parametros
    Npos = 10
    Wp = 260
    Len = 10
    We = 120
    S0 = 220
    xcgmin = -0.1*Len
    xcgmax = 0.2*Len
    xcge = -0.05*Len
    xcgt = 0.1*Len
    obj = 'min'
    sqaod_exec = 1


    # Path to instances
    #
    instance_name = "5-5-5"
    path_instances = "instancias/" + instance_name + "/"
    path_resultados = "resultados-qubo/" + instance_name + "/"



    for iter1 in range(1):
        # name of the instances
        filename = "i" + str(iter1+1) + "-" + instance_name
        print('Reading instance: {}\n'.format(filename))

        mass_x, mass_y, mass_z = crear_masas(path_instances+filename)

        print('Type 1 masses {}'.format(mass_x))
        print('Type 2 masses {}'.format(mass_y))
        print('Type 3 masses {}'.format(mass_z))

        print('\nParameters:')
        print('Npos = {}'.format(Npos))
        print('Wp   = {}'.format(Wp))
        print('Len  = {}'.format(Len))
        print('We   = {}'.format(We))
        print('S0   = {}'.format(S0))

        # Problem statement. -1 for minimization and +1 for maximization
        if obj=='min':
            Wa = -1
        else:
            Wa = +1


        print('\nConstructing qubo function Q:')

        Weight_alloc = 50
        Weight_payload = 1/100

        # -1 for maximization and, +1 for minimization
        if obj=='min':
            Weight_alloc = +1*Weight_alloc
            Weight_payload = +1*Weight_payload
        else:
            Weight_alloc = -1*Weight_alloc
            Weight_payload = -1*Weight_payload


        print('penalty_weights = ({}, {})'.format(Weight_alloc,Weight_payload))

        # Construye funcion QUBO
        Q, constant, offset = quboEtotal(mass_x, mass_y, mass_z, Npos, Wp,
                                        xcgmin, xcgmax, xcge, We, xcgt, Len,
                                        S0, Wa, Weight_alloc, Weight_payload)

        print('Dimension(Q)    = {}'.format(len(Q)+1))
        print('constant        = {}'.format(constant))


        # save to file qubo model
        #numvars = writequbo(Q, filename+'-allocation', constant)

        numvars = problem_size(Q)
        print('Num variables   = {}'.format(numvars))


        # Sampling instances
        #sampling(Npos, Wp, Len, We, S0, xcgmin, xcgmax, xcge, xcgt, numvars, constant, instance_name, 1)

        # CPLEX format
        writequbo_cplex(Q, filename+'-allocation', constant)

        if sqaod_exec==0:
            continue


        num_calls = 10

        list_energies = sqaod_caller(Q, mass_x, mass_y, mass_z, Npos, numvars, Wp, constant, Wa, Weight_alloc, Weight_payload, num_calls, obj)

        #list_energies = sq_caller(Q, mass_x, mass_y, mass_z, Npos, numvars, Wp, constant, Wa, Weight_alloc, Weight_payload, num_calls, obj)


        # - - - - - - - - - - - -
        #      Plot energies
        # - - - - - - - - - - - -
        plt.style.use('seaborn-whitegrid')
        plt.xlabel('Iteration calling sqaod')
        plt.ylabel('Energy')
        plt.title("min {obj1 + payload}")

        # plot energies
        last_n = len(list_energies)
        plt.scatter([i for i in range(last_n)], list_energies, marker='o', color='blue')

        # plot horizontal line indicating the maximum payload
        horiz_line_data = np.array([Wa*Wp for i in range(num_calls)])
        plt.plot([i for i in range(last_n)], horiz_line_data, 'r--')
        plt.tight_layout()
        plt.show()




if __name__ == "__main__":
    main()
