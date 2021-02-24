import math
import numpy as np
import sys
import itertools
import random
import os
#import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches
from random import randint
from random import gauss
import pickle
import blueqat.wq as bwq

from dwave_qbsolv import QBSolv
import neal
from tabu import TabuSampler




# - - - - - - - - - - - - - - - - - - #
#     Inicio de las definiciones      #
# - - - - - - - - - - - - - - - - - - #


# Convert a QUBO dictionary to a matrix
#
def dict_to_array(Q, numvars):
    qarray = np.zeros((numvars, numvars))

    for key in Q:
        qarray[key[0]][key[1]] = Q[key]

    return qarray



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

    print('\n############\nReading instance: {}\n'.format(archivo))
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
        return i + 2*j + (n1 + n2 + 2*n3)*Npos
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
                print('Invalid solution. Does not meet the equation 2a.')

    # Restriction eq. 2b
    for i in range(n1):
        t = 0
        for j in range(Npos):
            key = ix(i,j,Npos)
            t += solution[key]
        if t > 1:
            result = 1
            if verbose==1:
                print('Invalid solution. Does not meet the equation 2b.')

    # Restriction eq. 2c
    for i in range(n2):
        t = 0
        for j in range(Npos):
            key = iy(i,j,n1,Npos)
            t += solution[key]
        if t > 1:
            result = 1
            if verbose==1:
                print('Invalid solution. Does not meet the equation 2c.')


    # Restriction eq. 2d
    for i in range(n3):
        t = 0
        for j in range(Npos):
            key = iz(i,j,0,n1,n2,n3,Npos)
            t += solution[key]
        if t > 1:
            result = 1
            if verbose==1:
                print('Invalid solution. Does not meet the equation 2d.')

    # Restriction eq. 2e
    for i in range(n3):
        t = 0
        for j in range(Npos):
            key = iz(i,j,1,n1,n2,n3,Npos)
            t += solution[key]
        if t > 1:
            result = 1
            if verbose==1:
                print('Invalid solution. Does not meet the equation 2e.')


    # Restriction eq. 2f
    for i in range(n3):
        for j in range(Npos-1):
            key1 = iz(i, j, 0, n1, n2, n3, Npos)
            key2 = iz(i, j+1, 1, n1, n2, n3, Npos)
            if solution[key1] != solution[key2]:
                result = 1
                if verbose==1:
                    print('Invalid solution. Does not meet the equation 2f.')

    # Restrictions eqs. 2g and 2h
    for i in range(n3):
        key1 = iz(i, Npos-1, 0, n1, n2, n3, Npos)
        key2 = iz(i, 0, 1, n1, n2, n3, Npos)
        if solution[key1] != 0:
            result = 1
            if verbose==1:
                print('Invalid solution. Does not meet the equation 2g.')
        if solution[key2] != 0:
            result = 1
            if verbose==1:
                print('Invalid solution. Does not meet the equation 2h.')


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
            print('Invalid solution. Does not meet the equation 4. The maximum payload is exceeded.')

    if verbose==1:
        print('Payload of the solution: {}\n'.format(-obj1))

    return result




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
        list_slack = [islack(i,j,n1,n2,n3,Npos,0, offset) for i in range(2)]

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

        # + 4 A1 ep1 + 8 A1 ep2
        for x in list_x:
            key = ipair(x, list_slack[0])
            Q = updateKey(Q, key, 4)
            key = ipair(x, list_slack[1])
            Q = updateKey(Q, key, 8)

        # + 2 A2 ep1 + 4 A2 ep2
        for y in list_y:
            key = ipair(y, list_slack[0])
            Q = updateKey(Q, key, 2)
            key = ipair(y, list_slack[1])
            Q = updateKey(Q, key, 4)


        # + 4 A3 ep1 + 8 A3 ep2
        for z in list_z:
            key = ipair(z, list_slack[0])
            Q = updateKey(Q, key, 4)
            key = ipair(z, list_slack[1])
            Q = updateKey(Q, key, 8)

        # -3 ep1 - 4 ep2 + 4 ep1 ep2
        key = (list_slack[0], list_slack[0])
        Q = updateKey(Q, key, -3)
        key = (list_slack[1], list_slack[1])
        Q = updateKey(Q, key, -4)
        key = ipair(list_slack[0], list_slack[1])
        Q = updateKey(Q, key, 4)

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

    offset1 = offset + 2*Npos

    return Q, constant, offset1






# Construccion de la restriccion de maxima carga Ec. 28
#
def quboPayload(Q1, mass_x, mass_y, mass_z, Npos, Wp, offset, const):
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
    print('\nNumber of bits {} to represet Wp={}'.format(nbits, Wp))


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


    # Termino -2*Wp*F
    for i in range(len_list):
        k = list_var[i]
        Q = updateKey(Q, (k,k), -2*list_mass[i]*Wp)


    # Termino -2*Wp*Sw
    for i in range(nbits):
        k = list_slack[i]
        Q = updateKey(Q, (k,k), -2*coeff_slack[i]*Wp)


    # El termino constante se conserva  Wp**2 en la variable 'constant'
    constant += Wp**2
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
def quboEtotal(mass_x, mass_y, mass_z, Npos, Wp, xcgmin, xcgmax, xcge, We, xcgt, Len, S0, Wa, Weight):

    # Definicion del diccionario Q.
    Q = {}

    print('\nConstructing qubo function ...')

    constant = 0;
    offset = 0

    # Eq. 27
    Q, constant, offset = quboAllocation(Q, mass_x, mass_y, mass_z, Npos, offset, constant)

    # Eq. 28
    #Q, constant, offset = quboPayload(Q, mass_x, mass_y, mass_z, Npos, Wp, offset, constant)

    # Eq. 30
    #Q, constant, offset = quboGravityLeft(Q, mass_x, mass_y, mass_z, Npos, xcgmin, xcge, We, Len, offset, constant)

    # Eq. 31
    #Q, constant, offset = quboGravityRight(Q, mass_x, mass_y, mass_z, Npos, xcgmax, xcge, We, Len, offset, constant)

    # Eq. 32
    #Q, constant, offset = quboLeftShear(Q, mass_x, mass_y, mass_z, Npos, S0, Len, offset, constant)

    # Eq. 33
    #Q, constant, offset = quboRightShear(Q, mass_x, mass_y, mass_z, Npos, S0, Len, offset, constant)

    # Scale penalty terms
    Q, constant = quboScale(Q, Weight, constant)

    # Eq. 3
    Q, constant, offset = quboFunction1(Q, mass_x, mass_y, mass_z, Npos, offset, constant, Wa)

    # Eq. 15
    #Q, constant, offset = quboFunction2(Q, mass_x, mass_y, mass_z, Npos, xcgt, Wp, xcge, Len, offset, constant)


    # Limpia diccionario
    Q = RemoveZeroKeys(Q)

    return Q, constant, offset







# - - - - - - - - - - - - - - - - - - #
#     Fin de las definiciones         #
# - - - - - - - - - - - - - - - - - - #


def main():

    # Parametros
    Npos = 15
    Wp = 255
    Len = 15
    We = 120
    S0 = 220
    xcgmin = -0.1*Len
    xcgmax = 0.2*Len
    xcge = -0.05*Len
    xcgt = 0.1*Len

    obj = 'min'
    dwave_exec = 1


    # Path to instances
    #
    path_resultados = "resultados-qubo/5-5-5/"
    path_instances = "instancias/5-5-5/"

    for iter1 in range(1):
        # name of the instances
        filename = "i" + str(iter1+1) + "-5-5-5"

        mass_x, mass_y, mass_z = crear_masas(path_instances+filename)

        print('Type 1 masses {}'.format(mass_x))
        print('Type 2 masses {}'.format(mass_y))
        print('Type 3 masses {}'.format(mass_z))

        print('\nMaximum weight Wp={}'.format(Wp))


        # Weight for the objective function. -1 for minimization and +1 for maximization
        if obj=='min':
            Wa = -1
            find_max_val = False
        else:
            Wa = +1
            find_max_val = True

        # Upper bound for the penalty weight
        Weight = (sum(mass_x) + sum(mass_y) + sum(mass_z))*Npos


        # -1 for maximization and, +1 for minimization
        if obj=='min':
            Weight = +1*Weight
        else:
            Weight = -1*Weight

        print('\nPenalty Weight {}'.format(Weight))



        # Construye funcion QUBO
        Q, constant, offset = quboEtotal(mass_x, mass_y, mass_z, Npos, Wp, xcgmin, xcgmax, xcge, We, xcgt, Len, S0, Wa, Weight)

        print('\nNumber of terms in Q: {}'.format(len(Q)+1))
        print('constant = {}'.format(constant))
        #print('{}'.format(Q))




        # save to file qubo model
        numvars = writequbo(Q, filename+'-allocation', constant)

        print('\nNumber of variables: {}'.format(numvars))


        # blueqat
        #blueqat_problem = bwq.Opt()
        #qubo_matrix = dict_to_array(Q, numvars)
        #blueqat_problem.qubo = qubo_matrix
        #print('\n\n{}'.format(qubo_matrix))
        #answer = blueqat_problem.sqa()

        #for sol in answer:
        #    print('{} '.format(sol))




        # CPLEX format
        writequbo_cplex(Q, filename+'-allocation', constant)

        if dwave_exec==0:
            continue

        # D-Wave tool call
        #
        print('qbsolv sampler calling ...')

        # Number of calls to SimulatedAnnealingSampler
        num_calls = 1
        # Size of a sub-problem
        subqubo_size=30

        # to save the best solution per call
        all_solutions = {}
        best_solution = {}
        cnt = 0


        for k in range(num_calls):
            print('#{} '.format(k+1))
            sampler = neal.SimulatedAnnealingSampler()
            response = QBSolv().sample_qubo(Q, num_repeats=1000, solver=sampler, find_max=find_max_val, solver_limit=subqubo_size)
            energies = response.data_vectors['energy'] + constant
            solutions = list(response.samples())
            #print('{}'.format(energies))

            # save all solutions and energies for analysis
            all_solutions[k] = (energies, solutions)

            for j in range(5):
                best_solution[cnt] = (energies[j], solutions[j])
                cnt=cnt+1



        # Plot result
        #
        list_energies = [0 for i in range(cnt)]

        if obj=='min':
            best_energy = 10000
        else:
            best_energy = -10000

        k= 0

        for pair in best_solution:
            (energy, sol) = best_solution[pair]
            list_energies[k] = energy
            k = k + 1



        # Sort energies
        #
        list_energies.sort()
        #print('\nBest energies: {}'.format(list_energies))


        # pickle dump
        pfile = open(path_resultados+filename+"-result", 'wb')
        pickle.dump(all_solutions, pfile)
        pfile.close()

        pfile = open(path_resultados+filename+"-energies", 'wb')
        pickle.dump(list_energies, pfile)
        pfile.close()




if __name__ == "__main__":
    main()
