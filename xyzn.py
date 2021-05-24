import sys
import math
import numpy as np


def write_xyzn(filename, coords, normals):
    '''
    Write the samples coordinates (coords) and normals (normals) 
    to the file (given its name).
    
    xyzn Format is:
    x1 y1 z1 nx1 ny1 nz1
    x2 y2 z2 nx2 ny2 nz2
    ...
    xn yn zn nxn nyn nzn
    '''

    assert(len(coords) == len(normals))
    assert(len(coords[0]) == 3)
    assert(len(normals[0]) == 3)

    f = open(filename, 'w')

    l = len(coords)
    for i in range(l):
        x,y,z = coords[i]
        nx, ny, nz = normals[i]
        
        f.write(str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + 
                str(nx) + ' ' + str(ny) + ' ' + str(nz) + '\n')

    f.close

    
def read_xyzn(ps_filename):
    f = open(ps_filename)

    point_set = []
    for line in f:
        el = line.strip().split()
        if (len(el)==6):
            point = (float(el[0]), float(el[1]), float(el[2]),
                     float(el[3]), float(el[4]), float(el[5]))
        elif (len(el)==3):
            point = (float(el[0]), float(el[1]), float(el[2]))
        else:
            print('This line contains neither 3 nor 6 elements')
            continue
        point_set.append(point)

    f.close()
    return np.array(point_set)

