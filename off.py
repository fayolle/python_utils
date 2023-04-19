import sys
import math
import numpy as np


def read_off(filename):
    f = open(filename)

    off = f.readline()
    off = off.strip()
    assert (off.lower() == 'off')

    # number of vertices, triangles, edges
    nums = f.readline()
    nums = nums.strip()
    nums = nums.split()
    assert (len(nums) == 3)
    num_verts = int(nums[0])
    num_faces = int(nums[1])
    num_edges = int(nums[2])

    # list of vertices
    verts = []
    for i in range(num_verts):
        line = f.readline()
        line = line.strip()
        line = line.split()

        # skip empty lines
        while len(line) == 0:
            line = f.readline()
            line = line.strip()
            line = line.split()

        # TODO raise an error if there are no 3 elements
        verts.append((float(line[0]), float(line[1]), float(line[2])))

    # list of polygons
    tris = []
    for i in range(num_faces):
        line = f.readline()
        line = line.strip()
        line = line.split()
        num_vert = int(line[0])
        if num_vert != 3:
            raise Exception("Only OFF with triangles are handled")

        tris.append((int(line[1]), int(line[2]), int(line[3])))

    f.close()
    return (np.array(verts), np.array(tris))
