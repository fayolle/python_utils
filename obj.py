import numpy as np


def read_obj(filename):
    V = []  # vertex
    F = []  # face indexies

    fh = open(filename)
    for line in fh:
        if line[0] == '#':
            continue

        line = line.strip().split(' ')
        if line[0] == 'v':  # vertex
            V.append([float(line[i + 1]) for i in range(3)])
        elif line[0] == 'f':  # face
            face = line[1:]
            for i in range(0, len(face)):
                face[i] = int(face[i].split('/')[0]) - 1
            F.append(face)

    V = np.array(V)
    F = np.array(F)

    return V, F


def write_obj(filename, V, F):
    output = ""

    for i in range(V.size(0)):
        if V[i].abs().sum() > 0:
            output += "v {} {} {}\n".format(V[i][0], V[i][1], V[i][2])

    for i in range(F.size(0)):
        if F[i].abs().sum() > 0:
            output += "f {} {} {}\n".format(F[i][0] + 1, F[i][1] + 1,
                                            F[i][2] + 1)

    text_file = open(filename, "w")
    text_file.write(output)
    text_file.close()
