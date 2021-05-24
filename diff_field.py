import sys
import numpy as np # used for savetext
import math
import os


def readVTKUnstructuredGrid(file_name):
    vertices = []
    elements = []
    cell_types = []
    field = []

    fs = open(file_name)

    #readHeader(fs)
    line = fs.readline()
    line = line.strip().split()

    while len(line) == 0 or (line[0] != 'ASCII' and line[0] != 'BINARY'):
        line = fs.readline()
        line = line.strip().split()

    if line[0] == 'BINARY':
        print('BINARY VTK file not supported')
        sys.exit(1)

    line = fs.readline()
    line = line.strip().split()

    while len(line) == 0 or line[0] != 'POINTS':
        line = fs.readline()
        line = line.strip().split()

    num_vertices = int(line[1])

    #vertices = readVertices(fs)
    for i in range(num_vertices):
        line = fs.readline()
        line = line.strip().split()

        assert(len(line) == 3)

        x = float(line[0])
        y = float(line[1])
        z = float(line[2])
        vertices.append((x,y,z))

    line = fs.readline()
    line = line.strip().split()

    while len(line) == 0 or line[0] != 'CELLS':
        line = fs.readline()
        line = line.strip().split()

    num_cells = int(line[1])

    #elements = readElements(fs)
    for i in range(num_cells):
        line = fs.readline()
        line = line.strip().split()

        num_nodes = int(line[0])
        assert(len(line) == (num_nodes + 1))

        nodes = []
        for j in range(1+num_nodes):
            nodes.append(int(line[j]))

        elements.append(nodes)

    line = fs.readline()
    line = line.strip().split()

    while len(line) == 0 or line[0] != 'CELL_TYPES':
        line = fs.readline()
        line = line.strip().split()

    num_cell_types = int(line[1])

    #cell_types = readCellTypes(fs)
    for i in range(num_cell_types):
        line = fs.readline()
        line = line.strip().split()
        cell_types.append(int(line[0]))

    line = fs.readline()
    line = line.strip().split()

    while len(line) == 0 or line[0] != 'POINT_DATA':
        line = fs.readline()
        line = line.strip().split()
        
    num_field = int(line[1])

    # skip SCALARS ... and LOOKUP_TABLE ...
    line = fs.readline()
    line = line.strip().split()
    while len(line) == 0 or line[0] != 'SCALARS':
        line = fs.readline()
        line = line.strip().split()

    line = fs.readline()
    line = line.strip().split()
    while len(line) == 0 or line[0] != 'LOOKUP_TABLE':
        line = fs.readline()
        line = line.strip().split()

    #field = readFieldValues(fs)
    for i in range(num_field):
        line = fs.readline()
        line = line.strip().split()
        field.append(float(line[0]))


    fs.close()
    return (vertices, elements, cell_types, field)


def readHeader(fs):    
    pass


def readVertices(fs):
    pass


def readElements(fs):
    pass


def readCellTypes(fs):
    pass


def readFieldValues(fs):
    pass


def checkSimilar(data1, data2):
    # Perform some quick checks to verify that the data correspond
    # to similar data-sets

    (vertices1, elements1, cell_types1, field1) = data1
    (vertices2, elements2, cell_types2, field2) = data2

    # check vertices are same up to eps
    vert_ok = checkVertices(vertices1, vertices2)
    ele_ok = checkElements(elements1, elements2)
    types_ok = checkTypes(cell_types1, cell_types2)

    if not vert_ok:
        print('Vertices are different')
        sys.exit(1)

    if not ele_ok:
        print('Elements are different')
        sys.exit(1)

    if not types_ok:
        print('Cell types are different')
        sys.exit(1)


def checkVertices(vert1, vert2, eps=1e-16):
    if len(vert1) != len(vert2):
        return False

    for i in range(len(vert1)):
        v1 = vert1[i]
        v2 = vert2[i]

        d = dist(v1,v2)
        if d > eps:
            return False

    return True


def dist(v1, v2):
    x1,y1,z1 = v1
    x2,y2,z2 = v2

    return math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)


def checkElements(ele1, ele2):
    if len(ele1) != len(ele2):
        return False

    for i in range(len(ele1)):
        if len(ele1[i]) != len(ele2[i]):
            return False

        for j in range(len(ele1[i])):
            if ele1[i][j] != ele2[i][j]:
                return False

    return True


def checkTypes(cell_types1, cell_types2):
    if len(cell_types1) != len(cell_types2):
        return False

    for i in range(len(cell_types1)):
        if cell_types1[i] != cell_types2[i]:
            return False

    return True


# Compute the relative point-wise error:
# |field1 - field2| / |field1|
def computeRelativeAbsDiff(field1, field2):
    assert(len(field1)==len(field2))
    diff_field = []

    for i in range(len(field1)):
        v1 = field1[i]
        v2 = field2[i]

        err = abs(v1 - v2)
        
        eps = 1e-16
        if abs(v1) > eps:
            err = err / abs(v1)
        else:
            # prevent division by 0
            err = err / (abs(v1) + eps)

        #diff_field.append(abs(v1 - v2))
        diff_field.append(err)

    return diff_field


def computeAbsDiff(field1, field2):
    assert(len(field1)==len(field2))
    diff_field = []

    for i in range(len(field1)):
        v1 = field1[i]
        v2 = field2[i]

        diff_field.append(abs(v1 - v2))

    return diff_field


# L1 norm for vector
def L1VecNorm(field):
    l1_norm = 0.0
    for i in range(len(field)):
        l1_norm = l1_norm + abs(field[i])
    return l1_norm


# L2 norm for vector
def L2VecNorm(field):
    l2_norm = 0.0
    for i in range(len(field)):
        l2_norm = l2_norm + field[i]*field[i]

    l2_norm = math.sqrt(l2_norm)
    return l2_norm


# L^inf norm for vector
def LinfVecNorm(field):
    linf_norm = abs(field[0])
    for i in range(1, len(field)):
        linf_norm = max(linf_norm, abs(field[i]))

    return linf_norm


def compute_bounding_box(ps):
    # A bounding box is defined as a 2-tuple of points. A point is a 3-tuple.
    # The first 3-tuple is the lower corner of the box. The second 3-tuple
    # is the upper corner of the box. 
    max_flt = sys.float_info.max
    min_flt = -max_flt
    min_pt = [max_flt, max_flt, max_flt]
    max_pt = [min_flt, min_flt, min_flt]

    for p in ps:
        x,y,z = p
        min_pt = [min(x,min_pt[0]), min(y,min_pt[1]), min(z,min_pt[2])]
        max_pt = [max(x,max_pt[0]), max(y,max_pt[1]), max(z,max_pt[2])]
        
    bounding_box = (tuple(min_pt), tuple(max_pt))
    return bounding_box


# Compute |field1 - field2| / s
# where s is the size of the bounding box for the domain
# i.e. the length of this box diagonal.
def computeBBoxScaledAbsDiff(field1, field2, vertices):
    # Compute the bounding box and its diag length
    (minpt, maxpt) = compute_bounding_box(vertices)
    diag_len2 = 0.0
    assert(len(minpt)==len(maxpt))
    for i in range(len(minpt)):
        diag_len2 = diag_len2 + (maxpt[i]-minpt[i])**2
    diag_len = math.sqrt(diag_len2)

    # Compute |f1 - f2|
    diff_field = computeAbsDiff(field1, field2)

    # Scale each entry by the diag length
    for i in range(len(diff_field)):
        diff_field[i] = diff_field[i] / diag_len

    return diff_field


# Compute |field1 - field2| / s
# where s is the amplitude of field1
# i.e. |max(field1) - min(field1)|
def computeFieldScaledAbsDiff(field1, field2):
    # Compute |f1 - f2|
    diff_field = computeAbsDiff(field1, field2)

    min_field1 = min(field1)
    max_field1 = max(field1)
    scale = abs(max_field1 - min_field1)

    # Scale each entry by the diag length
    for i in range(len(diff_field)):
        diff_field[i] = diff_field[i] / scale

    return diff_field


def cross2D(u, v):
    # u x v where u and v are 2D vectors
    return u[0]*v[1] - v[0]*u[1]


def computeTriArea(v1coord, v2coord, v3coord):
    # assume triangles are in 2D with their z-coordinate being 0
    v1v2 = [v2coord[0] - v1coord[0], v2coord[1] - v1coord[1]]
    v1v3 = [v3coord[0] - v1coord[0], v3coord[1] - v1coord[1]]

    return 0.5 * cross2D(v1v2, v1v3)


def L1FunNorm(list_nodes, list_tris, field):
    num_tris = len(list_tris)
    integral = 0.0

    for i in range(num_tris):
        curr_tri = list_tris[i]
        # should be a triangle
        assert(len(curr_tri) == 3+1)
        # curr_tri[0] corresponds to the number of vertices
        v1 = curr_tri[1]
        v2 = curr_tri[2]
        v3 = curr_tri[3]

        v1_coord = list_nodes[v1]
        assert(len(v1_coord) == 2 or len(v1_coord) == 3)
        v2_coord = list_nodes[v2]
        assert(len(v2_coord) == 2 or len(v2_coord) == 3)
        v3_coord = list_nodes[v3]
        assert(len(v3_coord) == 2 or len(v3_coord) == 3)

        # assume v1, v2, v3 are all in the same (ccw) order
        signed_area = computeTriArea(v1_coord, v2_coord, v3_coord)
        # the area is signed, take the abs in case some triangles
        # are not ccw
        area = abs(signed_area)

        # approximate f at the centroid by 1/3(sum f(vi))
        field_bary = (1.0/3.0)*(field[v1]+field[v2]+field[v3])
        integral = integral + abs(field_bary) * area

    return integral


def L2FunNorm(list_nodes, list_tris, field):
    num_tris = len(list_tris)
    integral = 0.0

    for i in range(num_tris):
        curr_tri = list_tris[i]
        # should be a triangle
        assert(len(curr_tri) == 3+1)
        # curr_tri[0] corresponds to the number of vertices
        v1 = curr_tri[1]
        v2 = curr_tri[2]
        v3 = curr_tri[3]

        v1_coord = list_nodes[v1]
        assert(len(v1_coord) == 2 or len(v1_coord) == 3)
        v2_coord = list_nodes[v2]
        assert(len(v2_coord) == 2 or len(v2_coord) == 3)
        v3_coord = list_nodes[v3]
        assert(len(v3_coord) == 2 or len(v3_coord) == 3)

        # assume v1, v2, v3 are all in the same (ccw) order
        signed_area = computeTriArea(v1_coord, v2_coord, v3_coord)
        # the area is signed, take the abs in case some triangles
        # are not ccw
        area = abs(signed_area)

        # approximate f at the centroid by 1/3(sum f(vi))
        field_bary = (1.0/3.0)*(field[v1]+field[v2]+field[v3])
        integral = integral + (field_bary**2) * area

    return math.sqrt(integral)


# Same as computeLinfNorm() above
def LinfFunNorm(list_nodes, list_tris, field):
    return LinfVecNorm(field)


def writeVTKUnstructuredGrid(data, file_name):
    (vertices, elements, cell_types, field) = data
    number_nodes = len(vertices)
    number_elements = len(elements)
    element_order = elements[0][0]
    assert(element_order == 3 or element_order == 4)

    with open(file_name, 'w') as f:
        f.write("# vtk DataFile Version 2.0\n")
        f.write("Field\n")
        f.write("ASCII\n")
        f.write("\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        f.write("POINTS %d double\n" % (len(vertices)))

        # Write vertex coordinates
        np.savetxt(f, vertices, fmt="%f %f %f")

        cell_size = number_elements * (element_order + 1)
        f.write("\n")
        f.write("CELLS %d %d\n" % (number_elements, cell_size))
        for i in range(number_elements):
            f.write(" %d" % (element_order))
            for j in range(1, element_order+1):
                f.write(" %d" % (elements[i][j]))
            f.write("\n")

        f.write("\n")
        f.write("CELL_TYPES %d\n" % (number_elements))

        # Linear elements is cell type 5 (in 2D)
        # and cell type 10 (in 3D)
        if elements[0][0] == 3:
            for i in range(number_elements):
                f.write("5\n")
        elif elements[0][0] == 4:
            for i in range(number_elements):
                f.write("10\n")

        f.write("\n")
        f.write("POINT_DATA %d\n" % (number_nodes))
        f.write("SCALARS field double\n")
        f.write("LOOKUP_TABLE default\n")

        for i in range(number_nodes):
            f.write("%f\n" % (field[i]))

        f.write("\n")


# for compatibility, keep the same semantic
def printStatistics(err_field, exact, list_nodes, list_tris):
    (l1_err, l2_err, linf_err) = computeResidual(err_field, exact, list_nodes, list_tris)

    print("Residual (L1 norm): " + str(l1_err))
    print("Residual (L2 norm): " + str(l2_err))
    print("Residual (Linf norm): " + str(linf_err))


    (l1, l2, linf) = computeRelativeResidual(err_field, exact)

    print("Relative residual (L1 norm): " + str(l1))
    print("Relative residual (L2 norm): " + str(l2))
    print("Relative residual (Linf norm): " + str(linf))


def computeTotalArea(list_nodes, list_tris):
    num_tris = len(list_tris)
    total_area = 0.0

    for i in range(num_tris):
        curr_tri = list_tris[i]
        # should be a triangle
        assert(len(curr_tri) == 3+1)
        # curr_tri[0] corresponds to the number of vertices
        v1 = curr_tri[1]
        v2 = curr_tri[2]
        v3 = curr_tri[3]

        v1_coord = list_nodes[v1]
        assert(len(v1_coord) == 2 or len(v1_coord) == 3)
        v2_coord = list_nodes[v2]
        assert(len(v2_coord) == 2 or len(v2_coord) == 3)
        v3_coord = list_nodes[v3]
        assert(len(v3_coord) == 2 or len(v3_coord) == 3)

        # assume v1, v2, v3 are all in the same (ccw) order
        signed_area = computeTriArea(v1_coord, v2_coord, v3_coord)
        # the area is signed, take the abs in case some triangles
        # are not ccw
        area = abs(signed_area)

        total_area = total_area + area

    return total_area


def computeResidual(err_field, exact, list_nodes, list_tris):
    # assume that err_field = computeAbsDiff(field1, field2)

    # Compute L1 norm, L2 norm and L^inf norm for err_field
    l1_err = L1VecNorm(err_field)
    l2_err = L2VecNorm(err_field)
    linf_err = LinfVecNorm(err_field)

    total_area = computeTotalArea(list_nodes, list_tris)
    n = len(err_field)

    # cheap approximation of the integral
    l1_err = total_area / float(n) * l1_err
    l2_err = math.sqrt(total_area / float(n)) * l2_err

    return (l1_err, l2_err, linf_err)


def computeRelativeResidual(err_field, exact):
    # assume that err_field = computeAbsDiff(field1, field2)
    
    # Compute L1 norm, L2 norm and L^inf norm for err_field
    l1_err = L1VecNorm(err_field)
    l2_err = L2VecNorm(err_field)
    linf_err = LinfVecNorm(err_field)

    # Compute L1 norm, L2 norm and L^inf norm for exact
    l1_exact = L1VecNorm(exact)
    l2_exact = L2VecNorm(exact)
    linf_exact = LinfVecNorm(exact)

    # Compute relative residual
    l1 = l1_err / l1_exact
    l2 = l2_err / l2_exact
    linf = linf_err / linf_exact

    return (l1, l2, linf)


def computeResidualQuadrature(err_field, exact_field, list_nodes, list_tris):
    # Compute L1, L2 and L^inf norm for err_field using quadrature
    l1_err = L1FunNorm(list_nodes, list_tris, err_field)
    l2_err = L2FunNorm(list_nodes, list_tris, err_field)
    linf_err = LinfFunNorm(list_nodes, list_tris, err_field)

    return (l1_err, l2_err, linf_err)


def computeRelativeResidualQuadrature(err_field, exact_field, list_nodes, list_tris):
    # Compute L1, L2 and L^inf norm for err_field using quadrature
    l1_err, l2_err, linf_err = computeResidualQuadrature(err_field, exact_field, list_nodes, list_tris)
    
    # Compute L1, L2 and L^inf norm for the exact dist
    l1_exact = L1FunNorm(list_nodes, list_tris, exact_field)
    l2_exact = L2FunNorm(list_nodes, list_tris, exact_field)
    linf_exact = LinfFunNorm(list_nodes, list_tris, exact_field)

    # Compute relative residual
    l1 = l1_err / l1_exact
    l2 = l2_err / l2_exact
    linf = linf_err / linf_exact

    return (l1, l2, linf)


def printStatisticsQuadrature(err_field, exact_field, list_nodes, list_tris):
    (l1_err, l2_err, linf_err) = computeResidualQuadrature(err_field, exact_field, list_nodes, list_tris)

    print("Residual (L1 norm): " + str(l1_err))
    print("Residual (L2 norm): " + str(l2_err))
    print("Residual (Linf norm): " + str(linf_err))


    (l1, l2, linf) = computeRelativeResidualQuadrature(err_field, exact_field, list_nodes, list_tris)
    print("Relative residual (L1 norm): " + str(l1))
    print("Relative residual (L2 norm): " + str(l2))
    print("Relative residual (Linf norm): " + str(linf))


def checkUsage(args):
    prog_name = args[0]
    num_args = len(args)
    if (num_args != (1+2)) and (num_args != (1+3)):
        print('Usage:')
        print(args[0]+' exact.vtk GW_interpolation.vtk [output.vtk]')
        print('If output.vtk is ommitted, then only statistics')
        print('are computed and printed.')
        sys.exit(1)


def main(args):
    # check usage
    checkUsage(args)

    # Should we output anything
    save_to_file = False

    # as strings
    file1 = args[1]
    file2 = args[2]
    if len(args) == (1+3):
        out_file = args[3]
        save_to_file = True


    # read file 1
    (vertices1, elements1, cell_type1, field1) = readVTKUnstructuredGrid(file1)
    
    # read file 2
    (vertices2, elements2, cell_type2, field2) = readVTKUnstructuredGrid(file2)

    # perform some checking to verify they correspond to same data
    data1 = (vertices1, elements1, cell_type1, field1)
    data2 = (vertices2, elements2, cell_type2, field2)
    checkSimilar(data1, data2)

    # compute: |field1 - field2|
    diff_field = computeAbsDiff(field1, field2)
    
    # compute: |field1 - field2| / len of bbox diag
    bbox_scaled_diff_field = computeBBoxScaledAbsDiff(field1, field2, vertices1)

    # compute: |field1 - field2| / amplitude(field1)
    field_scaled_diff_field = computeFieldScaledAbsDiff(field1, field2)

    # compute: |field1 - field2| / |field1|
    relative_diff_field = computeRelativeAbsDiff(field1, field2)


    if save_to_file:
        # write diff field to file
        diff_data = (vertices1, elements1, cell_type1, diff_field)
        bbox_scaled_diff_data = (vertices1, elements1, cell_type1, bbox_scaled_diff_field)
        field_scaled_diff_data = (vertices1, elements1, cell_type1, field_scaled_diff_field)
        relative_diff_data = (vertices1, elements1, cell_type1, relative_diff_field)
    
        out_file_base, out_file_ext = os.path.splitext(out_file)
        out_file_unscaled = out_file
        out_file_bbox_scaled = out_file_base + '_bbox_scaled' + out_file_ext
        out_file_field_scaled = out_file_base + '_field_scaled' + out_file_ext
        out_file_relative = out_file_base + '_relative' + out_file_ext

        writeVTKUnstructuredGrid(diff_data, out_file_unscaled)
        writeVTKUnstructuredGrid(bbox_scaled_diff_data, out_file_bbox_scaled)
        writeVTKUnstructuredGrid(field_scaled_diff_data, out_file_field_scaled)
        writeVTKUnstructuredGrid(relative_diff_data, out_file_relative)


    # write some statistics on stdout
    printStatistics(diff_field, field1, vertices1, elements1)

    # Relative residual computed by numerical integration
    printStatisticsQuadrature(diff_field, field1, vertices1, elements1)


if __name__ == "__main__":
    main(sys.argv)
