import sys
import math

import xyzn
import off


def compute_centroids(verts, tris):
    '''
    Return the list of centroids of the triangles in the triangle mesh.
    centroids[i] is the centroid of the triangle tris[i] (the i+1 th triangle).
    '''
    centroids = []
    for t in tris:
        v0,v1,v2 = t
        p0 = verts[v0]
        p1 = verts[v1]
        p2 = verts[v2]
        cx = 1.0/3.0 * (p0[0] + p1[0] + p2[0])
        cy = 1.0/3.0 * (p0[1] + p1[1] + p2[1])
        cz = 1.0/3.0 * (p0[2] + p1[2] + p2[2])
        centroids.append((cx,cy,cz))

    return centroids


def compute_face_normals(verts, tris):
    ''' 
    Compute the normal vector to each triangle in the triangle mesh.
    face_normals[i] is the vector normal to triangle tris[i].
    '''
    face_normals = []
    for t in tris:
        v0,v1,v2 = t
        p0 = verts[v0]
        p1 = verts[v1]
        p2 = verts[v2]

        # form the vectors p0p1, p0p2
        p0p1 = (p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2])
        p0p2 = (p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2])

        # compute the cross product
        n = compute_cross_product(p0p1, p0p2)

        # normalize it
        nu = normalize(n)

        face_normals.append(nu)


    return face_normals
    

def compute_cross_product(v0, v1):
    v0x,v0y,v0z = v0
    v1x,v1y,v1z = v1
    n = (-(v0z*v1y) + v0y*v1z,v0z*v1x - v0x*v1z,-(v0y*v1x) + v0x*v1y)
    return n


def compute_norm(v):
    return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)


def normalize(n):
    norm = compute_norm(n)
    if norm == 0.0:
        return n
    return (n[0]/norm, n[1]/norm, n[2]/norm)


def convert_off_to_xyzn(inname, outname):
    verts, tris = off.read_off(inname)
    # compute centroids and normals to triangles at centroids
    c = compute_centroids(verts, tris)
    n = compute_face_normals(verts, tris)

    # write the result as a xyzn file
    xyzn.write_xyzn(outname, c, n)


def usage(prog):
    print('Usage:')
    print(prog + ' in.off out.xyzn')


if __name__ == '__main__':
    args = sys.argv
    progname = args[0]
    args = args[1:len(args)] # keep only the arguments
    if len(args) != 2:
        usage(progname)
        sys.exit(1)
    
    print('converting from ply2 to xyzn')
    convert_off_to_xyzn(args[0], args[1])
  
