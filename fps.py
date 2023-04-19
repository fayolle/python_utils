import numpy as np


# Farthest point sampling (FPS)
#
def farthest_point_sampling(pts, K):
    if K > 0:
        if pts.shape[0] < K:
            return pts
    else:
        return pts

    def calc_distances(p0, points):
        return ((p0[:3] - points[:, :3])**2).sum(axis=1)

    farthest_pts = np.zeros((K, pts.shape[1]))
    farthest_pts[0] = pts[np.random.randint(len(pts))]
    distances = calc_distances(farthest_pts[0], pts)

    for i in range(1, K):
        farthest_pts[i] = pts[np.argmax(distances)]
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))

    return farthest_pts
