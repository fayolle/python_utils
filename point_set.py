# Some utils for manipulating finite point-sets.

import sys

# Data-Structure:
# A point-set is simply a list of 3-tuple. Each 3-tuple represents a point.


def compute_bounding_box(ps):
    # A bounding box is defined as a 2-tuple of points. A point is a 3-tuple.
    # The first 3-tuple is the lower corner of the box. The second 3-tuple
    # is the upper corner of the box.
    max_flt = sys.float_info.max
    min_flt = -max_flt
    min_pt = [max_flt, max_flt, max_flt]
    max_pt = [min_flt, min_flt, min_flt]

    for p in ps:
        x, y, z = p
        min_pt = [min(x, min_pt[0]), min(y, min_pt[1]), min(z, min_pt[2])]
        max_pt = [max(x, max_pt[0]), max(y, max_pt[1]), max(z, max_pt[2])]

    bounding_box = (tuple(min_pt), tuple(max_pt))
    return bounding_box
