"""
#  constrain
cont - bin * min <= 0
"""

import collections

import pandas as pd

data = collections.defaultdict(list)
data["cont"] = [100, 50, 30, 100, 50, 30, 0, 0, 0]
data["bin"] = [1, 1, 1, 0, 0, 0, 0, 0, 0]


def constraint_max_value(c, b, max_value=50):
    return c - b * max_value


c < max_value


def constraint_min_value(c, b, min_value=50):
    return -c + b * min_value


"""


"""
out = []
for c, b in zip(data["cont"], data["bin"]):
    cstr = constraint_max_value(c, b)
    data["constraint-max"].append(cstr)
    data["violated-max"].append(cstr > 0)

    cstr = constraint_min_value(c, b)
    data["constraint-min"].append(cstr)
    data["violated-min"].append(cstr > 0)


# add_constraint(bin_import + bin_export == 1)
print(pd.DataFrame(data))
