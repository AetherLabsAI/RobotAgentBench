"""Bounded bidirectional joint-space planning with checked, dense edges."""
import time
import numpy as np


def plan_joint_path(start, goal, limits, valid, rng, *, iterations=700,
                    seconds=35., step=.22, resolution=.025):
    start, goal, limits = np.asarray(start), np.asarray(goal), np.asarray(limits)
    if not valid(start) or not valid(goal):
        return None

    def edge(a, b):
        count = max(2, int(np.ceil(np.max(abs(b-a))/resolution))+1)
        return all(valid(a+t*(b-a)) for t in np.linspace(0, 1, count))

    if edge(start, goal):
        return [start.copy(), goal.copy()]
    trees = [([start.copy()], [-1]), ([goal.copy()], [-1])]
    reverse = False
    deadline = time.monotonic()+seconds

    def extend(tree, target):
        nodes, parents = tree
        index = int(np.argmin(np.linalg.norm(np.asarray(nodes)-target, axis=1)))
        old = nodes[index]
        delta = target-old
        scale = max(1., float(np.max(abs(delta)))/step)
        new = old+delta/scale
        if not edge(old, new):
            return None
        nodes.append(new)
        parents.append(index)
        return len(nodes)-1

    def chain(tree, index):
        result = []
        while index >= 0:
            result.append(tree[0][index])
            index = tree[1][index]
        return result[::-1]

    for iteration in range(iterations):
        if time.monotonic() >= deadline:
            break
        a, b = trees
        if iteration % 5 == 0:
            sample = b[0][0]
        elif iteration % 2:
            weight = rng.uniform()
            sample = np.clip(start+weight*(goal-start)+rng.normal(0, .8, len(start)),
                             limits[:, 0], limits[:, 1])
        else:
            sample = rng.uniform(limits[:, 0], limits[:, 1])
        i = extend(a, sample)
        if i is not None:
            for _ in range(40):
                j = extend(b, a[0][i])
                if j is None:
                    break
                if np.max(abs(b[0][j]-a[0][i])) < 1e-8:
                    path = chain(a, i)+chain(b, j)[-2::-1]
                    if reverse:
                        path.reverse()
                    # Shortcuts retain the same dense collision check.
                    for _ in range(40):
                        if len(path) < 3:
                            break
                        left, right = sorted(rng.choice(len(path), 2, replace=False))
                        if right > left+1 and edge(path[left], path[right]):
                            path = path[:left+1]+path[right:]
                    return path
        trees.reverse()
        reverse = not reverse
    return None
