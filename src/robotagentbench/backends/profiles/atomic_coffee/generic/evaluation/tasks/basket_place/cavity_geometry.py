"""Read-only live open-container geometry; no physics/scoring thresholds.

Close the *measurement surface* at its single boundary to classify witness
points using oriented solid angles. This cap is never authored into USD or
physics. Witness containment alone is NOT whole-object containment/support.
"""
import numpy as np


class OpenCavity:
    def __init__(self, faces, vertex_count):
        faces = np.asarray(faces)
        if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
                or not np.issubdtype(faces.dtype, np.integer)
                or faces.min() < 0 or faces.max() >= vertex_count):
            raise ValueError('Expected valid triangle topology')
        self.faces = faces.astype(np.int64, copy=True)
        self.vertex_count = vertex_count
        edges = {}
        for a, b, c in self.faces:
            if len({a, b, c}) != 3:
                raise ValueError('Degenerate index triangle')
            for u, v in ((a,b),(b,c),(c,a)):
                edges.setdefault(tuple(sorted((u,v))), []).append((u,v))
        boundary = []
        for pair in edges.values():
            if len(pair) == 1:
                boundary.append(pair[0])
            elif len(pair) != 2 or pair[0] != pair[1][::-1]:
                raise ValueError('Non-manifold or inconsistently oriented mesh')
        if not boundary:
            raise ValueError('Expected an open basket, not a closed solid')
        outgoing = dict(boundary)
        if (len(outgoing) != len(boundary)
                or len(set(outgoing.values())) != len(boundary)):
            raise ValueError('Invalid basket boundary')
        start = boundary[0][0]
        loop = [start]
        while True:
            end = outgoing.get(loop[-1])
            if end == start:
                break
            if end is None or end in loop or len(loop) >= len(boundary):
                raise ValueError('Unclosed basket boundary')
            loop.append(end)
        if len(loop) != len(boundary):
            raise ValueError('Multiple openings/holes need an explicit geometry audit')
        self.boundary = np.asarray(loop, dtype=np.int64)
        self.cap = np.asarray([(v,u,vertex_count) for u,v in boundary], dtype=np.int64)

    def inspect(self, vertices, witnesses):
        points = np.asarray(vertices, dtype=float)
        query = np.asarray(witnesses, dtype=float)
        if (points.shape != (self.vertex_count,3) or query.ndim != 2
                or query.shape[1] != 3 or not np.isfinite(points).all()
                or not np.isfinite(query).all()):
            raise ValueError('Missing/nonfinite current geometry')
        opening = points[self.boundary]
        cap_centre = opening.mean(0)
        augmented = np.vstack((points, cap_centre))
        triangles = augmented[np.vstack((self.faces,self.cap))]
        if np.any(np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],
                    triangles[:,2]-triangles[:,0]),axis=1) == 0):
            raise ValueError('Collapsed current mesh or measurement cap')
        # Star cap follows the current rim including non-planar deformation.
        # Self-intersection is not certified by this diagnostic, so do not
        # expose a complete-object placement predicate from these witnesses.
        values=[]
        for q in query:
            a,b,c = (triangles-q).transpose(1,0,2)
            la,lb,lc = (np.linalg.norm(v,axis=1) for v in (a,b,c))
            if min(la.min(),lb.min(),lc.min()) == 0:
                values.append(None); continue  # Boundary point is ambiguous.
            numerator=np.einsum('ij,ij->i',a,np.cross(b,c))
            denominator=(la*lb*lc + np.einsum('ij,ij->i',a,b)*lc
                + np.einsum('ij,ij->i',b,c)*la + np.einsum('ij,ij->i',c,a)*lb)
            values.append(float(np.arctan2(numerator,denominator).sum()/(2*np.pi)))
        return dict(winding_numbers=values,
            witness_inside=[None if x is None else abs(x) > .5 for x in values],
            opening_vertices=len(self.boundary), opening_min=opening.min(0).tolist(),
            opening_max=opening.max(0).tolist(),
            measurement_cap_only=True, whole_object_containment_verified=False,
            support_verified=False)
