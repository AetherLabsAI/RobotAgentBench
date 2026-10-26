"""Opt-in read-only diagnostic. Raw report impulses are not motor forces.

This deliberately preserves collider ordering. Report impulses may exclude
friction; do not label their sum a complete contact wrench without validating
the backend's reporting semantics.
"""
import json
import numpy as np


def vector(v):
    return [float(v.x), float(v.y), float(v.z)]


class ForceAudit:
    def __init__(self, output, views, dt):
        self.file = (output / 'force_audit.jsonl').open('x')
        meta = dict(physics_dt_s=dt, physics_modified=False, bodies={},
                    warning='Raw collider-ordered contact impulses; friction completeness not assumed.')
        for role, view in views.items():
            com, orientation = view.get_com()
            meta['bodies'][role] = dict(mass_kg=float(view.get_mass()),
                local_com_m=np.asarray(com).tolist(),
                com_orientation_wxyz=np.asarray(orientation).tolist(),
                inertia_tensor_kg_m2=np.asarray(view.get_inertia_tensor()).reshape(3,3).tolist())
        (output / 'force_audit_metadata.json').write_text(json.dumps(meta, indent=2))

    def record(self, report, stream, views, poses, step, time_s, phase):
        headers, data = report[:2]
        pairs = []
        for h in headers:
            paths, _, relevant = stream.identity.get(h.stage_id, h.collider0, h.collider1)
            if not relevant:
                continue
            points = []
            offset, count = int(h.contact_data_offset), int(h.num_contact_data)
            if h.type != stream.lost and count:
                for i in range(offset, offset + count):
                    p = data[i]
                    points.append(dict(position_m=vector(p.position),
                        normal=vector(p.normal), impulse_Ns=vector(p.impulse),
                        separation_m=float(p.separation) if hasattr(p, 'separation') else None))
            friction=[]
            if len(report)>2 and h.type != stream.lost:
                start=int(h.friction_anchors_offset)
                count=int(h.num_friction_anchors_data)
                for i in range(start,start+count):
                    anchor=report[2][i]
                    friction.append(dict(position_m=vector(anchor.position),
                                         impulse_Ns=vector(anchor.impulse)))
            pairs.append(dict(colliders=[str(p) for p in paths],
                              event=int(h.type), points=points, friction_anchors=friction))
        bodies = {role: dict(**poses[role],
            linear_velocity_m_s=np.asarray(view.get_linear_velocity()).tolist(),
            angular_velocity_rad_s=np.asarray(view.get_angular_velocity()).tolist())
            for role, view in views.items()}
        self.file.write(json.dumps(dict(reader_step=step, time_s=time_s,
            phase=phase, bodies=bodies, pairs=pairs,
            full_report=len(report)>2), allow_nan=False) + '\n')
        self.file.flush()

    def close(self):
        self.file.close()
