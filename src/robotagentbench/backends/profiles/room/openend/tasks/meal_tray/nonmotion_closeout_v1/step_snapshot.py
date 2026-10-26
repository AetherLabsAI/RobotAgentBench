"""Share read-only samples ONLY during one room after_physics invocation."""
import logging
import numpy as np

log = logging.getLogger(__name__)
IDENTITY_FIELDS = ('stage_id', 'actor0', 'actor1', 'collider0', 'collider1')


class ContactReportFilter:
    """Drop anonymous headers only; never invent contact or rewrite point data."""
    def __init__(self):
        self.reports = self.filtered_reports = self.ignored_headers = 0
        self.missing_fields = {name: 0 for name in IDENTITY_FIELDS}

    def normalize(self, report):
        if (not isinstance(report, tuple) or len(report) < 2
                or report[0] is None or report[1] is None):
            raise RuntimeError('Current PhysX contact report unavailable')
        headers, data = report[:2]
        valid = []
        ignored = 0
        fields = dict.fromkeys(IDENTITY_FIELDS, 0)
        for header in headers:
            missing = []
            for name in IDENTITY_FIELDS:
                value = getattr(header, name, None)
                # Invalid nonnumeric values remain errors, not ignored events.
                if value is None or int(value) <= 0:
                    missing.append(name)
            if missing:
                ignored += 1
                for name in missing:
                    fields[name] += 1
            else:
                valid.append(header)
        self.reports += 1
        if not ignored:
            return report
        self.filtered_reports += 1
        self.ignored_headers += ignored
        for name, count in fields.items():
            self.missing_fields[name] += count
        if self.filtered_reports == 1:
            log.warning('Ignoring %d anonymous contact headers; valid contacts remain unchanged. '
                        'Further events are counted in the episode contact filter.', ignored)
        # Keep original header identities/order, point buffer and offsets, and
        # extra fields. In particular, never compact data after dropping headers.
        return (valid, data, *report[2:])

    def summary(self):
        return dict(rule='ignore_anonymous_contact_event.v2', reports=self.reports,
                    filtered_reports=self.filtered_reports, ignored_headers=self.ignored_headers,
                    missing_fields=dict(self.missing_fields), physical_values_modified=False)

class SnapshotViews:
    def __init__(self,view):
        self.view=view;self.active=False;self.values={}
        self.reads=0;self.hits=0;self.steps=0
        self.contact_filter=ContactReportFilter()
    def begin(self):
        if self.active:raise RuntimeError('Nested snapshot window')
        self.values.clear();self.active=True;self.steps+=1
    def end(self):
        self.active=False;self.values.clear()
    def read(self,key,callback,array=False):
        self.reads+=1
        if not self.active:return callback()
        if key not in self.values:
            value=callback()
            self.values[key]=np.asarray(value).copy() if array else value
        else:self.hits+=1
        value=self.values[key]
        return value.copy() if array else value
    def create_rigid_body_view(self,paths):
        return SnapshotBody(self,self.view.create_rigid_body_view(paths))
    def read_contacts(self,callback):
        # Filter once per physical substep, before all task readers and routes.
        return self.read(('contacts',),lambda:self.contact_filter.normalize(callback()))
    def accept_contact_report(self,report):
        return self.read_contacts(lambda:report)
    def __getattr__(self,name):
        # Only view construction is forwarded, never simulator state setters.
        if name=='create_articulation_view':return getattr(self.view,name)
        raise AttributeError(name)

class SnapshotBody:
    def __init__(self,pool,view):
        self.pool=pool;self.view=view
        self.prim_paths=tuple(map(str,view.prim_paths));self.count=view.count
        if self.count!=len(self.prim_paths) or len(set(self.prim_paths))!=self.count:
            raise ValueError('Ambiguous native body view')
    def get_transforms(self):
        return self.pool.read(('transforms',self.prim_paths),self.view.get_transforms,True)
    def get_velocities(self):
        return self.pool.read(('velocities',self.prim_paths),self.view.get_velocities,True)
    def __getattr__(self,name):
        if name.startswith('get_'):return getattr(self.view,name)
        raise AttributeError(name)

class SnapshotContacts:
    def __init__(self,pool,interface):self.pool=pool;self.interface=interface
    def get_contact_report(self):
        # PhysX's room-global report is borrowed only until end of this step.
        # Readers keep their own contact lifecycle and may not mutate the report.
        return self.pool.read_contacts(self.interface.get_contact_report)
    def is_sleeping(self,*args):return self.interface.is_sleeping(*args)

def bind_contacts(ports,pool,routes=None):
    readers=[routes] if routes is not None else []
    for p in ports.values():
        readers.extend((p,getattr(p,'source',None),getattr(getattr(p,'owner',None),'source',None)))
    seen=set()
    for reader in readers:
        if reader is None or id(reader) in seen:continue
        seen.add(id(reader))
        interface=getattr(reader,'interface',None)
        if interface is not None:
            if isinstance(interface,SnapshotContacts):raise RuntimeError('Duplicate snapshot binding')
            reader.interface=SnapshotContacts(pool,interface)
