"""Read recorded compact initial states; no generator or simulator imports."""
import hashlib,json
from pathlib import Path,PurePosixPath
SCHEMA='robotagentbench.compact_initial_states.v1'
STATE='robotagentbench.compact_initial_state.v1'
KINDS=frozenset(('short','composite','diagnostic','cloth'))

def _json(path):
    def unique(pairs):
        value={}
        for key,item in pairs:
            if key in value:raise ValueError('Duplicate compact catalogue key')
            value[key]=item
        return value
    return json.loads(Path(path).read_text(),object_pairs_hook=unique,
        parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite compact state')))

def checked(root,record):
    relative=record['path'];p=PurePosixPath(relative)
    if (not isinstance(relative,str)or not relative or p.is_absolute()or'..'in p.parts
            or p.as_posix()!=relative or'\\'in relative):raise ValueError('Canonical compact member path required')
    source=Path(root)/relative
    if source.resolve()!=source or not source.is_file()or source.stat().st_size!=record['bytes']:
        raise ValueError('Compact initial-state member absent, aliased or wrong size')
    with source.open('rb')as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if digest!=record['sha256']:raise ValueError('Compact initial-state checksum differs')
    return source

def read(path):
    path=Path(path).absolute()
    if path.resolve()!=path or not path.is_file():raise ValueError('Canonical compact catalogue required')
    value=_json(path)
    if value.get('schema')!=SCHEMA or value.get('geometry_included')is not False:
        raise ValueError('Recorded compact initial-state catalogue required')
    seen=set()
    for row in value['instances']:
        key=(row['task_id'],row['episode'])
        if (row['kind']not in KINDS or type(row['episode'])is not int or not 1<=row['episode']<=30
                or type(row['actual_seed'])is not int or row['actual_seed']<1 or key in seen):
            raise ValueError('Ambiguous compact task/episode identity')
        seen.add(key)
    return value

def select(path,task,episode):
    """Return trusted operator data; only state.public_input is model-visible."""
    path=Path(path).absolute();catalogue=read(path)
    rows=[r for r in catalogue['instances']if r['task_id']==task and r['episode']==episode]
    if len(rows)!=1:raise ValueError('Requested saved task/episode is absent')
    row=rows[0];state=_json(checked(path.parent,row['state']))
    if (state.get('schema')!=STATE or(state['task_id'],state['episode'],state['actual_seed'],state['kind'])
            !=(row['task_id'],row['episode'],row['actual_seed'],row['kind'])):
        raise ValueError('Compact initial-state identity differs')
    for attachment in row.get('attachments',[]):checked(path.parent,attachment)
    return dict(root=str(path.parent),catalogue=catalogue,entry=row,state=state)
