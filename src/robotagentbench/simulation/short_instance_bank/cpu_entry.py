"""Set the existing CPU USD loader environment before any shared-library import."""
import argparse
import os
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk-site',type=Path,required=True)
    args,remaining=p.parse_known_args()
    from robotagentbench.example import sdk_paths,runtime
    site,usd,physx=sdk_paths(args.sdk_site)
    public=Path(__file__).resolve().parents[3]
    marker='ROBOTAGENTBENCH_SHORT_BANK_CPU_LOADER'
    if os.environ.get(marker)!=str(site):
        env=dict(os.environ)
        for key in('PYTHONPATH','PYTHONHOME','LD_PRELOAD','LD_AUDIT'):env.pop(key,None)
        env.update({marker:str(site),'PYTHONDONTWRITEBYTECODE':'1',
            'LD_LIBRARY_PATH':str(usd/'bin')+os.pathsep+str(physx/'bin'),
            'PXR_AR_DEFAULT_SEARCH_PATH':str(site/'isaacsim/kit/mdl/core/Base')})
        bootstrap=('import runpy,sys;sys.path.insert(0,'+repr(str(public))+');'
            'sys.argv=["short-bank-cpu",*sys.argv[1:]];'
            'runpy.run_module("robotagentbench.simulation.short_instance_bank.cpu_entry",run_name="__main__")')
        os.execve(sys.executable,[sys.executable,'-I','-S','-B','-c',bootstrap,*sys.argv[1:]],env)
        raise RuntimeError('CPU loader handoff returned')
    if sys.version_info[:2]!=(3,11):raise ValueError('Existing CPU USD profile requires Python3.11')
    paths=[str(usd),str(physx),str(site)]
    sys.path[:0]=paths
    if '--runtime'in remaining:
        _,root,_=runtime(remaining[remaining.index('--runtime')+1])
        os.environ['PYTHONPATH']=os.pathsep.join(paths+[str(root)])
    from .bank import main as bank_main
    return bank_main(remaining)


if __name__=='__main__':raise SystemExit(main())
