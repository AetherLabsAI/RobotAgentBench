"""Write a local App Server configuration for a trusted remote JSONL worker."""
import argparse,json
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--codex-executable',type=Path,required=True)
    p.add_argument('--cli-version',required=True)
    p.add_argument('--worker-argv-json',required=True,help='Trusted operator argv array, e.g. strict SSH wrapper plus an existing remote owner command')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--model',default='gpt-6-astra');p.add_argument('--effort',default='medium')
    p.add_argument('--max-model-responses',type=int,default=200);a=p.parse_args(argv)
    worker=json.loads(a.worker_argv_json)
    if not isinstance(worker,list)or not worker or any(not isinstance(x,str)or not x or '\x00'in x for x in worker):
        raise ValueError('Nonempty trusted argv array required, never a model-provided shell')
    cli=a.codex_executable.resolve(strict=True)
    output=a.output.resolve()
    if output.exists():raise ValueError('Fresh local agent output required')
    if not a.destination.parent.is_dir():raise ValueError('Existing configuration parent required')
    value=dict(schema='robotagentbench.codex_example.v1',codex_executable=str(cli),
        expected_cli_version=a.cli_version,model=a.model,effort=a.effort,
        max_model_responses=a.max_model_responses,transport=None,prepare_argv=[],
        worker_argv=worker,output=str(output))
    with a.destination.open('x')as stream:json.dump(value,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(operator=str(a.destination.resolve()),native_started=False,model_started=False)));return 0

if __name__=='__main__':raise SystemExit(main())
