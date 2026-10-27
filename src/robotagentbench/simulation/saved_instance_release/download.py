"""Download and safely install a pinned saved-instance archive collection."""
import argparse,json
from pathlib import Path
from robotagentbench import resources

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--offline',action='store_true',help='Use previously downloaded verified archives')
    args=parser.parse_args(argv)
    manifest=resources.read_manifest(args.manifest)
    if manifest.get('index')!='release.json':
        raise ValueError('A pinned saved-instance release manifest is required')
    if not args.offline:resources.fetch(manifest,args.cache)
    receipt=resources.install(manifest,args.cache,args.destination)
    print(json.dumps(receipt,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
