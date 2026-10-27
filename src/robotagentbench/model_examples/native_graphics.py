"""Original NVIDIA ICD configuration, not a driver or simulator installer.

The original142-byte file is retained verbatim. prepare() writes only a fresh
operator-owned directory. No SDK, device, Vulkan API or model is imported/probed.
The noncanonical container target avoids runtime replacement of /etc's ICD.
"""
import argparse
import hashlib
import json
from pathlib import Path

SCHEMA='robotagentbench.native_graphics_binding.v1'
TARGET='/task_graphics/nvidia_original_icd.json'
SHA256='987f809c9289ab1a4b95b900991e8826fe0af37c7a88c42e3ff047d7f8d22d6b'
ICD_BYTES=(b'{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n'
           b'        "library_path" : "libGLX_nvidia.so.0",\n'
           b'        "api_version" : "1.3.194"\n    }\n}\n')
SOURCE=dict(image='sha256:b272c84fbf8a4b7ec921a6fd680bc1bc1f3c1d5e153b653d394a9d6cc07912b6',
    path='/etc/vulkan/icd.d/nvidia_icd.json',sha256=SHA256,
    description='Original image configuration bytes; no driver binaries or new Vulkan API version bundled')
CAPABILITIES=frozenset(('compute','graphics','utility'))


def describe():
    return dict(schema=SCHEMA,source=dict(SOURCE),bytes=len(ICD_BYTES),target=TARGET,
        runtime='nvidia',environment=dict(VK_DRIVER_FILES=TARGET),
        required_driver_library='libGLX_nvidia.so.0',driver_binary_bundled=False,
        sdk_imported=False,gpu_probed=False)


def validate(value):
    if (not isinstance(value,dict)or set(value)!={'schema','source','sha256','target','runtime','environment'}
        or value['schema']!=SCHEMA or value['sha256']!=SHA256 or value['target']!=TARGET
        or value['runtime']!='nvidia' or value['environment']!={'VK_DRIVER_FILES':TARGET}):
        raise ValueError('Exact original ICD, explicit noncanonical read-only alias and NVIDIA runtime required')
    source=Path(value['source'])
    if not source.is_absolute()or source.resolve(strict=True)!=source or not source.is_file():
        raise ValueError('Existing canonical operator-owned original ICD required')
    if source.read_bytes()!=ICD_BYTES or hashlib.sha256(source.read_bytes()).hexdigest()!=SHA256:
        raise ValueError('Original NVIDIA ICD configuration changed')
    return source


def prepare(destination):
    """Write exact small configuration files only; never touch shared SDK paths."""
    out=Path(destination)
    if (not out.is_absolute()or out.resolve()!=out or out.exists()or not out.parent.is_dir()
        or out==Path(out.anchor)or out==Path.home()):
        raise ValueError('Fresh canonical owned configuration directory required')
    if hashlib.sha256(ICD_BYTES).hexdigest()!=SHA256:raise RuntimeError('Bundled original ICD seal differs')
    out.mkdir();source=out/'nvidia_original_icd.json'
    with source.open('xb')as stream:stream.write(ICD_BYTES)
    source.chmod(0o440)
    binding=dict(schema=SCHEMA,source=str(source),sha256=SHA256,target=TARGET,
        runtime='nvidia',environment=dict(VK_DRIVER_FILES=TARGET))
    validate(binding)
    with(out/'binding.json').open('x')as stream:json.dump(binding,stream,indent=2);stream.write('\n')
    with(out/'source.json').open('x')as stream:json.dump(describe(),stream,indent=2);stream.write('\n')
    return binding


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='mode')
    sub.add_parser('describe')
    setup=sub.add_parser('prepare');setup.add_argument('destination',type=Path)
    args=parser.parse_args(argv)
    print(json.dumps(prepare(args.destination)if args.mode=='prepare'else describe(),indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
