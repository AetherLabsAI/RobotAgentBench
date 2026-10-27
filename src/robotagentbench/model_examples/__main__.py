"""Select an optional model example without importing its model runtime."""
import importlib
import sys


def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ('-h','--help'):
        print('Usage: python -m robotagentbench.model_examples {internvla,codex} COMMAND [OPTIONS]')
        print('Use an example\'s --help for configuration and explicit run commands. Models are not loaded by this selector.')
        return 0
    name=args.pop(0)
    if name not in ('internvla','codex'):
        raise SystemExit('Choose one installed example: internvla or codex')
    module=importlib.import_module('robotagentbench.model_examples.'+name+'.__main__')
    return module.main(args)


if __name__=='__main__':raise SystemExit(main())
