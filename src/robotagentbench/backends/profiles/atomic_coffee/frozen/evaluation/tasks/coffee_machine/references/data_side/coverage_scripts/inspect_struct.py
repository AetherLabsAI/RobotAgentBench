#!/usr/bin/env python3
"""Print a compact structural outline of a JSON file (keys, types, list lengths), never dumping big content."""
import json, sys

def outline(o, depth=0, maxdepth=4, maxkeys=40, prefix=''):
    ind = '  ' * depth
    if isinstance(o, dict):
        keys = list(o.keys())
        for k in keys[:maxkeys]:
            v = o[k]
            if isinstance(v, dict):
                print(f"{ind}{k}: dict[{len(v)}]")
                if depth < maxdepth:
                    outline(v, depth + 1, maxdepth, maxkeys)
            elif isinstance(v, list):
                t = type(v[0]).__name__ if v else 'empty'
                extra = ''
                if v and isinstance(v[0], (int, float, str, bool)) and len(v) <= 12:
                    extra = ' = ' + json.dumps(v, ensure_ascii=False)[:200]
                elif v and isinstance(v[0], list):
                    extra = f' inner_len={len(v[0])}'
                print(f"{ind}{k}: list[{len(v)}] of {t}{extra}")
                if v and isinstance(v[0], dict) and depth < maxdepth:
                    outline(v[0], depth + 1, maxdepth, maxkeys)
            else:
                s = json.dumps(v, ensure_ascii=False)
                if len(s) > 160:
                    s = s[:160] + '...'
                print(f"{ind}{k}: {s}")
        if len(keys) > maxkeys:
            print(f"{ind}... (+{len(keys)-maxkeys} more keys)")
    elif isinstance(o, list):
        print(f"{ind}list[{len(o)}]")
        if o and depth < maxdepth:
            outline(o[0], depth + 1, maxdepth, maxkeys)

if __name__ == '__main__':
    md = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    o = json.load(open(sys.argv[1]))
    outline(o, maxdepth=md)
