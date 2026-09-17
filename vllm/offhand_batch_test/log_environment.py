#!/usr/bin/env python3
"""Log only environment names supplied by the matrix configuration."""

import json
import os
import sys


CONFIG_ENV_KEYS = "OFFHAND_CONFIG_ENV_KEYS"


def format_environment(env, names):
    # JSON keeps empty and unset values distinct and escapes embedded newlines.
    return json.dumps({name: env.get(name) for name in sorted(names)}, ensure_ascii=False)


if __name__ == "__main__":
    names = json.loads(os.environ.get(CONFIG_ENV_KEYS, "[]"))
    if names:
        print(f"{sys.argv[1]}: {format_environment(os.environ, names)}", flush=True)
