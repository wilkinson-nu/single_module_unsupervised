from pathlib import Path
import yaml
import os

def dump_args(args, path):
    payload = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=True, default_flow_style=False)
    os.replace(tmp, path)

def load_config(path, _seen=None):
    """Read a YAML config, resolving _base_ inheritance relative to each file."""
    path = Path(path).resolve()
    _seen = set() if _seen is None else _seen
    if path in _seen:
        raise SystemExit(f"circular _base_ involving {path}")
    _seen.add(path)

    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    base = cfg.pop("_base_", None)
    if base is None:
        return cfg

    merged = load_config(path.parent / base, _seen)
    merged.update(cfg)
    return merged

def apply_config(parser, cfg):
    """Validate a config dict against a parser and install it as defaults.

    Keys must be argparse dests. Every non-required, typed argument must be
    present, so adding an option to the parser without adding it to the config
    fails loudly instead of silently becoming None.
    """
    actions = {a.dest: a for a in parser._actions}

    unknown = set(cfg) - set(actions)
    if unknown:
        raise SystemExit(f"unknown config keys: {sorted(unknown)}")

    # Required args and flags (store_true, --help) are CLI business, not config.
    expected = {dest for dest, a in actions.items()
                if a.type is not None and not a.required}
    missing = expected - set(cfg)
    if missing:
        raise SystemExit(f"config is missing keys: {sorted(missing)}")

    resolved = {}
    for key, val in cfg.items():
        act = actions[key]
        if val is not None and act.type is not None:
            try:
                val = act.type(val)
            except (TypeError, ValueError) as exc:
                raise SystemExit(f"config key {key!r}: cannot coerce {val!r}: {exc}")
        if val is not None and act.choices is not None and val not in act.choices:
            raise SystemExit(f"config key {key!r}: {val!r} not in {sorted(act.choices)}")
        resolved[key] = val

    parser.set_defaults(**resolved)
    return parser
