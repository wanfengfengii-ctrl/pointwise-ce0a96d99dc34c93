"""Command-line entry point for the banded affine-gap alignment.

Subcommands
-----------
``demo``   Run a self-contained alignment demonstration on sample channels.
``verify`` Run the independent verification suite (dense reference DP,
           exhaustive enumeration, validation and immutability checks).
``align``  Align two integer channels read from JSON files or literal JSON
           arrays on the command line.

Everything uses the Python standard library only.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import align  # noqa: E402
import verify  # noqa: E402


def _load_channel(spec):
    """Accept either a path to a JSON file or an inline JSON array string."""
    if os.path.isfile(spec):
        with open(spec, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    else:
        data = json.loads(spec)
    if not isinstance(data, list):
        raise SystemExit("channel input must be a JSON array of integers")
    return data


def cmd_align(args):
    left = _load_channel(args.left)
    right = _load_channel(args.right)
    result = align.solve(left, right, args.band, args.open, args.extend)
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0


def cmd_demo(args):
    # Two long sampled channels: the right channel drops a few samples
    # (creating gaps) and carries small measurement noise.
    left = [10 + (i % 7) for i in range(24)]
    right = [left[i] + (1 if i % 5 == 0 else 0)
             for i in range(len(left)) if i not in (6, 7, 15)]
    B, open_, extend = 3, 4, 1
    result = align.solve(left, right, B, open_, extend)
    print("demo alignment (B=%d open=%d extend=%d)" % (B, open_, extend))
    print("left :", left)
    print("right:", right)
    print(json.dumps(result, ensure_ascii=False))
    # Independently recompute and print the cost as an in-line sanity check.
    recomputed = align.path_cost(left, right, B, open_, extend,
                                 result["steps"])
    print("recomputed cost:", recomputed)
    assert recomputed == result["cost"]
    return 0


def cmd_verify(args):
    return verify.main()


def build_parser():
    parser = argparse.ArgumentParser(
        prog="cli",
        description="Banded affine-gap alignment of two integer channels.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_align = sub.add_parser("align", help="align two JSON integer channels")
    p_align.add_argument("left", help="JSON file path or inline JSON array")
    p_align.add_argument("right", help="JSON file path or inline JSON array")
    p_align.add_argument("--band", type=int, default=16)
    p_align.add_argument("--open", type=int, default=4)
    p_align.add_argument("--extend", type=int, default=1)
    p_align.set_defaults(func=cmd_align)

    p_demo = sub.add_parser("demo", help="run a built-in alignment demo")
    p_demo.set_defaults(func=cmd_demo)

    p_verify = sub.add_parser(
        "verify", help="run the independent verification suite")
    p_verify.set_defaults(func=cmd_verify)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
