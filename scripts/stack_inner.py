"""The stacked member's inputs: inner runs of the GRU member and the stack table (``prc.eval.stack``).

``exp0235_r7d_cat_stack_gru`` is a CatBoost member with one extra column, the seed-averaged
normal-expert prediction of the GRU member ``exp0223_r7c_seqenc_slog``. For each outer fold the
training rows get that column from inner runs that never see the outer month: the fold's 11
training months in 3 interleaved groups. Validation rows get it from the GRU's own
``month_loo`` fold, test rows from its full fit.

    # list the (seed, outer training-month set) tasks of a job array (39 for 3 seeds)
    uv run python scripts/stack_inner.py plan --member exp0223_r7c_seqenc_slog
    # one array task: the inner run(s) of task INDEX (or all of them without --index)
    uv run python scripts/stack_inner.py inner --member exp0223_r7c_seqenc_slog \\
        --out outputs/runs/inner-exp0223_r7c_seqenc_slog --index 0
    # after every inner run and the member's month_loo / tail_nov_dec runs: the table
    uv run python scripts/stack_inner.py build --member exp0223_r7c_seqenc_slog \\
        --inner-root outputs/runs/inner-exp0223_r7c_seqenc_slog \\
        --loo 'outputs/runs/loo-exp0223_r7c_seqenc_slog/exp0223_r7c_seqenc_slog__s{seed}__month_loo' \\
        --served 'x11-12=outputs/runs/fwd-exp0223_r7c_seqenc_slog/exp0223_r7c_seqenc_slog__s{seed}__tail_nov_dec'

The table lands in ``data/stack/exp0223_r7c_seqenc_slog__g3/``, where the ``stack_member`` key of
the consuming YAML finds it.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from prc.eval import stack


def _tasks(args) -> list[tuple[int, list[str]]]:
    sets = stack.outer_train_sets(args.schemes.split(","))
    return [(s, t) for s in args.seed_list for t in sets]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("cmd", choices=["plan", "inner", "build"])
    p.add_argument("--member", required=True, help="member experiment name")
    p.add_argument("--groups", type=int, default=3, help="inner month groups")
    p.add_argument("--seeds", default=None, help="comma-separated; default the member's seeds")
    p.add_argument(
        "--schemes",
        default="month_loo,tail_nov_dec",
        help="consumer schemes whose folds need inner runs (month_jan_jul is inside month_loo)",
    )
    p.add_argument("--out", type=Path, help="inner: output root")
    p.add_argument("--index", type=int, help="inner: run only this task of `plan`")
    p.add_argument("--inner-root", type=Path, help="build: the inner runs' root")
    p.add_argument("--loo", help="build: member month_loo run dir template with {seed}")
    p.add_argument("--served", action="append", default=[], help="build: KEY=template")
    p.add_argument("--table", type=Path, help="build: output dir (default data/stack/...)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    member = stack.member_config(args.member)
    args.seed_list = [int(s) for s in args.seeds.split(",")] if args.seeds else list(member.seeds)
    tasks = _tasks(args)
    if args.cmd == "plan":
        for i, (s, t) in enumerate(tasks):
            name = stack.inner_run_name(member.name, s, stack.inner_scheme(t, args.groups))
            print(f"{i}\t{name}")
        print(f"--array=0-{len(tasks) - 1}")
        return 0
    if args.cmd == "inner":
        if args.out is None:
            p.error("inner needs --out")
        todo = tasks if args.index is None else [tasks[args.index]]
        for s, t in todo:
            stack.run_inner(member, [t], s, args.out, args.groups)
        return 0
    if args.inner_root is None or args.loo is None:
        p.error("build needs --inner-root and --loo")
    served = dict(kv.split("=", 1) for kv in args.served)
    stack.build_table(
        member,
        args.seed_list,
        args.groups,
        stack.outer_train_sets(args.schemes.split(",")),
        args.inner_root,
        args.loo,
        served,
        args.table,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
