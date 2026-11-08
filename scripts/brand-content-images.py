#!/usr/bin/env python3
"""Inventory, sample, publish, and verify the configured teaching-image branding.

Examples (run in the repository with a Pillow-enabled Python):
  python -B scripts/brand-content-images.py dry-run --report output/image-branding/dry-run.json
  python -B scripts/brand-content-images.py sample --only basics/example.png --destination output/branding-samples
  python -B scripts/brand-content-images.py apply --jobs 4
  python -B scripts/brand-content-images.py verify --jobs 4

No model calls, semantic image review, builds, or preview regeneration occur here.
Run prepare-home-previews.py once after a successful full apply/verify.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from image_branding import BrandingError, BrandingStore, atomic_json, load_policy


ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['dry-run', 'apply', 'verify', 'status', 'sample'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--only', action='append', help='Exact path relative to public/images; repeatable')
    parser.add_argument('--jobs', type=int, default=4, choices=range(1, 7))
    parser.add_argument('--report', type=Path)
    parser.add_argument('--destination', type=Path, help='Required for sample; must be outside public')
    parser.add_argument('--deep', action='store_true', help='Verify by reproducing output from the original backup')
    parser.add_argument('--update-policy', action='store_true',
                        help='For apply: replace a recorded policy using verified original backups')
    args = parser.parse_args(argv)
    if args.update_policy and args.command != 'apply':
        parser.error('--update-policy is only supported by apply')
    if args.command == 'sample' and (not args.destination or not args.only):
        parser.error('sample requires --destination and at least one --only path')
    policy = load_policy(args.root, args.config)
    if policy is None:
        parser.error('An enabled config/image-branding.json policy is required')
    store = BrandingStore(args.root, policy)
    paths = store.inventory(args.only)
    # A missing public file must not disappear from a complete verify/status.
    if not args.only and store.state.is_dir():
        for record_path in (store.state / 'records').glob('*.json'):
            record = json.loads(record_path.read_text())
            relative = record.get('path')
            if not isinstance(relative, str):
                raise BrandingError('Invalid journal path in ' + str(record_path))
            if not policy.excluded(relative) and relative not in paths:
                paths.append(relative)
        paths.sort()
    started = time.monotonic()
    print(json.dumps({'event': 'start', 'command': args.command, 'total': len(paths),
                      'jobs': args.jobs, 'policy_sha256': policy.fingerprint}), flush=True)

    def run(relative):
        if args.command == 'apply':
            return store.publish(relative, allow_policy_update=args.update_policy)
        if args.command == 'sample':
            return store.sample(relative, args.destination)
        return store.inspect(relative, deep=args.deep, require_published=args.command == 'verify')

    rows, failures = [], []
    counts = Counter()
    last_progress = started
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        tasks = {pool.submit(run, path): path for path in paths}
        for future in as_completed(tasks):
            relative = tasks[future]
            try:
                row = future.result()
            except (BrandingError, OSError, ValueError, KeyError) as error:
                row = {'path': relative, 'status': 'error', 'error': str(error)}
                failures.append(row)
                print(json.dumps({'event': 'error', **row}, ensure_ascii=False), flush=True)
            rows.append(row)
            counts[row['status']] += 1
            now = time.monotonic()
            if now - last_progress >= 2 or len(rows) == len(paths):
                print(json.dumps({'event': 'progress', 'completed': len(rows), 'total': len(paths),
                                  'counts': dict(counts), 'elapsed_seconds': round(now - started, 1)}), flush=True)
                last_progress = now
    result = {'version': 1, 'command': args.command, 'passed': not failures, 'total': len(paths),
              'policy_sha256': policy.fingerprint, 'policy': policy.provenance,
              'counts': dict(counts), 'elapsed_seconds': round(time.monotonic() - started, 1),
              'images': sorted(rows, key=lambda row: row['path']), 'errors': failures}
    if args.report:
        atomic_json(args.report, result)
    if args.command == 'apply':
        # Individual records, not this possibly interrupted/subset summary, are authoritative.
        atomic_json(store.state / 'last-apply.json', result)
    if args.command in {'verify', 'status'} and not args.only:
        atomic_json(store.state / 'manifest.json', result)
    print(json.dumps({key: value for key, value in result.items() if key not in {'images', 'policy'}},
                     ensure_ascii=False), flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (BrandingError, OSError, ValueError) as error:
        print(json.dumps({'event': 'fatal', 'error': str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
