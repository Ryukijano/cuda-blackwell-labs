#!/usr/bin/env python3
"""Generate graphs and tables for the memory bandwidth lab."""

import csv
import os
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except ImportError:
    MATPLOTLIB = False
    print("matplotlib not available; only tables will be written.", file=sys.stderr)

RESULTS = Path('results')
OUT_DIR = RESULTS
PEAK = 273.0


def read_csv(path):
    rows = []
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            for k, v in r.items():
                try:
                    r[k] = float(v)
                except ValueError:
                    pass
            rows.append(r)
    return rows


def plot_workset():
    rows = read_csv(RESULTS / 'bandwidth_workset.csv')
    wanted = {'sequential_read', 'sequential_write', 'read_write_copy'}
    data = {}
    for r in rows:
        k = r['kernel']
        if k not in wanted:
            continue
        data.setdefault(k, {'x': [], 'y': []})
        data[k]['x'].append(r['size_bytes'])
        data[k]['y'].append(r['bandwidth_gb_s'])

    if not MATPLOTLIB:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for k in wanted:
        if k in data:
            ax.plot(data[k]['x'], data[k]['y'], marker='o', label=k)
    ax.axhline(PEAK, color='r', linestyle='--', label='peak 273 GB/s')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Working-set size (bytes)')
    ax.set_ylabel('Bandwidth (GB/s)')
    ax.set_title('Working-set size vs effective bandwidth')
    ax.legend()
    ax.grid(True, which='both', ls='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'bandwidth_workset.png')
    plt.close(fig)


def plot_stride():
    rows = read_csv(RESULTS / 'bandwidth_stride.csv')
    x = [r['stride'] for r in rows]
    y = [r['bandwidth_gb_s'] for r in rows]

    if not MATPLOTLIB:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, marker='o')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Stride')
    ax.set_ylabel('Bandwidth (GB/s)')
    ax.set_title('Strided read bandwidth at 64 MB')
    ax.grid(True, which='both', ls='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'bandwidth_stride.png')
    plt.close(fig)


def plot_block():
    rows = read_csv(RESULTS / 'bandwidth_block.csv')
    x = [r['block_size'] for r in rows]
    y = [r['bandwidth_gb_s'] for r in rows]

    if not MATPLOTLIB:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, marker='o')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Block size')
    ax.set_ylabel('Bandwidth (GB/s)')
    ax.set_title('Block size vs bandwidth for sequential_read at 64 MB')
    ax.grid(True, which='both', ls='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'bandwidth_block.png')
    plt.close(fig)


def write_table(name, rows, keys, outname):
    out = []
    out.append(f'## {name}')
    out.append('| ' + ' | '.join(keys) + ' |')
    out.append('|' + '|'.join(['---' for _ in keys]) + '|')
    for r in rows:
        line = []
        for k in keys:
            v = r.get(k, '')
            if isinstance(v, float):
                v = f'{v:.2f}'
            line.append(str(v))
        out.append('| ' + ' | '.join(line) + ' |')
    out.append('')
    with open(OUT_DIR / outname, 'w') as f:
        f.write('\n'.join(out))


def tables():
    try:
        alloc = read_csv(RESULTS / 'bandwidth_alloc.csv')
        write_table('Allocation type vs bandwidth', alloc,
                    ['alloc_type', 'access', 'bandwidth_gb_s', 'percent_peak'],
                    'table_alloc.md')
    except FileNotFoundError:
        pass

    try:
        uma = read_csv(RESULTS / 'bandwidth_uma.csv')
        write_table('UMA contention', uma,
                    ['alloc_type', 'contention', 'bandwidth_gb_s', 'percent_peak'],
                    'table_uma.md')
    except FileNotFoundError:
        pass


def main():
    if not RESULTS.exists():
        print(f'{RESULTS} does not exist.', file=sys.stderr)
        return 1
    plot_workset()
    plot_stride()
    plot_block()
    tables()
    print('Plots and tables generated in', OUT_DIR)
    return 0


if __name__ == '__main__':
    sys.exit(main())
