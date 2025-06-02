import pandas as pd
import numpy as np
import os
import shutil
import sys
import glob
from pathlib import Path
import re
import argparse


def reindex_pdb(input_path, output_path):
    """
    If any ATOM/HETATM residue index is 0, shift all by +1; otherwise copy file unchanged.
    """
    lines = Path(input_path).read_text().splitlines(keepends=True)
    seqs = []
    for L in lines:
        if L.startswith(('ATOM  ', 'HETATM')):
            try:
                seqs.append(int(L[22:26]))
            except ValueError:
                pass
    # nothing to do if already 1-based
    if not seqs or min(seqs) >= 1:
        shutil.copy2(input_path, output_path)
        return
    # else shift everything by +1
    print(f"Reindexing {input_path} to {output_path} by shifting all ATOM/HETATM indices by +1")
    shift = 1
    with open(output_path, 'w') as fout:
        for L in lines:
            if L.startswith(('ATOM  ', 'HETATM')):
                try:
                    old = int(L[22:26])
                    new = old + shift
                    fout.write(L[:22] + f"{new:>4}" + L[26:])
                except ValueError:
                    fout.write(L)
            else:
                fout.write(L)


def parse_placements(row):
    """
    Build a contig string of head/gaps/tail with chain IDs for multiple motif segments.
    Example: sample_placements="11_25,35_49", length_fixed=125 → "11/A/9/B/75".
    """
    # split into start-end pairs
    segs = row['sample_placements'].split(',')
    starts, ends = zip(*(map(int, s.split('_')) for s in segs))
    length = row.get('length_fixed', row.get('length'))

    # Extract chain order from contig string
    contig = row['contig']
    # Find all chain IDs in the contig string (they appear after numbers and before numbers)
    chain_ids = []
    for part in contig.split(','):
        # Split on numbers and filter out empty strings
        parts = [p for p in re.split(r'\d+', part) if p]
        # Extract chain IDs (they're single letters)
        for p in parts:
            if p.strip() and p.strip().isalpha():
                chain_ids.append(p.strip())
    
    # Remove duplicates while preserving order
    chain_ids = list(dict.fromkeys(chain_ids))
    
    if len(chain_ids) != len(starts):
        print(f"Warning: Number of chains in contig ({len(chain_ids)}) doesn't match number of segments ({len(starts)}) for {row['pdb_id']}")
        # Fallback to alphabetical ordering if there's a mismatch
        chain_ids = [chr(65 + i) for i in range(len(starts))]

    # compute head, inter-segment gaps, and tail
    head = starts[0]
    gaps = []
    for i in range(len(starts) - 1):
        gaps.append(starts[i+1] - ends[i] - 1)
    tail = length - ends[-1] - 1

    # build tokens
    tokens = []
    if head > 0:
        tokens.append(str(head))

    for i in range(len(starts)):
        tokens.append(chain_ids[i])
        if i < len(starts) - 1:
            if gaps[i] > 0:
                tokens.append(str(gaps[i]))

    if tail > 0:
        tokens.append(str(tail))

    return '/'.join(tokens)


def extract_motif_segments(parent_dir):
    parent_dir = Path(parent_dir)
    records = []
    for fpath in parent_dir.glob('sample_*/motif_segments_*.txt'):
        sample_id = int(fpath.parent.name.split('_', 1)[1])
        problem = fpath.parents[3].name
        raw = fpath.read_text().strip()
        for seg in raw.replace('/', '\n').split():
            records.append({
                'problem': problem,
                'sample_num': sample_id,
                'sample_placements': seg
            })
    df = pd.DataFrame.from_records(records).sort_values(by='sample_num').reset_index(drop=True)
    df['pdb_id'] = df['problem'].apply(lambda x: x.split('_', 1)[1])
    contig_file = '/scratch/users/yaowei/tds/twisted_diffusion_sampler/protein_exp/motif_scaffolding/mb_test_cases.csv'
    contigs = pd.read_csv(contig_file)
    df_joined = df.merge(contigs, left_on='problem', right_on='target', how='left')
    df_joined['motif_placements'] = df_joined.apply(parse_placements, axis=1)
    return df_joined

def extract_test_case(test_case):
    pdb_id = test_case['pdb_id']
    motif_residues = test_case['motif_residues'].split(';')
    num_segments = len(motif_residues)
    chain_id = [chr(65 + i) for i in range(num_segments)]
    motif_inds, motif_lengths = [], []
    for segment in motif_residues:
        if '-' in segment:
            start, end = map(int, segment[1:].split('-'))
            idxs = np.arange(start, end + 1)
            motif_inds.append(idxs)
            motif_lengths.append(len(idxs))
        else:
            idx = np.array([int(segment[1:])])
            motif_inds.append(idx)
            motif_lengths.append(1)
    length = test_case['length']
    return pdb_id, chain_id, motif_residues, motif_inds, motif_lengths, length

def create_scaffold_info(motif_segments_df, out_dir):
    scaffold_df = motif_segments_df[['sample_num', 'motif_placements']]
    scaffold_df.to_csv(os.path.join(out_dir, 'scaffold_info.csv'), index=False)
    return scaffold_df

def extract_scaffolds(parent_dir, output_dir, test_case_id=None):
    subbed_dir = os.path.dirname(re.sub(r'(?<=/tds_run/)[^/]+', test_case_id, parent_dir))
    subdirs = [d for d in sorted(os.listdir(subbed_dir)) if os.path.isdir(os.path.join(subbed_dir, d))]
    if not subdirs:
        raise RuntimeError(f"No subdirectories found in {subbed_dir}")
    length_dir = subdirs[0]
    parent_dir = os.path.join(subbed_dir, length_dir)

    if not test_case_id:
        print(f"ERROR: Could not parse test_case_id from {parent_dir}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(parent_dir):
        print(f"ERROR: parent_dir not found: {parent_dir}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)

    for entry in sorted(os.listdir(parent_dir)):
        sample_dir = os.path.join(parent_dir, entry)
        if not os.path.isdir(sample_dir) or not entry.startswith("sample_"):
            continue

        pdbs = [f for f in os.listdir(sample_dir) if f.lower().endswith(".pdb")]
        if not pdbs:
            print(f"[WARN] no .pdb in {sample_dir}", file=sys.stderr)
            continue
        
        sc_results = [f for f in os.listdir(sample_dir) if f.lower().endswith(".csv")]
        if not sc_results:
            print(f"[WARN] no sc_results.csv in {sample_dir}", file=sys.stderr)
            continue

        src = os.path.join(sample_dir, pdbs[0])
        dst_dir = os.path.join(output_dir, 'scaffolds', test_case_id)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, f"{test_case_id}_{entry.split('_')[1]}.pdb")
        reindex_pdb(src, dst)
        # shutil.copy2(src, dst)

        # copy per-sample CSV
        src_sc = os.path.join(sample_dir, sc_results[0])
        dst_sc_dir = os.path.join(output_dir, 'sc_results', test_case_id)
        os.makedirs(dst_sc_dir, exist_ok=True)
        dst_sc = os.path.join(dst_sc_dir, f"{test_case_id}_{entry.split('_')[1]}.csv")
        shutil.copy2(src_sc, dst_sc)

    return parent_dir

def combine_sc_results(test_case_id, out_dir):
    sc_dir = os.path.join(out_dir, 'sc_results', test_case_id)
    pattern = os.path.join(sc_dir, f"{test_case_id}_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[WARN] no sc_results to combine in {sc_dir}", file=sys.stderr)
        return

    dfs = []
    for fn in files:
        df = pd.read_csv(fn)
        df['test_case_id'] = test_case_id
        sample_id = os.path.basename(fn).split('_', 1)[1].rsplit('.csv', 1)[0]
        df['sample_id'] = sample_id
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    out_path = os.path.join(sc_dir, f"{test_case_id}_combined_sc_results.csv")
    combined.to_csv(out_path, index=False)
    print(f"Combined {len(files)} files → {out_path}")

def combine_all_sc_results(out_dir):
    sc_root = os.path.join(out_dir, 'sc_results')
    all_dfs = []
    for tc in sorted(os.listdir(sc_root)):
        tc_dir = os.path.join(sc_root, tc)
        combined_path = os.path.join(tc_dir, f"{tc}_combined_sc_results.csv")
        if os.path.isfile(combined_path):
            df = pd.read_csv(combined_path)
            all_dfs.append(df)
        else:
            print(f"[WARN] missing {combined_path}", file=sys.stderr)

    if not all_dfs:
        print("[WARN] no combined per-test CSVs found; skipping final merge", file=sys.stderr)
        return

    final_df = pd.concat(all_dfs, ignore_index=True)
    out_path = os.path.join(sc_root, 'all_sc_results.csv')
    final_df.to_csv(out_path, index=False)
    print(f"Written final all-combined CSV → {out_path}")


def compile_scaffold_outputs(parent_dir, out_dir):
    gp = os.path.dirname(os.path.dirname(os.path.dirname(parent_dir)))
    test_cases = sorted([d for d in os.listdir(gp) if os.path.isdir(os.path.join(gp, d))])

    for test_case_id in test_cases:
        sample_dir = extract_scaffolds(parent_dir, out_dir, test_case_id)
        motif_segments_df = extract_motif_segments(sample_dir)
        scaffold_dir = os.path.join(out_dir, 'scaffolds', test_case_id)
        create_scaffold_info(motif_segments_df, scaffold_dir)
        combine_sc_results(test_case_id, out_dir)
    
    combine_all_sc_results(out_dir)
    print('done!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile scaffold outputs")
    parser.add_argument("--parent_dir", type=str, help="Parent directory containing scaffold outputs")
    parser.add_argument("--out_dir", type=str, help="Output directory to save compiled results")
    args = parser.parse_args()

    compile_scaffold_outputs(args.parent_dir, args.out_dir)
