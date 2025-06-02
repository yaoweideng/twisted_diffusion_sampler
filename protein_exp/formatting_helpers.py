import os
import glob
import pandas as pd
from pathlib import Path
import re

def parse_placements(row):
    segs = row['sample_placements'].split(',')
    contig_parts = row['contig'].split(',')
    length = row['length_fixed']
    # extract chain IDs in order from contig (anything with letters)
    chain_ids = [
        re.match(r'([A-Za-z]+)', p).group(1)
        for p in contig_parts
        if re.search(r'[A-Za-z]', p)
    ]
    if len(segs) != len(chain_ids):
        raise ValueError(f"got {len(segs)} segments but {len(chain_ids)} chains in row {row.name}")
    # parse starts and ends
    starts, ends = zip(*(map(int, s.split('_')) for s in segs))
    # prefix = start₁ - 1
    tokens = [ str(starts[0] - 1) ]
    # for each chain except the last, emit: /Chainᵢ / (endᵢ-startᵢ+1)
    for i in range(len(chain_ids)-1):
        tokens += [ chain_ids[i], str(ends[i] - starts[i] + 1) ]
    # for the last chain, emit: /Chainₙ / (length_fixed - endₙ)
    tokens += [ chain_ids[-1], str(length - ends[-1]) ]
    return '/'.join(tokens)

def extract_motif_segments(parent_dir):
    parent_dir = Path(parent_dir)
    records = []
    for fpath in parent_dir.glob('sample_*/motif_segments_*.txt'):
        # extract sample number
        sample_id = int(fpath.parent.name.split('_', 1)[1])
        # problem folder three levels up
        problem = fpath.parents[3].name
        # read and split segments
        raw = fpath.read_text().strip()
        for seg in raw.replace('/', '\n').split():
            records.append({
                'problem': problem,
                'sample_num': sample_id,
                'sample_placements': seg
            })
    df = pd.DataFrame.from_records(records).sort_values(by = 'sample_num').reset_index(drop = True)
    df['pdb_id'] = df['problem'].apply(lambda x: x.split('_', 1)[1])    # join with contigs
    contig_file = '/home/groups/btrippe/yaowei/twisted_diffusion_sampler/protein_exp/motif_scaffolding/mb_test_cases.csv'
    contigs = pd.read_csv(contig_file)
    df_joined = df.merge(contigs, left_on='problem', right_on='target', how='left')
    df_joined['motif_placements'] = df_joined.apply(parse_placements, axis=1)
    return df_joined
