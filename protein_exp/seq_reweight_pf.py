# a wrapper to do sequence-aware reweighting of the particle filter after a specified timestep

# Standard library imports
import sys
import os
import copy

# Third-party imports
import pandas as pd
import numpy as np
import torch
from hydra import main, initialize, compose
from omegaconf import OmegaConf, DictConfig

# Project-specific imports
from experiments.inference_motif_scaffolding import Sampler, construct_output_dir
from experiments.inference_particle_filter import run_particle_filter, init_particle_filter, log_and_validate_PF_output
from smc_utils_prot import feynman_kac_pf, smc_utils
from ProteinMPNN import protein_mpnn_run
from motif_scaffolding import save_motif_segments, twisting
from ProteinMPNN.protein_mpnn_utils import (
    loss_nll, loss_smoothed, gather_edges, gather_nodes, gather_nodes_t, cat_neighbors_nodes, _scores, _S_to_seq, tied_featurize, parse_PDB,
    StructureDataset, StructureDatasetPDB, ProteinMPNN
)
from analysis import utils as au
from data import all_atom
from openfold.utils import rigid_utils as ru

sys.path.append(os.path.abspath("experiments"))
with initialize(version_base = None, config_path = "config"): cfg = compose(config_name = "inference")

def compute_logp_seq(
    sampler,
    rigids7_np,      # numpy array [P, L, 7]
    mpnn,            # ProteinMPNN instance
    cfg,             # Hydra config
    sample_dir       # path to the working sample directory
) -> torch.Tensor:
    """
    converts P rigid-7 frames → PDB → ProteinMPNN log-likelihood [P]
    """
    entries = []
    P = rigids7_np.shape[0]

    for k in range(P):
        r7 = torch.tensor(rigids7_np[k], dtype = torch.float64)
        # print('[DEBUG compute_logp_seq] r7 shape: ', r7.shape)
        rigid = ru.Rigid.from_tensor_7(r7)
        # print('[DEBUG compute_logp_seq] rigid shape: ', rigid.shape)
        L = r7.shape[0]
        # print('[DEBUG compute_logp_seq] L: ', L)

        psis = torch.zeros((L, 2), dtype = torch.float64, device = r7.device)
        # print('[DEBUG compute_logp_seq] psis shape: ', psis.shape)
        atom37_coords, *_ = all_atom.compute_backbone(rigid, psis)  # → [L,37,3]

        tmp_prefix = os.path.join(sample_dir, f"tmp_seq_{k}")
        pdb_path = au.write_prot_to_pdb(
            atom37_coords.cpu().numpy(),
            tmp_prefix,
            aatype=None,
            b_factors=None
        )

        entries.extend(parse_PDB(pdb_path))

        os.remove(pdb_path)

    feats = tied_featurize(
        entries,
        device = "cpu",
        chain_dict = {e["name"]: ([], ["A"]) for e in entries}
    )
    (X, S, mask, lengths, chain_M, chain_encoding, letter_list, visible, masked, masked_lens, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos, pssm_coef, pssm_bias, pssm_logodds, bias_by_res, tied_beta) = feats

    with torch.no_grad():
        noise = torch.randn(chain_M.shape, device = X.device)
        log_probs = mpnn(
            X, S, mask,
            chain_M * chain_M_pos,
            residue_idx, chain_encoding,
            noise,
            use_input_decoding_order = False
        ) 
    summed = (log_probs.gather(-1, S.unsqueeze(-1)).squeeze(-1) * mask).sum(-1)
    lengths = mask.sum(-1).clamp(min=1)
    logp_seq = summed / lengths
    return logp_seq

def composite_G2_factory(
    sampler, compute_logp_seq, mpnn, cfg, sample_dir, omega_geom, omega_seq, T2
):
    """
    Returns a weight function G2(sample_feats, extra) that always
    uses both geometry and sequence terms:
        ω_geom*log p_spatial + ω_seq*logp_seq
    """
    def composite_G2(t, xtp1, xt, extra_vals):
        def unpack_rigids(x):
            return x['rigids_t'] if isinstance(x, dict) else x
        if xtp1 is None:
            t0 = sampler._diff_conf.num_t
            # x0_val = xt['rigids_t']
            xt_raw = unpack_rigids(xt)
            xtp1_raw = xt_raw
            logp_s = sampler.G(t = sampler._diff_conf.num_t, xtp1=xt_raw, xt=xt_raw, extra_vals=extra_vals)
        else:
            xtp1_raw = unpack_rigids(xtp1)
            # print('[DEBUG composite_G2] xtp1_raw shape: ', xtp1_raw.shape)
            xt_raw = unpack_rigids(xt)
            # print('[DEBUG composite_G2] xt_raw shape: ', xt_raw.shape)
            # logp_s = sampler.G(t = t, xtp1 = xtp1, xt = xt, extra_vals = extra_vals)    # spatial term [P]
            logp_s = sampler.G(t=t, xtp1=xtp1_raw, xt=xt_raw, extra_vals=extra_vals)
        
        # rig7 = xt["rigids_t"][:, 0].cpu().numpy()
        # print('[DEBUG composite_G2] xt_raw shape: ', xt_raw.shape)
        rigids7_np = xt_raw.cpu().numpy()
        print('[DEBUG composite_G2] rigids7_np shape: ', rigids7_np.shape)
        # logp_q = compute_logp_seq(sampler, rig7, mpnn, cfg, sample_dir)
        # print('[DEBUG composite_G2] rigids7_np shape: ', rigids7_np.shape)
        logp_q = compute_logp_seq(sampler, rigids7_np, mpnn, cfg, sample_dir)
        # print('[DEBUG composite_G2] logp_q shape: ', logp_q.shape)
        beta = float((T2 - t) / T2)
        beta = max(0.0, min(1.0, beta))
        print('[DEBUG composite_G2] beta: ', beta)
        comp_logp = omega_geom * logp_s + beta * (omega_seq * logp_q)
        # print('[DEBUG composite_G2] comp_logp: ', comp_logp)
        return comp_logp
    return composite_G2

# def M2_factory(x0_stage1, sampler, T2):
#     def M2(t, xtp1, extra_vals, P=None):
#         if t == T2:
#             # sampler.PF_cache["sample_feats"]["rigids_t"] = x0_stage1
#             feats = dict(sampler.PF_cache["sample_feats"])
#             feats["rigids_t"] = x0_stage1
#             return feats, extra_vals
#             # return x0_stage1, extra_vals
#         return sampler.M(t=t, xtp1=xtp1, extra_vals=extra_vals, P=P)
#     return M2

def M2_factory(x0_stage1, sampler, T2):
    def M2(t, xtp1, extra_vals, P=None):
        if t == T2:
            return x0_stage1, extra_vals

        out, new_extra = sampler.M(t=t, xtp1=xtp1, extra_vals=extra_vals, P=P)
        # if out is a dict, pull out "rigids_t"; otherwise assume it's already the tensor:
        if isinstance(out, dict):
            rigids = out["rigids_t"]
        else:
            rigids = out
        return rigids, new_extra

    return M2

def run_stage2(
    sampler, x0_stage1, compute_logp_seq,
    mpnn, cfg, sample_dir,
    omega_geom, omega_seq, T2, resample):

    P = cfg.inference.particle_filtering.number_of_particles

    G2 = composite_G2_factory(
        sampler, compute_logp_seq, mpnn, cfg, sample_dir,
        omega_geom, omega_seq, T2)
    # print('[DEBUG run_stage2] G2: ', G2)
    M2 = M2_factory(x0_stage1, sampler, T2)
    # print('[DEBUG run_stage2] M2: ', M2)

    return feynman_kac_pf.smc_FK(
        M2, G2, resample, T2, P,
        verbose=True
    )

def normalize_weights(logw: torch.Tensor):
    logw_c = logw - torch.logsumexp(logw, dim=0)
    w_prob = torch.exp(logw_c)
    ess = 1.0 / torch.sum(w_prob**2)
    return w_prob, ess


@main(version_base = None, config_path = "config", config_name = "inference")
def two_stage(cfg: DictConfig):
    # new params from config file
    reweight_t = cfg.inference.motif_scaffolding.reweight_t
    omega_geom = cfg.inference.motif_scaffolding.omega_geom
    omega_seq = cfg.inference.motif_scaffolding.omega_seq

    # old params
    test_name = cfg.inference.motif_scaffolding.test_name
    num_samples = cfg.inference.motif_scaffolding.number_of_samples
    num_t = cfg.inference.diffusion.num_t

    sampler = Sampler(cfg)
    output_dir_stem = sampler._output_dir
    inpaint_df = pd.read_csv(sampler._infer_conf.motif_scaffolding.inpaint_cases_csv)
    contigs_by_test_case = save_motif_segments.load_contigs_by_test_case(inpaint_df)
    motif_contig_info = contigs_by_test_case[test_name]

    sampler._output_dir = construct_output_dir(sampler, test_name, output_dir_stem)
    print("output_dir: ",sampler._output_dir)
    os.makedirs(sampler._output_dir, exist_ok = True)
    
    keep_motif_seq = sampler._infer_conf.motif_scaffolding.keep_motif_seq
    P = sampler._infer_conf.particle_filtering.number_of_particles
    insert_motif_at_t0 = sampler._infer_conf.particle_filtering.insert_motif_at_t0
    verbose = True

    for sample_id in range(sampler._infer_conf.motif_scaffolding.number_of_samples):
        print('Running sample: ', sample_id)
        row = list(inpaint_df[inpaint_df.target==test_name].iterrows())[0][1]
        motif_contig_info = save_motif_segments.load_contig_test_case(row)

        init_particle_filter(sampler, motif_contig_info, P = P)
        print("Initialized particle filter with motif contig info: ", motif_contig_info)

        sample_dir = os.path.join(sampler._output_dir, f'length_{sampler.PF_cache["length"]}', f'sample_{sample_id}')
        if os.path.isdir(sample_dir):
            print("Skipping sample, prev sample already run", sample_id)
            continue
        os.makedirs(sample_dir, exist_ok = True)

        # Set transition kernels and potential fucntion
        M = sampler.M
        G = sampler.G
        T = sampler._diff_conf.num_t
        resample = smc_utils.resampling_function(
            ess_threshold=sampler._infer_conf.particle_filtering.resample_threshold,
            verbose=verbose)
        
        # Run particle filter
        sampler._log.info(f'Running particle filter')
        xts, log_w, resample_indices_trace, ess_trace, log_w_trace = feynman_kac_pf.smc_FK(
            M, G, resample, T, P, verbose=verbose)

        x0_stage1 = xts[-1] # [K, L, 7]
        w0_stage1 = log_w_trace[-1] # [K]

        print('x0_stage1 shape: ', x0_stage1.shape)
        print('w0_stage1 shape: ', w0_stage1.shape)
        print("Stage 1 PF complete, now running stage 2")

        mpnn_ckpt = torch.load(
            os.path.join(cfg.inference.pmpnn_dir, 'vanilla_model_weights', 'v_48_020.pt'),
            weights_only = False,
            map_location = 'cpu'
        )

        hidden_dim = 128
        num_layers = 3

        device = 'cpu'
        mpnn = ProteinMPNN(
            num_letters = 21,
            node_features = hidden_dim, 
            edge_features = hidden_dim, 
            hidden_dim = hidden_dim, 
            num_encoder_layers = num_layers, 
            num_decoder_layers = num_layers, 
            k_neighbors = mpnn_ckpt['num_edges'],
            augment_eps = 0
        )
        mpnn.to(device)
        mpnn.load_state_dict(mpnn_ckpt['model_state_dict'])
        mpnn.eval()

        xts2, logw2, resample_idx2, ess2, logw_trace2 = run_stage2(
            sampler,
            x0_stage1,
            compute_logp_seq,
            mpnn,
            cfg,
            sample_dir,
            omega_geom,
            omega_seq,
            reweight_t,
            resample
        )

        # write out final Stage 2 outputs exactly like vanilla TDS:
        log_and_validate_PF_output(
            sampler,
            log_w_trace=logw_trace2,
            ess_trace=ess2,
            sample_id=sample_id,
            sample_dir=sample_dir,
            keep_motif_seq=keep_motif_seq,
            insert_motif_at_t0=insert_motif_at_t0
        )

        motif_segments = sampler.PF_cache["rigids_motif"]
        sampler.run_self_consistency(
            sampler._output_dir,
            rigids_motif=motif_segments,
            use_motif_seq=keep_motif_seq,
            motif_contig_info=motif_contig_info
        )

        torch.cuda.empty_cache()

if __name__ == "__main__":
    two_stage()
