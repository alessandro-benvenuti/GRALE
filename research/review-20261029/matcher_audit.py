"""CPU checkpoint inspection only: no training or checkpoint mutation.

Run from GRALE with the existing environment:
  module load arch/h100 pytorch-gpu/py3/2.3.1
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \\
    python research/review-20261029/matcher_audit.py

Writes matcher_audit.json beside this script. Target columns are conditioned
over all 64 outputs; active output rows for the L1 sign/cost diagnostic are
those receiving more than 0.5 total mass from real target nodes.
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
root=Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
from GRALE.main import GRALE_model
from GRALE.data.geometry import GeometricDataset
from GRALE.data.dataset import custom_collate_fn
from GRALE.loss.matching import batched_hungarian_projection
from GRALE.loss.utils import pairwise_L1_norm

torch.set_num_threads(1)
manifest=root/'logs/geometric-overfit-20261029/subsets/tiny-1234.json'
data=GeometricDataset(manifest)
records=[]
for config in ['octa_straight','octa_straight_laplacian']:
    ckpt=root/f'logs/geometric-overfit-20261029/tiny/{config}/last.ckpt'
    model=GRALE_model.load_from_checkpoint(ckpt,map_location='cpu').eval()
    for count in [40,2,8]:
        ix=next(i for i,s in enumerate(data.samples) if s['nodes']==count)
        batch=custom_collate_fn([data[ix]])
        with torch.no_grad():
            inp=model.input_builder(batch.clone())
            encoded, z=model.encoder(inp)
            decemb,out=model.decoder(z)
            P,_=model.matcher(encoded,~inp.h,decemb)
            hard,_=model.matcher(encoded,~inp.h,decemb,hard=True)
            active=P[0,:,:count]
            # Conditional output distribution for each active target.
            conditional=active/active.sum(0,keepdim=True)
            entropy=-(conditional*conditional.clamp_min(1e-30).log()).sum(0)
            hardxy=out.nodes.features[0][hard[0]][:count]
            gt=inp.nodes.features[0,:count]
            bary=(conditional.T@out.nodes.features[0])
            a=model.matcher.linear_outputs(decemb+model.matcher.positionnal_encoding_outputs)
            emb=encoded.clone()
            emb[~inp.h]=model.matcher.node_padding_features[0,0]
            b=model.matcher.linear_inputs(emb)
            C=pairwise_L1_norm(a,b)
            C=C/C.sum((1,2),keepdim=True)/model.matcher.epsilon
            selected=P[0,:,:count].sum(1)>.5
            residual=C[0,selected,:count]
            residual=residual-residual.mean(0,keepdim=True)-residual.mean(1,keepdim=True)+residual.mean()
            signs=(a[0,selected,None]-b[0,None,:count]).sign()
            constant_sign_dims=((signs.amax(0).amax(0)-signs.amin(0).amin(0))==0)
            # For each output, compare its learned location with the target
            # barycenter induced by that output's conditional transport row.
            rowweights=active/active.sum(1,keepdim=True).clamp_min(1e-30)
            weightedgt=rowweights@gt
            mass=active.sum(1)
            informative=mass>.5
            softcost=((out.nodes.features[0,:,None]-gt[None])**2).sum(-1)
            result={
                'config':config,'checkpoint':str(ckpt.relative_to(root)),
                'sample':data.samples[ix]['sample_id'],'n':count,'transport_shape':list(P.shape),
                'target_entropy_mean':entropy.mean().item(),
                'target_entropy_normalized_log64':(entropy/np.log(64)).mean().item(),
                'target_max_weight_mean':conditional.max(0).values.mean().item(),
                'active_output_mass_gt_half':informative.sum().item(),
                'row_barycenter_vs_prediction_rmse_px':(((out.nodes.features[0]-weightedgt).square().sum(-1)*mass).sum()/mass.sum()).sqrt().item()*128,
                'transport_soft_xy_rmse_px':((softcost*active).sum()/active.sum()).sqrt().item()*128,
                'hard_matcher_xy_rmse_px':(hardxy-gt).square().sum(-1).mean().sqrt().item()*128,
                'target_barycenter_output_rmse_px':(bary-gt).square().sum(-1).mean().sqrt().item()*128,
                'logkernel_cost_range':(C.max()-C.min()).item(),
                'active_logkernel_cost_range':(C[0,:,:count].max()-C[0,:,:count].min()).item(),
                'active_cost_doublecenter_max_abs':residual.abs().max().item(),
                'constant_L1_sign_dims':constant_sign_dims.sum().item(),
                'encoder_active_embedding_spread':encoded[0,:count].std(0,unbiased=False).norm().item(),
                'encoder_active_projected_spread':b[0,:count].std(0,unbiased=False).norm().item(),
                'marginal_row_max_error':(P.sum(2)-1).abs().max().item(),
                'marginal_col_max_error':(P.sum(1)-1).abs().max().item(),
                'pred_xy_spread_px':out.nodes.features[0][informative].std(0,unbiased=False).square().sum().sqrt().item()*128,
            }
            records.append(result)
            print(json.dumps(result),flush=True)
Path(__file__).with_suffix('.json').write_text(json.dumps({
    'mode':'CPU inference only; no training', 'torch':torch.__version__,
    'manifest':str(manifest.relative_to(root)), 'records':records,
},indent=2)+'\n')
