import sys,json,math
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
root=Path('/lustre/fswork/projects/rech/vnc/upz25mj/projects/GRALE')
sys.path.insert(0,str(root))
from GRALE.main import GRALE_model
from GRALE.data.geometry import GeometricDataset
from GRALE.data.dataset import custom_collate_fn
from GRALE.loss.utils import pairwise_L1_norm

torch.set_num_threads(1)
data=GeometricDataset(root/'logs/geometric-continue-20261029/subsets/tiny-1234.json')
records=[]
for cfg in ['octa_straight','octa_straight_laplacian']:
 checkpoint=root/f'logs/geometric-continue-20261029/tiny/{cfg}/checkpoints/step-030000.ckpt'
 model=GRALE_model.load_from_checkpoint(str(checkpoint),map_location='cpu').eval()
 for n in [2,40]:
  ix=next(i for i,r in enumerate(data.samples) if r['nodes']==n)
  batch=custom_collate_fn([data[ix]])
  with torch.no_grad():
   enc,z=model.encoder(model.input_builder(batch.clone()))
   demb,out=model.decoder(z)
   P,_=model.matcher(enc,~batch.h,demb)
   real=P[0,:,:n]
   mass=real.sum(1)
   ids=torch.where(mass>.5)[0]
   gt=batch.nodes.features[0,:n]
   rowweights=real/mass[:,None].clamp_min(1e-30)
   bary=rowweights@gt
   a=model.matcher.linear_outputs(demb+model.matcher.positionnal_encoding_outputs)
   emb=enc.clone();emb[~batch.h]=model.matcher.node_padding_features[0,0]
   b=model.matcher.linear_inputs(emb)
   C=pairwise_L1_norm(a,b)[0]
   cost=C[ids,:n]
   centered=cost-cost.mean(0,keepdim=True)-cost.mean(1,keepdim=True)+cost.mean()
   signs=(a[0,ids,None]-b[0,None,:n]).sign()
   same_sign=((signs.amax((0,1))-signs.amin((0,1)))==0).sum().item()
   conditional=real/real.sum(0,keepdim=True)
   spread=lambda x:x.std(0,unbiased=False).square().sum().sqrt().item()*128
  xy=out.nodes.features[0].detach().requires_grad_(True)
  loss_soft=(((xy[:,None]-gt[None])**2).sum(-1)*real.detach()).sum()/n
  grad_soft=torch.autograd.grad(loss_soft,xy)[0]
  costs=((xy[ids,None].detach()-gt[None])**2).sum(-1).numpy()
  rows,cols=linear_sum_assignment(costs)
  loss_fixed=(xy[ids[rows]]-gt[cols]).square().sum()/n
  grad_fixed=torch.autograd.grad(loss_fixed,xy)[0]
  result={'config':cfg,'step':30000,'sample':data.samples[ix]['sample_id'],'n':n,
   'selected_output_rows':len(ids),'mean_max_assignment_weight':conditional.max(0).values.mean().item(),
   'uniform_reference_weight':1/n,'pred_spread_px':spread(xy[ids].detach()),'target_spread_px':spread(gt),
   'pred_vs_transport_target_average_px':((xy.detach()[ids]-bary[ids]).square().sum(-1).mean().sqrt()*128).item(),
   'encoder_node_embedding_spread':enc[0,:n].std(0,unbiased=False).norm().item(),
   'decoder_matching_embedding_spread':demb[0,ids].std(0,unbiased=False).norm().item(),
   'L1_constant_sign_dimensions':same_sign,'L1_total_dimensions':a.shape[-1],
   'raw_active_cost_doublecenter_max_abs':centered.abs().max().item(),
   'soft_coordinate_gradient_norm':grad_soft.norm().item(),
   'fixed_coordinate_gradient_norm':grad_fixed.norm().item(),
   'fixed_to_soft_gradient_ratio':grad_fixed.norm().item()/max(grad_soft.norm().item(),1e-30),
   'soft_coordinate_loss':loss_soft.item(),'fixed_coordinate_loss':loss_fixed.item()}
  records.append(result);print(json.dumps(result),flush=True)
Path('/tmp/grale_collapse_30k.json').write_text(json.dumps(records,indent=2))
