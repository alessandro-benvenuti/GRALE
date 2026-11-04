"""Analytic assignment/gradient checks, plus optional pilot checkpoint replay."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from GRALE.data import BatchedDenseData
from GRALE.data.dataset import custom_collate_fn
from GRALE.data.geometry import GeometricDataset
from GRALE.loss.objective import GraphAutoencoderObjective
from GRALE.main import GRALE_model
from GRALE.model.geometric_assignment import GeometricAssignment, AssignmentForward
from scripts.run_spatial_anchor_pilot import forward as original_forward


def graph(nodes, h, adjacency, *, requires_grad=False):
    nodes = torch.as_tensor(nodes, dtype=torch.float32).unsqueeze(0)
    h = torch.as_tensor(h).unsqueeze(0)
    A = torch.as_tensor(adjacency, dtype=torch.float32).unsqueeze(0)
    if requires_grad:
        nodes.requires_grad_()
        A.requires_grad_()
    labels = torch.stack((torch.zeros_like(A), A), dim=-1)
    return BatchedDenseData(h, nodes={'features': nodes},
                           edges={'adjacency': A, 'labels': labels,
                                  'features': torch.zeros_like(A).unsqueeze(-1)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path)
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    target = graph([[.1,.2],[0,0],[.8,.9],[0,0]], [True,False,True,False],
                   [[0,0,1,0],[0,0,0,0],[1,0,0,0],[0,0,0,0]])
    output = graph([[0,0],[.2,.2],[.5,.5],[.8,.9]], [-5.,5.,-5.,5.], torch.zeros(4,4), requires_grad=True)
    P, _ = GeometricAssignment('hungarian')(output, target)
    assert P[0,1,0] == P[0,3,2] == 1
    assert not P.requires_grad
    assert torch.equal(P.sum(1), torch.ones(1,4,dtype=P.dtype))
    assert torch.equal(P.sum(2), torch.ones(1,4,dtype=P.dtype))
    alpha = dict(h=0,node_features=1,node_labels=0,adjacency=1,edge_features=0,edge_labels=0,marginals=0)
    loss, _ = GraphAutoencoderObjective(alpha)(output, target, P)
    loss.sum().backward()
    assert torch.allclose(output.nodes.features.grad[0,1], torch.tensor([.1/math.sqrt(2),0.],dtype=torch.float32),atol=1e-6)
    assert torch.count_nonzero(output.nodes.features.grad[0,[0,2]]) == 0
    assert output.edges.adjacency.grad[0,1,3] < 0
    assert output.edges.adjacency.grad[0,1,1] > 0
    assert torch.count_nonzero(output.edges.adjacency.grad[0,0]) == 0
    Q, _ = GeometricAssignment('fgw',alpha=0,candidate_count=4)(output,target)
    assert torch.equal(P,Q), 'Unique alpha-zero FGW must reproduce unary Hungarian'
    empty = graph(torch.zeros(4,2), [False]*4, torch.zeros(4,4))
    for method in ['hungarian','fgw']:
        Q,_ = GeometricAssignment(method)(output,empty)
        assert torch.equal(Q[0],torch.eye(4,dtype=Q.dtype))
    # Fifteen real targets exercise the represented-mass rounding case.
    xy = torch.stack((torch.linspace(.1,.9,15),torch.linspace(.8,.2,15)),dim=-1)
    target15 = graph(torch.cat((xy,torch.zeros(17,2))), [True]*15+[False]*17, torch.zeros(32,32))
    output32 = graph(torch.cat((xy,torch.full((17,2),2.))), [5.]*15+[-5.]*17, torch.zeros(32,32))
    Q,_ = GeometricAssignment('fgw',alpha=0)(output32,target15)
    assert torch.equal(Q[0,:15,:15],torch.eye(15,dtype=Q.dtype))
    # Graph structure must be able to resolve an otherwise constant unary cost.
    truth = graph(torch.zeros(4,2), [True,True,True,False],
                  [[0,1,1,0],[1,0,0,0],[1,0,0,0],[0,0,0,0]])
    logits = torch.tensor([[0,0,1,0],[0,0,1,0],[1,1,0,0],[0,0,0,0]],dtype=torch.float32)*30-15
    prediction = graph(torch.zeros(4,2), [0.]*4, logits)
    unary,_=GeometricAssignment('hungarian')(prediction,truth)
    structural,_=GeometricAssignment('fgw',alpha=.9,candidate_count=4)(prediction,truth)
    def error(plan):
        selected=plan[0,:,:3]
        return ((selected.T@logits.sigmoid()@selected-truth.edges.adjacency[0,:3,:3])**2).mean()
    assert error(structural) < error(unary)-.1
    print('Analytic orientation, padding, empty, gradient, alpha-zero, n=15 and structural checks passed')
    if args.pilot:
        assert args.manifest is not None
        dataset=GeometricDataset(args.manifest,'train',64)
        batch=custom_collate_fn([dataset[2],dataset[-1]])
        state=json.loads((args.pilot/'result.json').read_text())
        assert len({v['initial_parameter_sha256'] for v in state['arms'].values()})==1
        assert len({v['minibatch_order_sha256'] for v in state['arms'].values()})==1
        for method in ['learned','hungarian','fgw']:
            model=GRALE_model.load_from_checkpoint(str(args.pilot/method/'last.ckpt'),map_location='cpu').eval()
            with torch.no_grad():
                out,T,loss,_=AssignmentForward(method)(model,batch.clone())
                aligned,api_T=model(batch.clone(),logits=True,return_permutations=True)
                assert torch.equal(T,api_T)
                expected=out.clone();expected.permute_(T)
                assert torch.equal(expected.nodes.features,aligned.nodes.features)
                assert torch.equal(T,model.canonical_permutation(batch.clone()))
                model.log_dict=lambda *a,**k:None
                assert torch.equal(loss,model.training_step(batch.clone()))
                if method=='learned':
                    old=original_forward(model,batch.clone())
                    assert torch.equal(old[1],T) and torch.equal(old[2],loss)
            assert model.geometric_assignment is None if method=='learned' else model.geometric_assignment.method==method
        print('All three checkpoints reload, native/core APIs agree, default objective is unchanged, paired hashes match')


if __name__=='__main__':
    main()
