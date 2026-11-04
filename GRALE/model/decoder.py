import torch.nn as nn
from torch import Tensor
from GRALE.data import BatchedDenseData
from GRALE.attention import EvoformerDecoder, TransformerDecoder, EvoformerEncoder
from .utils import MLP_head, build_E_from_X, MLP
import torch
import math

class AbstractGraphDecoder(nn.Module):
    '''
    Abstract class for Decoder part of the Graph Autoencoder.
    '''

    def forward(self, data: BatchedDenseData):
        '''
        Input: graph_embedding of shape (batch_size, graph_embedding_size, graph_embedding_dim)
        Outputs: 
            node_embeddings of shape (batch_size, n_nodes_max, node_model_dim)
            output (BatchedDenseData) with attributes:
                outputs.h shape (batch_size, n_nodes_max) (node activations)
                outputs.nodes.labels shape (batch_size, n_nodes_max, node_labels_dim) (one-hot encoding of node labels)
                outputs.nodes.features shape (batch_size, n_nodes_max, n_node_features) (node features)
                outputs.edges.A (batch_size, n_nodes_max, n_nodes_max, edge_labels_dim) (one-hot encoding of edge labels)
                outputs.edges.labels (batch_size, n_nodes_max, n_nodes_max, edge_labels_dim) (one-hot encoding of edge labels)
                outputs.edges.features (batch_size, n_nodes_max, n_nodes_max, n_edge_features) (edge features)
        All outputs are logits.
        '''
        
        h = ...
        node_labels = ...
        node_features = ...
        node_embeddings = ...
        A = ...
        edge_labels = ...
        edge_features = ...
        
        
        outputs = BatchedDenseData(h = h,
                                   nodes={'labels': node_labels, 'features': node_features}, 
                                   edges={'labels': edge_labels, 'features': edge_features, 'adjacency': A})
        
        return node_embeddings, outputs
    
class EvoformerGraphDecoder(AbstractGraphDecoder):
    '''
    Initialize X and E with learnable queries
    X, E = EvoformerDecoder(X, E, graph_embedding)
    node_embeddings, node_labels,node_features = MLP(X)
    A, edge_labels, edge_features = MLP(E)
    
    Cost: O(M³+KM²) M = n_nodes_max, K = graph_embedding_size
    '''
    
    def __init__(self, 
                 node_labels_dim,
                 node_features_dim,
                 node_model_dim,
                 node_hidden_dim,
                 edge_labels_dim,
                 edge_features_dim,
                 edge_model_dim,
                 edge_hidden_dim,
                 graph_embedding_dim,
                 n_nodes_max,
                 n_heads,
                 n_layers,
                 dropout_attn,
                 dropout_skip,
                 dropout_mlp,
                 norm_post_or_pre='pre',
                 triangular_multiplication_mode='symmetric',
                 triangular_attention_mode='none',
                 geometric=False,
                 coordinate_init='default',
                 coordinate_anchor_seed=1234,
                 ):
        
        super().__init__()
        self.geometric = geometric
        
        self.EvoformerDecoder = EvoformerDecoder(n_layers=n_layers,
                                                 src_dim=graph_embedding_dim, 
                                                 trgt_node_dim=node_model_dim,
                                                 trgt_edge_dim=edge_model_dim,
                                                 trgt_node_hidden_dim=node_hidden_dim,
                                                 trgt_edge_hidden_dim=edge_hidden_dim,
                                                 trgt_size=n_nodes_max,
                                                 n_heads=n_heads, 
                                                 dropout_attn=dropout_attn,
                                                 dropout_skip=dropout_skip, 
                                                 dropout_mlp=dropout_mlp,
                                                 norm_post_or_pre=norm_post_or_pre,
                                                 triangular_multiplication_mode=triangular_multiplication_mode,
                                                 triangular_attention_mode=triangular_attention_mode)
        
        self.head_h = MLP_head(in_dim = node_model_dim, hidden_dim = None, out_dim = 1, dropout = 0.)
        self.head_node_embeddings = MLP_head(in_dim = node_model_dim, hidden_dim = node_hidden_dim, out_dim = node_model_dim, dropout = 0.)
        self.head_node_labels = MLP_head(in_dim = node_model_dim, hidden_dim = None, out_dim = node_labels_dim, dropout = 0.) if node_labels_dim else None
        self.head_node_features = MLP_head(in_dim = node_model_dim, hidden_dim = node_hidden_dim, out_dim = node_features_dim, dropout = 0.)
        self.head_A = MLP_head(in_dim = edge_model_dim, hidden_dim = None, out_dim = 1, dropout = 0.)
        self.head_edge_labels = MLP_head(in_dim = edge_model_dim, hidden_dim = None, out_dim = edge_labels_dim, dropout = 0.)
        self.head_edge_features = MLP_head(in_dim = edge_model_dim, hidden_dim = edge_hidden_dim, out_dim = edge_features_dim, dropout = 0.)
        # Optional spatial initialization for a controlled experiment. The default
        # adds no state-dict entries and retains the original decoder behavior.
        if coordinate_init not in ('default', 'center', 'jittered_grid'):
            raise ValueError(f'Unknown coordinate_init: {coordinate_init}')
        self.register_buffer('coordinate_anchors', None)
        self.register_buffer('coordinate_residual_scale', None)
        if coordinate_init != 'default':
            if not geometric or node_features_dim != 2:
                raise ValueError('Spatial coordinate initialization requires geometric XY outputs')
            anchors = torch.full((n_nodes_max, 2), .5)
            if coordinate_init == 'jittered_grid':
                side = math.isqrt(n_nodes_max)
                if side * side != n_nodes_max:
                    raise ValueError('Jittered-grid initialization requires a square slot capacity')
                rng = torch.Generator().manual_seed(coordinate_anchor_seed)
                y, x = torch.meshgrid(torch.arange(side), torch.arange(side), indexing='ij')
                anchors = (torch.stack([x, y], dim=-1).reshape(-1, 2)
                           + .1 + .8 * torch.rand(n_nodes_max, 2, generator=rng)) / side
                anchors = anchors[torch.randperm(n_nodes_max, generator=rng)]
            self.coordinate_anchors = anchors.unsqueeze(0)
            self.coordinate_residual_scale = torch.tensor(1.)
            final = self.head_node_features.mlp
            final = final if isinstance(final, nn.Linear) else final[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
    
    
    def forward(self, graph_embedding: Tensor):
        
        X, E = self.EvoformerDecoder(graph_embedding)
        
        h = self.head_h(X).squeeze(-1)
        node_embeddings = self.head_node_embeddings(X)
        node_features = self.head_node_features(X)
        if self.coordinate_anchors is not None:
            node_features = self.coordinate_anchors + self.coordinate_residual_scale * node_features
        A = self.head_A(E).squeeze(-1)
        edge_labels = self.head_edge_labels(E)
        edge_features = self.head_edge_features(E)
        nodes = {'features': node_features}
        if self.head_node_labels is not None:
            nodes['labels'] = self.head_node_labels(X)
        if self.geometric:
            A = (A + A.transpose(1, 2)) / 2
            edge_labels = (edge_labels + edge_labels.transpose(1, 2)) / 2
            edge_features = (edge_features + edge_features.transpose(1, 2)) / 2
        
        outputs = BatchedDenseData(h = h,
                                   nodes=nodes,
                                   edges={'labels': edge_labels, 'features': edge_features, 'adjacency': A})
        
        return node_embeddings, outputs

class EvoformerGraphDecoderV2(AbstractGraphDecoder):
    '''
    X = [learnable | latent ]
    E = [[learnable | learnable ],
         [learnable | learnable ]]
    X, E = EvoformerEncoder(X, E)
    X = X[:n_queries]
    E = E[:n_queries, :n_queries]
    node_embeddings, node_labels,node_features = MLP(X)
    A, edge_labels, edge_features = MLP(E)
    
    Cost: O((M+K)³) M = n_nodes_max, K = graph_embedding_size
    '''
    
    def __init__(self, 
                 node_labels_dim,
                 node_features_dim,
                 node_model_dim,
                 node_hidden_dim,
                 edge_labels_dim,
                 edge_features_dim,
                 edge_model_dim,
                 edge_hidden_dim,
                 graph_embedding_dim,
                 graph_embedding_size,
                 n_nodes_max,
                 n_heads,
                 n_layers,
                 dropout_attn,
                 dropout_skip,
                 dropout_mlp,
                 norm_post_or_pre='pre',
                 triangular_multiplication_mode='symmetric',
                 triangular_attention_mode='none'
                 ):
        
        super().__init__()
        
        self.graph_embedding_size = graph_embedding_size
        self.n_nodes_max = n_nodes_max
        
        self.MLP_graph = MLP(graph_embedding_dim, node_hidden_dim, node_model_dim, dropout_mlp)
        
        self.queries_nodes = nn.Parameter(torch.rand(1,n_nodes_max, node_model_dim))
        self.queries_edges = nn.Parameter(torch.rand(1,n_nodes_max+graph_embedding_size,n_nodes_max+graph_embedding_size, edge_model_dim))
        
        self.EvoformerEncoder = EvoformerEncoder(n_layers=n_layers,
                                                node_dim=node_model_dim, 
                                                edge_dim=edge_model_dim,
                                                node_hidden_dim=node_hidden_dim,
                                                edge_hidden_dim=edge_hidden_dim,
                                                n_heads=n_heads, 
                                                dropout_attn=dropout_attn,
                                                dropout_skip=dropout_skip, 
                                                dropout_mlp=dropout_mlp,
                                                norm_post_or_pre=norm_post_or_pre,
                                                triangular_multiplication_mode=triangular_multiplication_mode,
                                                triangular_attention_mode=triangular_attention_mode)
        
        self.head_h = MLP_head(in_dim = node_model_dim, hidden_dim = None, out_dim = 1, dropout = 0.)
        self.head_node_embeddings = MLP_head(in_dim = node_model_dim, hidden_dim = node_hidden_dim, out_dim = node_model_dim, dropout = 0.)
        self.head_node_labels = MLP_head(in_dim = node_model_dim, hidden_dim = None, out_dim = node_labels_dim, dropout = 0.)
        self.head_node_features = MLP_head(in_dim = node_model_dim, hidden_dim = node_hidden_dim, out_dim = node_features_dim, dropout = 0.)
        self.head_A = MLP_head(in_dim = edge_model_dim, hidden_dim = None, out_dim = 1, dropout = 0.)
        self.head_edge_labels = MLP_head(in_dim = edge_model_dim, hidden_dim = None, out_dim = edge_labels_dim, dropout = 0.)
        self.head_edge_features = MLP_head(in_dim = edge_model_dim, hidden_dim = edge_hidden_dim, out_dim = edge_features_dim, dropout = 0.)
    
    
    def forward(self, graph_embedding: Tensor):
        batchsize = graph_embedding.shape[0]
        
        X = torch.cat([self.queries_nodes.repeat(batchsize,1,1), self.MLP_graph(graph_embedding)], dim=1)
        E = self.queries_edges.repeat(batchsize,1,1,1)
        
        X, E = self.EvoformerEncoder(X = X, E = E)
        
        X = X[:,:self.n_nodes_max,:]
        E = E[:,:self.n_nodes_max,:self.n_nodes_max,:]

        h = self.head_h(X).squeeze(-1)
        node_embeddings = self.head_node_embeddings(X)
        node_labels = self.head_node_labels(X)
        node_features = self.head_node_features(X)
        A = self.head_A(E).squeeze(-1)
        edge_labels = self.head_edge_labels(E)
        edge_features = self.head_edge_features(E)
        
        outputs = BatchedDenseData(h = h,
                                   nodes={'labels': node_labels, 'features': node_features}, 
                                   edges={'labels': edge_labels, 'features': edge_features, 'adjacency': A})
        
        return node_embeddings, outputs
    
    
import inspect
def get_decoder(config):
    decoder_model = config.get('decoder_model') 
    if decoder_model == 'evoformer':
        DecoderClass = EvoformerGraphDecoder
    elif decoder_model == 'evoformer2':
        DecoderClass = EvoformerGraphDecoderV2
    else:
        print("Warning: decoder_model not recognized, using 'evoformer' by default.")
        DecoderClass = EvoformerGraphDecoder
    params = inspect.signature(DecoderClass).parameters
    valid_args = {k: v for k, v in config.items() if k in params}
    return DecoderClass(**valid_args)
