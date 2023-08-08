import torch
from torch import nn
from torch.nn import functional as F

hidden_dim = 36

class GraphConvolution(nn.Module):

    def __init__(self, input_dim, output_dim, num_features_nonzero,
                 dropout=0.5,
                 is_sparse_inputs=False,
                 bias=False,
                 activation = F.relu,
                 featureless=False):
        super(GraphConvolution, self).__init__()


        self.dropout = dropout
        self.bias = bias
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless
        self.num_features_nonzero = num_features_nonzero

        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))


    def forward(self, inputs):
        # print('inputs:', inputs)
        x, support = inputs
       
#         if self.training and self.is_sparse_inputs:
#             x = sparse_dropout(x, self.dropout, self.num_features_nonzero)
        if self.training:
            x = F.dropout(x, self.dropout)

        # convolve
        if not self.featureless: # if it has features x
            xw = torch.matmul(x, self.weight)
        else:
            xw = self.weight
        
        out = torch.matmul(xw, support)

        if self.bias is not None:
            out += self.bias

        return self.activation(out), support

class GCN(nn.Module):


    def __init__(self, input_dim, output_dim, num_features_nonzero):
        super(GCN, self).__init__()

        self.input_dim = input_dim # 1433
        self.output_dim = output_dim

        print('input dim:', input_dim)
        print('output dim:', output_dim)
        print('num_features_nonzero:', num_features_nonzero)


        self.layers = nn.Sequential(GraphConvolution(self.input_dim, hidden_dim, num_features_nonzero,
                                                              activation=F.relu,
                                                     dropout=0.5,
                                                     is_sparse_inputs=True),

                                    GraphConvolution(hidden_dim, output_dim, num_features_nonzero,
                                                     activation=F.relu,
                                                     dropout=0.5,
                                                     is_sparse_inputs=False),

                                    )

    def forward(self, inputs):
        x, support = inputs

        x = self.layers((x, support))

        return x
