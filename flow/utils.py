import torch.nn as nn
from flow.graph_models import GraphConvEmbeddedGaussian, EGGC, GAT

class MultiInputSequential(nn.Sequential):
    def forward(self, *input):
        multi_inp = False
        if len(input) > 1:
            multi_inp = True
            _, edge_index = input[0], input[1]

        for module in self._modules.values():
            if multi_inp:
                if hasattr(module, "weight"):
                    input = [module(*input)]
                else:
                    # Only pass in the features to the Non-linearity
                    input = [module(input[0]), edge_index]
            else:
                input = [module(*input)]
        return input[0]


kwargs_layer = {
    "Linear": nn.Linear,
    "GraphConvEmbeddedGaussian": GraphConvEmbeddedGaussian,
    "EGGC": EGGC,
    "GAT": GAT,
}


def create_coupling_blocks(
    input_size, hidden_size, n_blocks, n_hidden, layer_type="Linear"
):
    nets, nett = [], []
    for _ in range(n_blocks):
        block_nets = [kwargs_layer[layer_type](input_size, hidden_size)]
        block_nett = [kwargs_layer[layer_type](input_size, hidden_size)]
        for _ in range(n_hidden):
            block_nets += [
                nn.Tanh(),
                kwargs_layer[layer_type](hidden_size, hidden_size),
            ]
            block_nett += [
                nn.Tanh(),
                kwargs_layer[layer_type](hidden_size, hidden_size),
            ]
        block_nets += [
            nn.Tanh(),
            kwargs_layer[layer_type](hidden_size, input_size),
            nn.Tanh(),
        ]
        block_nett += [nn.Tanh(), kwargs_layer[layer_type](hidden_size, input_size)]
        nets += [MultiInputSequential(*block_nets)]
        nett += [MultiInputSequential(*block_nett)]

    


    s = nets = MultiInputSequential(*nets)
    t = nett = MultiInputSequential(*nett)
    return s, t



