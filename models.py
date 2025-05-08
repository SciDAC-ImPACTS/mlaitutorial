import torch
import torch.nn as nn
import numpy as np


###########################################################################################
# Define MLP layers
###########################################################################################
class MLPLayer(nn.Module):
    def __init__(self, in_features, out_features, activation_fn, bias=True, activate=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.act_fn = activation_fn
        self.activate = activate
    
    def forward(self, x):
        if self.activate:
            return self.act_fn(self.linear(x))
        return self.linear(x)

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim:int, num_layers: int, activation=nn.ReLU(), last_activation=nn.Sigmoid(), bias=True, activate_final=False):
        super().__init__()
        self.net = []
        self.net.append(MLPLayer(input_dim, hidden_dim, activation, bias=bias, activate=True))
        for i in range(num_layers-1):
            self.net.append(MLPLayer(hidden_dim, hidden_dim, activation, bias=bias, activate=True))
        self.net.append(MLPLayer(hidden_dim, output_dim, last_activation, bias=bias, activate=activate_final))
        
        self.net = nn.Sequential(*self.net)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.net(input)
        return output

###########################################################################################
# Define positional encoding for low spatial spectrum input
###########################################################################################
@torch.jit.script
def positional_encoding(v: torch.Tensor, sigma: float, m: int) -> torch.Tensor:
    j = torch.arange(m, device=v.device)
    coeffs = 2 * np.pi * sigma ** (j / m)
    vp = coeffs * torch.unsqueeze(v, -1)
    vp_cat = torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)
    return vp_cat.flatten(-2, -1)

class PositionalEncoding(nn.Module):
    def __init__(self, sigma: float, m: int):
        super().__init__()
        self.sigma = sigma
        self.m = m

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return positional_encoding(v, self.sigma, self.m)


###########################################################################################
# Adapted from SIREN notebook
# Link: https://github.com/vsitzmann/siren/blob/master/explore_siren.ipynb
###########################################################################################
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(coords)
        return output


###########################################################################################
# Feature Grid based INR
###########################################################################################

class DecompGrid(torch.nn.Module):
    '''
    grid_shape: [x_3d, y_3d, z_3d]
    '''
    def __init__(self, grid_shape, num_feats) -> None:
        super().__init__()
        
        self.grid_shape = grid_shape
        self.num_feats = num_feats
        self.feature_grids = []
        for i in range(len(grid_shape)):
            self.feature_grids.append(torch.nn.Parameter(
                torch.Tensor(1, num_feats, *reversed([grid_shape[i]]*3)),
                requires_grad=True
            ))
        self.feature_grids = torch.nn.ParameterList(self.feature_grids)
        for i in range(len(self.feature_grids)):
            torch.nn.init.uniform_(self.feature_grids[i], a=-0.0001, b=0.0001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        input: (Batch, Ndim)
        output: (Batch, num_feats)
        '''
        for i in range(len(self.feature_grids)):
            new_feats = torch.nn.functional.grid_sample(self.feature_grids[i],
                                x.reshape(([1]*x.shape[-1]) + list(x.shape)),
                                mode='bilinear', align_corners=True)
            new_feats = new_feats.squeeze()
            if i == 0:
                feats = new_feats.T
            else:
                feats = torch.cat((feats, new_feats.T), 1)
        return feats

class FeatureGrid(torch.nn.Module):
    def __init__(self, grid_shape, num_feats:int, hidden_nodes:int, num_layers: int, output_dim:int) -> None:
        super().__init__()
        self.dg = DecompGrid(grid_shape=grid_shape, num_feats=num_feats)
        self.mlp = MLP(input_dim=num_feats*len(grid_shape), hidden_dim=hidden_nodes, output_dim=output_dim, num_layers=num_layers, activate_final=True)

    def forward(self, x):
        x = self.dg(x)
        x = self.mlp(x)
        return x