from nerfstudio.fields.base_field import *
import math
import tinycudann as tcnn
from NFLStudio import raymarching
import torch.nn.functional as F
from NFLStudio.libs.activation import trunc_exp
from NFLStudio.libs.utils import _EPS
from copy import deepcopy
from nerfstudio.field_components import encodings, mlp
from nerfstudio.fields import sdf_field
from typing import Union
from nerfstudio.field_components.encodings import Encoding, expected_sin, NeRFEncoding
import numpy as np
import torch
class NFLField(Field): ## with SDF
    def __init__(self,
                 encoding_sdf="HashGrid",
                 encoding_dir="SphericalHarmonics",
                 num_layers_sdf=3,
                 num_layers_reflectance = 3,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 per_level_scale = 2.0,
                 bound = 1.0,
                 extent = 100,  # scene extent
                 ):
        super().__init__()
        self.bound = bound
        self.scene_extent  = extent
        print("extent: ", extent)
        self.hidden_dim = hidden_dim
        self.geo_feat_dim =  geo_feat_dim

        encoding_config ={
            "otype": encoding_sdf,
            "n_levels": 16,
            "n_features_per_level": 2,
            "log2_hashmap_size": 19,
            "base_resolution": 16,
            "per_level_scale": per_level_scale,
        }
        self.n_output_dim = 32
        
        # positional encoding
        self.encoder_sdf= tcnn.Encoding(
            n_input_dims=3,
            encoding_config=encoding_config,
            dtype=torch.float32
        )
        sdfnetwork_config={
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": hidden_dim,
            "n_hidden_layers": num_layers_sdf - 1,
        }

        # sdf network
        self.sdf_net = mlp.MLP(in_dim=self.n_output_dim, num_layers=num_layers_sdf, layer_width=hidden_dim,
                               out_dim=1+self.geo_feat_dim)


        beta_init = 0.1
        self.deviation_network = sdf_field.LearnedVariance(init_val=beta_init)
        self._cos_anneal_ratio = 1.0
        # view-direction encoding
        self.encoder_dir = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": encoding_dir,
                "degree": 4,
            },
            dtype = torch.float32 
        )
        # intensity network
        self.intensity_net = mlp.MLP(in_dim=self.geo_feat_dim + self.encoder_dir.n_output_dims, num_layers=num_layers_reflectance, 
                                        layer_width=hidden_dim, out_dim=1)
        # ray drop network
        self.ray_drop_net = mlp.MLP(in_dim=self.geo_feat_dim + self.encoder_dir.n_output_dims, num_layers=num_layers_reflectance, 
                                        layer_width=hidden_dim, out_dim=2)

    def hash_encoding(self, x):
        """
        Input:
            x:      [N, 3]
        Output:
        s   sigma:  [N]
            x_pos:  [N, C]
        """
        # x: [N, 3], in [-bound, bound]
        x = (x + self.bound) / (2 * self.bound) # to [0, 1]
        x_pos = self.encoder_sdf(x)
        return x_pos


    def finite_difference_normals_approximator(self, x, bound=1, epsilon = 0.005, sdf=None):
        # finite difference
        # f(x+h, y, z), f(x, y+h, z), f(x, y, z+h) - f(x-h, y, z), f(x, y-h, z), f(x, y, z-h)
        pos_x = x + torch.tensor([[epsilon, 0.00, 0.00]], device=x.device)
        dist_dx_pos = self.pos_2_sdf(pos_x.clamp(-bound, bound))
        dist_dx_pos = dist_dx_pos[:,None]
        pos_y = x + torch.tensor([[0.00, epsilon, 0.00]], device=x.device)
        dist_dy_pos = self.pos_2_sdf(pos_y.clamp(-bound, bound))
        dist_dy_pos = dist_dy_pos[:,None]
        pos_z = x + torch.tensor([[0.00, 0.00, epsilon]], device=x.device)
        dist_dz_pos= self.pos_2_sdf(pos_z.clamp(-bound, bound))
        dist_dz_pos = dist_dz_pos[:,None]

        neg_x = x + torch.tensor([[-epsilon, 0.00, 0.00]], device=x.device)
        dist_dx_neg = self.pos_2_sdf(neg_x.clamp(-bound, bound))
        dist_dx_neg = dist_dx_neg[:,None]
        neg_y = x + torch.tensor([[0.00, -epsilon, 0.00]], device=x.device)
        dist_dy_neg = self.pos_2_sdf(neg_y.clamp(-bound, bound))
        dist_dy_neg = dist_dy_neg[:,None]
        neg_z = x + torch.tensor([[0.00, 0.00, -epsilon]], device=x.device)
        dist_dz_neg = self.pos_2_sdf(neg_z.clamp(-bound, bound))
        dist_dz_neg = dist_dz_neg[:,None]

        return torch.cat([0.5*(dist_dx_pos - dist_dx_neg) / epsilon, 0.5*(dist_dy_pos - dist_dy_neg) / epsilon, 0.5*(dist_dz_pos - dist_dz_neg) / epsilon], dim=-1)
    
    def get_sdf(self, ray_samples: RaySamples):
        """predict the sdf value for ray samples"""
        positions = ray_samples.frustums.get_start_positions()
        positions_flat = positions.view(-1, 3)
        hidden_output = self.forward_geonetwork(positions_flat).view(*ray_samples.frustums.shape, -1)
        sdf, _ = torch.split(hidden_output, [1, self.geo_feat_dim], dim=-1)
        return sdf

    # optimizer utils
    def get_params(self, lr):
        params = [
            {'params': self.encoder_sdf.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.intensity_net.parameters(), 'lr': lr},
            {'params': self.ray_drop_net.parameters(),'lr': lr},
            {'params': self.two_return_net.parameters(), 'lr': lr}
        ]
        
        return params

    def forward_geonetwork(self, inputs):
        """forward the geonetwork"""
        x_pos = self.hash_encoding(inputs)
        h = self.sdf_net(x_pos)

        return h

    def pos_2_sdf(self, inputs):
        h = self.forward_geonetwork(inputs)
        sdf, _ = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        return sdf
    
    def set_cos_anneal_ratio(self, value):
        self._cos_anneal_ratio = value

    def get_alpha(
        self,
        xyzs: Optional[Float[Tensor, "...  num_samples 3"]] = None,
        directions: Optional[Float[Tensor, "...  1  3"]] = None,
        deltas = None,
        sdf: Optional[Float[Tensor, "... num_samples 1"]] = None,
        gradients: Optional[Float[Tensor, "... num_samples  3"]] = None,
        )-> Float[Tensor, "... num_samples  1"]:

        
        inv_s = self.deviation_network.get_variance()  # Single parameter

        true_cos = (directions * gradients).sum(-1, keepdim=True)

        # anneal as NeuS
        cos_anneal_ratio = self._cos_anneal_ratio

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(
            F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) + F.relu(-true_cos) * cos_anneal_ratio
        )  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * deltas * 0.5
        estimated_prev_sdf = sdf - iter_cos * deltas * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf ** 2 - next_cdf ** 2
        c = prev_cdf ** 2

        alpha = ((p + 1e-6) / (c + 1e-6)).clip(0.0, 1.0)

        return alpha

    def _forward(self, ray_samples: RaySamples, rays_batch=None):
        results = {}
        rays_o  = ray_samples.frustums.origins[:,0,:]
        rays_d  = ray_samples.frustums.directions[:,0,:]
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        N = rays_o.size(0) 
        device = rays_o.device
        # generate xyzs
        z_vals =ray_samples.frustums.starts.squeeze()
        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1) # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
        N, T, _ = xyzs.size()
        # query sdf
        xyzs = xyzs.view(-1,3)
        h = self.forward_geonetwork(xyzs)
        sdf, geo_feat = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        # calculate numerical gradient
        gradient_sdf_coarse = self.finite_difference_normals_approximator(xyzs.view(-1,3), epsilon=1e-3 / self.scene_extent, sdf=sdf)
        gradient_sdf_coarse = gradient_sdf_coarse.view(N,T,3)
        # calculate alpha and weights using active sensing formulation
        sample_dist = ray_samples.deltas
        alphas = self.get_alpha(xyzs=None, sdf=sdf.view(N,T,1), directions=rays_d[:,None,:], 
                                gradients=gradient_sdf_coarse, deltas=sample_dist)
        weights = ray_samples.get_weights_and_transmittance_from_alphas(alphas, True).squeeze(-1)
        weights = weights / (torch.sum(weights, dim=1, keepdim=True) + _EPS) 
        # render depth
        depth_vol_c = (weights * z_vals).sum(1)
        #intensity
        dirs = (rays_d  + 1) / 2
        dirs_embedding = self.encoder_dir(dirs)
        input_mat = torch.cat([dirs_embedding.repeat_interleave(T, 0), geo_feat], dim=1)
        intensity = self.intensity_net(input_mat)
        intensity = intensity.view(N, T, -1).squeeze()  # [N, T]
        intensity = (intensity * weights).sum(1)
        #surface points' SDF
        xyzs_surface = rays_o + rays_d * rays_batch['first_dist'][:,None]
        h = self.forward_geonetwork(xyzs_surface.view(-1,3))
        surface_sdf, _ = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        # ray drop
        ray_drop_prob = self.ray_drop_net(input_mat)
        ray_drop_prob = torch.clamp(ray_drop_prob, min=-1000, max=1000)
        ray_drop_prob = ray_drop_prob.view(-1, T, 2).squeeze() #[N, T, 2]
        ray_drop_vol = (ray_drop_prob * weights[:,:,None]).sum(1) #[N, 2]


        results = {
            'depth_vol_c': depth_vol_c,
            'ray_drop_prob': ray_drop_vol,
            'gradient_sdf_coarse': ((gradient_sdf_coarse.norm(2, dim=-1) - 1) ** 2).mean(-1),      
            'surface_sdf': surface_sdf.contiguous(),
            'intensity': intensity,
        }

        return results

    def forward(self, ray_samples: RaySamples, compute_normals: bool = False, rays_batch=None):
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        results = self._forward(ray_samples, rays_batch=rays_batch)
        return results