from NFLStudio.baseloss import NeRFBaseLoss
from NFLStudio.libs.render_results import get_colored_img, get_colored_img_binary
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import torch, os, math
from NFLStudio.libs.utils import makedirs, _EPS


class NerfLoss(NeRFBaseLoss):
    """
    NeRF loss and evaluation metrics
    """
    def __init__(self,configs):
        NeRFBaseLoss.__init__(self, configs)
        self.context_name = configs.datamanager_config.dataparser_config.context_name
        self.save_dir = os.path.join(configs.save_dir, f'test_results_{self.context_name}')
        makedirs(self.save_dir)

    def get_eikonal_loss(self, inputs, outputs, results, epoch):
        mask = torch.logical_or(inputs['static_mask'].bool(), inputs['vehicle_mask'].bool())
        eik_grad_coarse = outputs['gradient_sdf_coarse']
        results['eikonal'] =  eik_grad_coarse[mask].mean().half()
        results['dynamic_eikonal'] =  eik_grad_coarse[inputs['vehicle_mask'].bool()].mean()


    def get_sdf_loss(self, inputs, outputs, results, epoch):
        mask = torch.logical_or(inputs['static_mask'].bool(), inputs['vehicle_mask'].bool())
        dynamic_mask = inputs['vehicle_mask'].bool()
        dist_1 = inputs['first_dist'][mask] * self.extent #[N]
        dist_dynamic = inputs['first_dist'][dynamic_mask] * self.extent
        depth_from_vol = outputs['depth_vol_c'][mask] * self.extent  # depth from truncated volumetric rendering
        depth_dynamic = outputs['depth_vol_c'][dynamic_mask].detach() * self.extent

        results['depth_vol'] = self.l1_loss(depth_from_vol.half(), dist_1.half())

        depth_vol = outputs['depth_vol_c'][mask].detach() * self.extent
        error_1 = torch.abs(depth_vol - dist_1)
        results['depth_vol_mean'] =error_1.mean()
        results['depth_vol_median'] =error_1.median()
        results['depth_vol_recall'] = (error_1 < 1).float().mean()
        error_dynamic = torch.abs(depth_dynamic - dist_dynamic)
        results['dynamic_depth_vol_mean'] =error_dynamic.mean()
        results['dynamic_depth_vol_median'] =error_dynamic.median()
        results['dynamic_depth_vol_recall'] = (error_dynamic < 1).float().mean()

    def get_surface_loss(self, inputs, outputs, results):
        
        mask = torch.logical_or(inputs['static_mask'].bool(), inputs['vehicle_mask'].bool())
        surface_sdf = outputs['surface_sdf'][mask]*self.extent
        dynamic_surface_sdf = outputs['surface_sdf'][inputs['vehicle_mask'].bool()]*self.extent

        results['surface_sdf'] = self.l1_loss(surface_sdf.half(), torch.zeros_like(surface_sdf))
        results['dynamic_surface_sdf'] = self.l1_loss(dynamic_surface_sdf, torch.zeros_like(dynamic_surface_sdf))

    def get_vehicle_mask_loss(self, inputs, outputs, results):
        
        # mask = inputs['first_mask'].bool()
        predicted_vehicle_mask = outputs['predicted_vehicle_mask']
        gt_vehicle_mask = inputs['vehicle_mask']
        ones = torch.ones_like(predicted_vehicle_mask, requires_grad=False)
        intersect = torch.logical_and(predicted_vehicle_mask, gt_vehicle_mask)
        results['pos_recall'] = ones[intersect].sum() / ones[gt_vehicle_mask].sum() if ones[gt_vehicle_mask].sum() >0 else torch.tensor([1])
        results['pos_precision'] = ones[intersect].sum() / ones[predicted_vehicle_mask].sum() if ones[predicted_vehicle_mask].sum() >0 else torch.tensor([1])
        intersect = torch.logical_and(~predicted_vehicle_mask, ~gt_vehicle_mask)
        results['neg_recall'] = ones[intersect].sum() / ones[~gt_vehicle_mask].sum() if ones[~gt_vehicle_mask].sum()>0 else torch.tensor([1])
        results['neg_precision'] = ones[intersect].sum() / ones[~predicted_vehicle_mask].sum() if ones[~predicted_vehicle_mask].sum()>0 else torch.tensor([1])
        results['num_vehicle_points'] = ones[gt_vehicle_mask].sum()

    def get_intensity_loss(self, inputs, outputs, results, epoch):
        mask = torch.logical_or(inputs['static_mask'].bool(), inputs['vehicle_mask'].bool())
        gt_intensity = inputs['first_intensity'][mask].half()
        est_intensity = outputs['intensity'][mask]
        

        # 1) penalty term
        diff = (gt_intensity - est_intensity)
        # if diff.shape[0] == 0: diff = torch.tensor([0])
        i_l1 = torch.abs(diff).mean()
        i_l2 = (diff ** 2).mean()

        # 2）think about more meaningful error measurements here
        results['i_l1'] = i_l1
        results['i_l2'] = i_l2.half()
        results['i_med_rel_error'] = (torch.abs(gt_intensity - est_intensity) / gt_intensity).median()
        results['i_mean_rel_error'] = (torch.abs(gt_intensity - est_intensity) / gt_intensity).mean()
    
    def get_all_ray_drop_loss(self, inputs, outputs, results):
        ##ray drop for static nerf will neglect the vehicle points

        ray_hit_mask = outputs['hit_mask_all'].bool()
        ray_drop = ~ray_hit_mask
        if ray_drop.sum():
            ray_drop_map = outputs['raydrop_prob_all']  # the prob. to drop the ray
            ray_drop_stats = self.get_seg_loss(ray_drop.long(), ray_drop_map)

            results['rdrop_bce'] = ray_drop_stats['bce_loss']
            results['rdrop_ls'] = ray_drop_stats['lovasz_loss']
            results['rdrop_nll'] = ray_drop_stats['nll_loss']
            for key, value in ray_drop_stats['metric'].items():
                results[f'rdrop_{key}'] = torch.Tensor([value])   


    def get_loss_terms(self, inputs, outputs, phase, epoch):
        results = dict()
        if 'intensity' in outputs.keys():
            self.get_intensity_loss(inputs, outputs, results, epoch)
        if 'raydrop_prob_all' in outputs.keys():
            self.get_all_ray_drop_loss(inputs, outputs, results)
        if 'gradient_sdf_coarse' in outputs.keys():
            self.get_eikonal_loss(inputs, outputs, results, epoch)    
        if 'surface_sdf' in outputs.keys():
            self.get_surface_loss(inputs, outputs, results)
        if 'depth_vol_c' in outputs.keys():
            self.get_sdf_loss(inputs, outputs, results, epoch)
        if 'predicted_vehicle_mask' in outputs.keys():
            self.get_vehicle_mask_loss(inputs, outputs, results)

        # print("results_eikonal: ", results['eikonal'])
        # print("results_eikonal.shape: ", results['eikonal'].shape)
        device = inputs['first_intensity'].device
        ################################
        # 4. aggregate all the loss terms
        loss = torch.tensor([0.],requires_grad=True).float().to(device)
        for key, value in results.items():
            if torch.isnan(value):
                # print(key + 'has nan values')
                results[key] = torch.tensor([0.],requires_grad=True).float().to(device)
            else:
                if key in self.loss_weights and self.loss_weights[key] > 0:
                    loss += self.loss_weights[key] * value
                    # results[key] *= self.loss_weights[key] 
    
        # compute loss
        return loss, results
        

    def forward(self, outputs, inputs, phase, iter=0, epoch=0):
        results = {}
        rays_o, rays_dir, rays_dist = inputs['rays_o'], inputs['rays_d'], inputs['first_dist']
        mask = inputs['first_mask'].bool()
        device = rays_o.device
        loss = torch.tensor([0.],requires_grad=True).float().to(device)
        c_loss, c_results = self.get_loss_terms(inputs, outputs, phase, epoch)
        for key, value in c_results.items():
            results[f'{key}'] = value

        loss += c_loss
        results['loss'] = loss
        return {"loss": loss}, results