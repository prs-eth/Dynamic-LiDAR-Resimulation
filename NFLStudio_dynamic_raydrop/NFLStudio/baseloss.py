import torch, os, sys, math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from NFLStudio.libs.utils import get_cdf, to_array, makedirs, _EPS

from NFLStudio.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist

import matplotlib.pyplot as plt
from NFLStudio.libs.metrics import compute_iou
from NFLStudio.libs.lovasz_softmax import Lovasz_softmax
from NFLStudio.libs.render_results import lidar_to_range_img
# from NFLStudio.nflmodel import (
#     NFLModelConfig
# )

def get_ray_variance(ray_dist, scale, ray_max = 75 , sensor_acc = 0.02, k = 0.8):
    """
    maximum target std is: 0.02**0.8 = 0.044 
    """
    std = (ray_dist / ray_max * sensor_acc / scale)**k 
    ray_var = std.pow(2)
    return ray_var


def get_ce_weights(gt_label, n_classes, max_weights = 50):
    # get inverse_frequency of each class from ground truth label
    counts =[]
    device = gt_label.device
    for label in range(n_classes):
        counts.append((gt_label == label).sum().item()+_EPS)
    counts = torch.tensor(counts).to(device)
    inv_freq = counts.sum() / counts
    seg_weight = torch.clamp(torch.sqrt(inv_freq), 0, max_weights)
    return seg_weight  


class TukeyLoss(nn.Module):
    def __init__(self,c=4.685,normalized=False) -> None:
        super().__init__()
        self.c=c
        self.normalized = normalized
    def forward(self, y1,y2):
        error = y1 - y2
        if not self.normalized:
            mean = error.mean()
            std =torch.sqrt(error.var())
            error = (error - mean) / std
        abs_error = torch.abs(error)
        mask_lessc = abs_error<self.c
        abs_error[mask_lessc] = self.c*self.c / 6.0
        error_gtc = abs_error[~mask_lessc]/self.c
        tmp = 1 - error_gtc*error_gtc
        error = self.c *self.c / 6.0 * tmp*tmp*tmp
        abs_error[~mask_lessc] = error
        return abs_error.mean()




class NeRFBaseLoss(nn.Module):
    """
    NeRF loss and evaluation metrics
    """
    def __init__(self,configs):
        super(NeRFBaseLoss,self).__init__()
        self.device = configs.device
        self.extent = configs.extent
        self.center =  torch.Tensor(configs.center).to(self.device)

        self.loss_weights = configs.loss
        self.clamp_eps = configs.loss['clamp_eps']

        self.mse_loss = nn.MSELoss(reduction = 'mean')
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.gnll_loss = nn.GaussianNLLLoss(eps=self.clamp_eps)  # MSE loss between input and target, while pushing down the variance
        self.chamfer_loss = chamfer_3DDist()
        self.tukey_loss = TukeyLoss()
        self.huber_loss = nn.HuberLoss()

        self.reweight = configs.loss['reweight']
        self.softmax = nn.Softmax(dim = 1)
        self.lovasz_loss = Lovasz_softmax()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.nllloss = nn.NLLLoss()

        self.max_std = 0.4
        self.min_std = 0.075

    def get_seg_loss(self, gt, est):
        """segmentation loss
        Args:
            gt (tensor): [B] long tens
            est (tensor): [B, C]
        """
        stats = dict()
        
        # compute weights in an online fashion
        if self.reweight:
            seg_weights = get_ce_weights(gt, 2)
            criterion = torch.nn.CrossEntropyLoss(weight=seg_weights, ignore_index=-1)
        else:
            criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

        # 1. compute weighted ce loss and lovasz softmax loss
        ce_loss = criterion(est, gt)
        score_softmax = self.softmax(est)
        lovasz_loss = self.lovasz_loss(score_softmax, gt)
        
        score_logsoftmax = self.logsoftmax(est)
        nll_loss = self.nllloss(score_logsoftmax, gt)
        
        stats['bce_loss'] = ce_loss
        stats['lovasz_loss'] = lovasz_loss
        stats['nll_loss'] = nll_loss

        # 2. update intersection, union, recall, precision, 
        predictions = est.argmax(1)
        stats['metric'] = compute_iou(predictions, gt, 2, -1)
        
        return stats

    def get_depth_loss(self, inputs, outputs, results):
        """
        Compute the depth loss, we compute both L1 loss and L2 loss
        """
        first_mask = inputs['first_mask'].bool()
        rays_dist, est_dist = inputs['first_dist'][first_mask], outputs['depth'][first_mask]

        depth_l1 = self.l1_loss(est_dist * self.extent, rays_dist * self.extent)
        depth_mse = self.mse_loss(est_dist * self.extent, rays_dist * self.extent)
        results['d_l1'] = depth_l1
        results['d_l2'] = depth_mse

    def get_intensity_loss(self, inputs, outputs, results):
        """
        compute the loss terms over the intensity map
        We use the l1 loss
        """
        mask = inputs['first_mask'].bool()
        gt_intensity = inputs['first_intensity'][mask]
        est_intensity = outputs['intensity'][mask]

        # 1) penalty term
        diff = (gt_intensity - est_intensity)
        i_l1 = torch.abs(diff).mean()
        i_l2 = (diff ** 2).mean()

        # 2）think about more meaningful error measurements here
        results['i_l1'] = i_l1
        results['i_l2'] = i_l2
        results['i_med_rel_error'] = (torch.abs(gt_intensity - est_intensity) / gt_intensity).median()
        results['i_mean_rel_error'] = (torch.abs(gt_intensity - est_intensity) / gt_intensity).mean()


    def get_second_depth_loss(self, inputs, outputs, results, phase):
        """
        Compute the depth loss, we compute both L1 loss and L2 loss
        """
        second_mask = inputs['second_mask'].bool()
        # second_mask_est = outputs['two_return_prob'].argmax(1) == 1
        second_mask_est = inputs['second_mask'].bool()
        if phase == 'train':
            mask = second_mask
        else:
            mask = torch.logical_and(second_mask, second_mask_est)
        if mask.sum():
            rays_dist, est_dist = inputs['second_dist'][mask], outputs['second_depth'][mask]

            depth_l1 = self.l1_loss(est_dist * self.extent, rays_dist * self.extent)
            depth_mse = self.mse_loss(est_dist * self.extent, rays_dist * self.extent)
            results['d_l1_2'] = depth_l1
            results['d_l2_2'] = depth_mse

    def get_second_intensity_loss(self, inputs, outputs, results, phase):
        """
        compute the loss terms over the intensity map
        We use the l1 loss
        """
        second_mask = inputs['second_mask'].bool()
        second_mask_est = outputs['two_return_prob'].argmax(1) == 1
        if phase == 'train':
            mask = second_mask
        else:
            mask = torch.logical_and(second_mask, second_mask_est)
        if mask.sum():
            gt_intensity = inputs['second_intensity'][mask]
            est_intensity = outputs['second_intensity'][mask]

            # 1) penalty term
            diff = (gt_intensity - est_intensity)
            i_l1 = torch.abs(diff).mean()
            i_l2 = (diff ** 2).mean()

            # 2）think about more meaningful error measurements here
            results['i_l1_2'] = i_l1
            results['i_l2_2'] = i_l2
            results['i_med_rel_error_2'] = torch.median(torch.abs(gt_intensity - est_intensity) / gt_intensity)
            results['i_mean_rel_error_2'] = (torch.abs(gt_intensity - est_intensity) / gt_intensity).mean()

    def get_ray_drop_loss(self, inputs, outputs, results):
        ray_hit = inputs['first_mask'].bool()
        ray_drop = ~ray_hit
        if ray_drop.sum():
            ray_drop_map = outputs['ray_drop_prob']  # the prob. to drop the ray
            ray_drop_stats = self.get_seg_loss(ray_drop.long(), ray_drop_map)

            results['rdrop_bce'] = ray_drop_stats['bce_loss']
            results['rdrop_ls'] = ray_drop_stats['lovasz_loss']
            results['rdrop_nll'] = ray_drop_stats['nll_loss']
            for key, value in ray_drop_stats['metric'].items():
                results[f'rdrop_{key}'] = torch.Tensor([value])

    def get_two_return_loss(self, inputs, outputs, results):
        ray_hit = inputs['second_mask'].bool()
        if ray_hit.sum():
            second_return_map = outputs['two_return_prob']  # the prob. to drop the ray
            second_return_stats = self.get_seg_loss(ray_hit.long(), second_return_map)

            results['two_return_bce'] = second_return_stats['bce_loss']
            results['two_return_ls'] = second_return_stats['lovasz_loss']
            results['two_return_nll'] = second_return_stats['nll_loss']
            for key, value in second_return_stats['metric'].items():
                results[f'two_return_{key}'] = torch.Tensor([value])

    def get_normal_loss(self, inputs, outputs, results):
        """
        Compute the loss terms over the normal map
        1). reg term:         the normal direction is towards the ray direction
        2). penalty terms:    MSE loss over the predicted normal at ground truth point
        3). smooth terms:     MSE loss over the predicted normal at expected point
        """
        mask = inputs['first_mask'].bool()
        rays_dir = inputs['rays_d'][mask]  #[N,3]
        normal_est, = outputs['normals'][mask], 
        normal_gt = inputs['rays_normal'][mask]

        # 1) reg term
        inner_product = (normal_est * rays_dir).sum(-1)
        clamp_ip = torch.clamp_min(inner_product, 0)**2
        normal_reg = clamp_ip.mean()

        # 2) penalty terms
        diff_normal = normal_est - normal_gt
        normal_penalty = (diff_normal**2).sum(-1).mean()

        results['n_penalty'] = normal_penalty
        results['n_reg'] = normal_reg
        results['n_corr'] = (normal_est * normal_gt).sum(-1).mean()   


    def get_radiant_loss(self, inputs, outputs, results, dist_threshold = 2.0, min_threshold = 0.1, max_threshold = 10):
        """
        The std. decays from 0.5 to 0.1 meter, the corresponding max weights increases from 0.8 to 4.0
        The idea is to:
        push the radiants before the first peak bellow min_threshold
        push the weighted radius around the first peak above the max_threshold

        """
        first_mask = inputs['first_mask'].bool()
        second_mask = inputs['second_mask'].bool()
        dist_1 = inputs['first_dist'] * self.extent #[N]
        dist_2 = inputs['second_dist'] * self.extent #[N]
        dist_1[~first_mask] = -300
        dist_2[~second_mask] = -300

        z_vals = outputs['z_vals'] * self.extent #[N, T]
        radiant = outputs['radiants']
        decay_rate = outputs['decay_rate']
        std = self.max_std * (self.min_std / self.max_std) ** decay_rate

        # compute gt weights by assuming a Gaussian distribution
        scale = 1 / math.sqrt(2 * math.pi) / std
        gt_prob_1 = torch.exp(- (z_vals - dist_1[:,None])**2 / 2 / std**2) * scale
        gt_prob_2 = torch.exp(- (z_vals - dist_2[:,None])**2 / 2 / std**2) * scale

        # set the weights outside 2-sigma to be zero
        threshold = math.exp(-2) * scale
        gt_prob_1[gt_prob_1 < threshold] = 0.
        gt_prob_2[gt_prob_2 < threshold] = 0.

        sum_radiants_1 = (gt_prob_1 * radiant).sum(1)
        sum_radiants_2 = (gt_prob_2 * radiant).sum(1)

        # hinge loss over the peak
        loss_radiant_1 = F.relu(max_threshold - sum_radiants_1)[first_mask].mean()
        loss_radiant_2 = F.relu(max_threshold - sum_radiants_2)[second_mask].mean()

        # hinge loss for radiants before the first peak or between two peaks, and after the last peak
        cut_point_1 = dist_1 - 2 * std
        cut_point_2 = dist_1 + 2 * std
        cut_point_3 = dist_2 - 2 * std
        cut_point_4 = dist_2 + 2 * std
        cut_point_5 = torch.cat((cut_point_2[:,None], cut_point_4[:,None]), dim=1).max(1)[0] # take the right most point

        mask_1 = torch.logical_and(z_vals < cut_point_3[:,None], z_vals > cut_point_2[:,None])
        mask_2 = torch.logical_or(z_vals < cut_point_1[:,None], z_vals > cut_point_5[:,None])
        mask = torch.logical_or(mask_1, mask_2)
        mask = torch.logical_and(mask, first_mask[:,None]) # we only supervise points with at least one return
        loss_radiant_3 = F.relu(radiant - min_threshold)[mask].mean()

        loss_radiant_up = loss_radiant_1 + loss_radiant_2
        loss_radian_down = loss_radiant_3
        results['radiant_up'] = loss_radiant_up
        results['radian_down'] = loss_radian_down

        # get the accuracy of peak detection
        idx = torch.argmax(radiant, dim=1)
        depth_from_peak = torch.gather(z_vals, dim=1, index=idx[:,None]).squeeze()
        error_1 = torch.min(torch.cat((torch.abs(depth_from_peak - dist_1)[:,None], torch.abs(depth_from_peak - dist_2)[:,None]), dim=1), dim=1)[0][first_mask]
        results['depth_from_peak'] =error_1.mean()

        # get recall of volumetric rendering
        depth_from_vol = outputs['depth'] * self.extent
        error_2 = torch.min(torch.cat((torch.abs(depth_from_vol - dist_1)[:,None], torch.abs(depth_from_vol - dist_2)[:,None]), dim=1), dim=1)[0][first_mask]
        results['depth_vol'] =error_2.mean()


        recall_1 = (error_1 < dist_threshold).float().mean()
        results['recall_peak_detection'] = recall_1
        recall_2 = (error_2 < dist_threshold).float().mean()
        results['recall_vol_render'] = recall_2

    def get_reflectivity_loss(self, inputs, outputs, results):
        """
        We add global entropy to the reflectivity, but only to rays with returns
        """
        first_mask = inputs['first_mask'].bool()
        reflectivity = outputs['reflectivity'] #[N, T]
        n_gaussian_rays = int(reflectivity.size(0) / first_mask.size(0))
        first_mask = first_mask.repeat_interleave(n_gaussian_rays)
        reflectivity = reflectivity[first_mask]

        reflectivity = self.softmax(reflectivity) #[N, T] #[between 0 and 1] 
        entropy = -reflectivity * torch.log2(reflectivity + _EPS)
        entrop_loss = entropy.sum(1).mean(0)
        results['ref_entropy'] = entrop_loss        

    def get_ray_entropy_loss(self, inputs, outputs, results):
        mask = inputs['first_mask'].bool()
        opacity = outputs['opacity'][mask]
        normalised_opacity = opacity / (torch.sum(opacity, dim=1, keepdim=True) + _EPS)  #[between 0 and 1]
        entropy = -normalised_opacity * torch.log2(normalised_opacity + _EPS)
        entrop_loss = entropy.sum(1).mean(0)
        results['ray_entropy'] = entrop_loss


    def get_loss_terms(self, inputs, outputs, phase):
        results = dict()
        if 'depth' in outputs.keys():
            self.get_depth_loss(inputs, outputs, results)
        if 'second_depth' in outputs.keys():
            self.get_second_depth_loss(inputs, outputs, results, phase)
        if 'intensity' in outputs.keys():
            self.get_intensity_loss(inputs, outputs, results)
        if 'second_intensity' in outputs.keys():
            self.get_second_intensity_loss(inputs, outputs, results, phase)
        if 'ray_drop_prob' in outputs.keys():
            self.get_ray_drop_loss(inputs, outputs, results)
        if 'two_return_prob' in outputs.keys():
            self.get_two_return_loss(inputs, outputs, results)
        if 'normals' in outputs.keys():
            self.get_normal_loss(inputs, outputs, results)
        if 'radiants' in outputs.keys():
            self.get_radiant_loss(inputs, outputs, results)
        if 'reflectivity' in outputs.keys():
            self.get_reflectivity_loss(inputs, outputs, results)

        device = inputs['first_intensity'].device
        ################################
        # 4. aggregate all the loss terms
        loss = torch.tensor([0.],requires_grad=True).float().to(device)
        for key, value in results.items():
            if torch.isnan(value):
                print(key + 'has nan values')
                results[key] = torch.tensor([0.],requires_grad=True).float().to(device)
            else:
                if key in self.loss_weights and self.loss_weights[key] > 0:
                    loss += self.loss_weights[key] * value
    
        # compute loss
        return loss, results


    def forward(self, outputs, inputs, phase, iter=0, epoch=0):
        results = {}
        rays_o, rays_dir, rays_dist = inputs['rays_o'], inputs['rays_d'], inputs['first_dist']
        mask = inputs['first_mask'].bool()
        device = rays_o.device

        loss = torch.tensor([0.],requires_grad=True).float().to(device)
        c_key = 'coarse'
        c_loss, c_results = self.get_loss_terms(inputs, outputs[c_key], phase)
        est_dist = outputs[c_key]['depth']
        pts_est = (rays_o + rays_dir * est_dist[:,None]) * self.extent
        pts_gt = (rays_o + rays_dir * rays_dist[:,None]) * self.extent
        outputs[c_key]['pts_est'] = pts_est + self.center[None]

        for key, value in c_results.items():
            results[f'{key}'] = value
        results['opacity'] = outputs[c_key]['opacity'].max(1)[0].median()
        results['weight'] = outputs[c_key]['weights'].max(1)[0].median()
        if 'weights_2' in outputs[c_key].keys():
            results['opacity_2'] = outputs[c_key]['opacity_2'].max(1)[0].median()
            results['weight_2'] = outputs[c_key]['weights_2'].max(1)[0].median()

        loss += c_loss
        results['loss'] = loss

        outputs['pts_gt'] = pts_gt + self.center[None]

        if phase == 'test': # plot the ecdf curve for each scene
            # compute chamfer distance
            dist1, dist2, idx1, idx2 = self.chamfer_loss(pts_gt[mask][None], pts_est[mask][None])
            dist1, dist2 = dist1**0.5, dist2**0.5
            results['chamfer_dist'] = dist1.mean().detach() + dist2.mean().detach()

            # lidar_to_range_img(inputs, outputs, self.extent, os.path.join(self.save_dir, f'{iter}.jpeg'))
        return results