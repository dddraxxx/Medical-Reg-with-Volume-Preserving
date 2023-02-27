import torchvision.transforms as tf
import torch
import torch.nn as nn
from tools.utils import *
from tools.visualization import plot_to_PIL
import os
from metrics.losses import jacobian_det
import wandb

class PreRegister(nn.Module):
    """
    A module to pre-register the template image to the input image and get the organ mask. The weight of the model for pre-registration is fixed.
    """
    def __init__(self, RCN, template_input, template_seg):
        super(PreRegister, self).__init__()
        RCN.eval()
        self.RCN = RCN
        self.register_buffer('template_input', template_input)
        self.register_buffer('template_seg', template_seg)
        self.freeze()
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, fixed, moving, seg2=None, training=False, cfg=None):
        """ Compute mask for tumor region """
        stage1_model = self.RCN
        template_image = self.template_input
        template_seg = self.template_seg
        log_scalars, img_dict = {}, {}
        with torch.no_grad():
            stage1_inputs = moving
            template_input = template_image.expand_as(stage1_inputs)
            # achieve organ mask via regristration
            w_template, _, s1_agg_flows = stage1_model(stage1_inputs, template_input, return_neg=cfg.stage1_rev)
            s1_flow = s1_agg_flows[-1]
            w_template_seg = stage1_model.reconstruction(template_seg.expand_as(template_input), s1_flow)
            mask_moving = w_template_seg
        def warp(fixed, moving, seg, reg_model):
            """ Warp moving image to fixed image, and return warped seg """
            with torch.no_grad():
                w_moving, _, rs1_agg_flows = reg_model(fixed, moving)
                w_seg = reg_model.reconstruction(seg, rs1_agg_flows[-1])
            return w_moving[-1], w_seg, rs1_agg_flows[-1]
        def filter_nonorgan(x, seg):
            x[~(seg>0.5)] = 0
            return x
        def extract_tumor_mask(stage1_moving_flow, moving_mask, n=1, thres=3):
            flow_det_moving = jacobian_det(stage1_moving_flow, return_det=True)
            flow_det_moving = flow_det_moving.unsqueeze(1).abs()
            if cfg.stage1_rev: flow_ratio = 1/flow_det_moving
            else: flow_ratio = flow_det_moving
            # get n x n neighbors 
            flow_ratio = flow_ratio.clamp(1/thres,thres)
            sizes = F.interpolate(flow_ratio, size=fixed.shape[-3:], mode='trilinear', align_corners=False) \
                if flow_ratio.shape[-3:]!=fixed.shape[-3:] else flow_ratio
            flow_ratio = F.avg_pool3d(sizes, kernel_size=n, stride=1, padding=n//2)
            if moving_mask is not None:
                filter_nonorgan(flow_ratio, moving_mask)
            globals().update(locals())
            return flow_ratio
        def remove_boundary_weights(fratio, mask, r=3):
            """ remove the weights near the boundary """
            # the thres on how to define boundary: 1, .5 or 0
            # boundary_region = find_surf(mask, 3, thres=.5)
            boundary_region = find_surf(mask, r, thres=cfg.boundary_thickness)
            # wandb.config.boundary_thres=0
            if training:
                im = combo_imgs(boundary_region.float()[0,0], fratio[0,0])
                img_dict['before_boundary'] = tf.ToTensor()(im)
            fratio[boundary_region] = 0
            ### is removing boundary really needed?
            # check the boundary region
            # import pdb; pdb.set_trace()
            # combo_imgs(w_moving2[0,0], w_moving_seg2[0,0], boundary_region[0,0].float()).save('1.png')
            return fratio
        # find shrinking tumors
        w_moving, w_moving_seg, rs1_flow = warp(fixed, moving, mask_moving, stage1_model)
        target_ratio = (w_moving_seg>0.5).sum(dim=(2,3,4), keepdim=True).float() / (mask_moving>0.5).sum(dim=(2,3,4), keepdim=True).float()

        ### better to calculate the ratio compared to the target ratio
        flow_ratio = extract_tumor_mask(rs1_flow, w_moving_seg, n=cfg.masked_neighbor) * target_ratio
        if cfg.boundary_thickness>0:
            flow_ratio = remove_boundary_weights(flow_ratio, w_moving_seg)
        # temporarily use fix->moving flow to solve this inversion 
        ### will it cause the shrinking of tumor?
        # reverse it to input seg
        # w_moving_rev_flow = cal_rev_flow_gpu(rs1_flow)
        w_moving_rev_flow = stage1_model(moving, w_moving)[2][-1]
        rev_flow_ratio = stage1_model.reconstruction(flow_ratio, w_moving_rev_flow)
        if cfg.use_2nd_flow:
            # do the same for second stage1 registration
            w_moving2, w_moving_seg2, rs1_flow2 = warp(fixed, w_moving, w_moving_seg, stage1_model)
            target_ratio2 = (w_moving_seg2>0.5).sum(dim=(2,3,4), keepdim=True).float() / (w_moving_seg>0.5).sum(dim=(2,3,4), keepdim=True).float()
            flow_ratio2 = extract_tumor_mask(rs1_flow2, w_moving_seg2, n=cfg.masked_neighbor) * target_ratio2
            flow_ratio2 = remove_boundary_weights(flow_ratio2, w_moving_seg2)
            # w_moving_rev_flow2 = cal_rev_flow_gpu(rs1_flow2)
            w_moving_rev_flow2 = stage1_model(w_moving, w_moving2)[2][-1]
            comp_flow2 = stage1_model.composite_flow(w_moving_rev_flow2, w_moving_rev_flow)
            rev_flow_ratio2 = stage1_model.reconstruction(flow_ratio2, comp_flow2)

        #####
        ### will it be better to use the mask of aggregated mask? thought that would be minor though
        #####

        # TODO: try to combine rev_flow and rev_flow2?

        if training and cfg.debug:
            log_scalars['stage1_target_ratio'] = target_ratio.mean()
            img_dict['stage1_revflowratio'] = visualize_3d(rev_flow_ratio[0,0])
            if cfg.use_2nd_flow:
                img_dict['stage1_revflowratio2'] = visualize_3d(rev_flow_ratio2[0,0])

        if cfg.use_2nd_flow:
            dynamic_mask = rev_flow_ratio2
        else: 
            dynamic_mask = rev_flow_ratio
        if cfg.masked=='soft':
            # temporarily increase tumor area to check if dynamic weights works
            # rev_flow_ratio2 = torch.maximum(((seg2>1.5).float() + rev_flow_ratio2)/2, rev_flow_ratio2)
            # set up a functino to transfer flow_ratio to soft_mask for similarity (and possibly VP loss)
            sin_trsf_f = lambda x: torch.sin(((x-1.5)/.5).clamp(-1,1)*(np.pi/2)) * 0.5 + 0.5 # the value should be between 0 and 1
            # use (-5,5) interval of sigmoid to stimulate the desired transformation
            sigm_trsf_f = lambda x: torch.sigmoid((x-1.5)*5)
            if cfg.soft_transform=='sin':
                trsf_f = sin_trsf_f
            elif cfg.soft_transform=='sigm':
                trsf_f = sigm_trsf_f
            # wandb.config.soft_trsf = 'sigmoid*5'
            # corresponds to the moving image
            soft_mask = trsf_f(dynamic_mask) # intense value means the place is likely to be not similar to the fixed
            soft_mask = filter_nonorgan(soft_mask, mask_moving)
            input_seg = soft_mask + mask_moving
            if training:
                img_dict['input_seg'] = visualize_3d(input_seg[0,0])
                img_dict['soft_mask'] = visualize_3d(soft_mask[0,0])
                # should delete in real trainning, just to check the goodneess of rev1 and rev2
                rev_fratio = trsf_f(rev_flow_ratio)
                img_dict['rev1_mask_on_seg2'] = visualize_3d(draw_seg_on_vol(rev_fratio[0,0],
                                                                            seg2[0,0].round().long(),
                                                                            to_onehot=True))
                if cfg.use_2nd_flow:
                    img_dict['stage1_mask_on_seg2'] = visualize_3d(draw_seg_on_vol(soft_mask[0,0], 
                                                                        seg2[0,0].round().long(),
                                                                        to_onehot=True))
            returns = (input_seg, soft_mask)
        elif cfg.masked=='hard':
            # thres = torch.quantile(w_n, .9)
            thres = cfg.mask_threshold
            hard_mask = dynamic_mask > thres
            input_seg = hard_mask + mask_moving
            hard_mask_onimg = draw_seg_on_vol(moving[0,0], input_seg[0,0].round().long(), to_onehot=True)
            if training:
                img_dict['input_seg'] = visualize_3d(input_seg[0,0])
                img_dict['stage1_mask_on_seg2']= visualize_3d(hard_mask_onimg)
            returns = (input_seg, hard_mask)

        if training and cfg.debug:
            # visualize w_moving, seg2, moving
            save_dir = 'z_debug'
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            combo_imgs(*w_moving[:,0]).save(f'{save_dir}/w_moving.jpg')
            with torch.no_grad():
                # check the results of stage1 model on fixed-moving-registration
                _, _, ss1ag_flows = stage1_model(fixed, moving)
                wwseg2 = stage1_model.reconstruction(seg2, ss1ag_flows[-1])
                w1_seg2 = stage1_model.reconstruction(seg2, rs1_flow)
                w2_moving, _, rs2_agg_flows = stage1_model(template_input[:fixed.shape[0]], w_moving)
                w2_seg2 = stage1_model.reconstruction(w1_seg2, rs2_agg_flows[-1])
                flow_ratio2 = extract_tumor_mask(rs2_agg_flows[-1], w2_seg2)
                w3_moving, _, rs3_agg_flows = stage1_model(template_input[:fixed.shape[0]], w2_moving[-1])
                w3_seg2 = stage1_model.reconstruction(w2_seg2, rs3_agg_flows[-1])
                flow_ratio3 = extract_tumor_mask(rs3_agg_flows[-1], w3_seg2)
                # find the area ranked by dissimilarity
            def dissimilarity(x, y):
                x_mean = x-x.mean(dim=(1,2,3,4), keepdim=True)
                y_mean = y-y.mean(dim=(1,2,3,4), keepdim=True)
                correlation = x_mean*y_mean
                return -correlation
            def dissimilarity_mask(x, y, mask):
                x_mean = (x*mask).sum(dim=(1,2,3,4), keepdim=True)/mask.sum(dim=(1,2,3,4), keepdim=True)
                y_mean = (y*mask).sum(dim=(1,2,3,4), keepdim=True)/mask.sum(dim=(1,2,3,4), keepdim=True)
                x_ = x-x_mean
                y_ = y-y_mean
                correlation = x_*y_
                correlation[~mask] = correlation[mask].median()
                return -correlation
            # dissimilarity_moving = dissimilarity_mask(template_input[:fixed.shape[0]], w_moving[-1], w1_seg2>.5)
            # dissimilarity_moving2 = dissimilarity_mask(template_input[:fixed.shape[0]], w2_moving[-1], w2_seg2>.5)
            # combo_imgs(*dissimilarity_moving[:,0]).save(f'{save_dir}/dissimilarity_moving.jpg')
            # combo_imgs(*dissimilarity_moving2[:, 0]).save(f"{save_dir}/dissimilarity_moving2.jpg")
            # what if we use the second stage flow?
            combo_imgs(*w2_moving[-1][:,0]).save(f'{save_dir}/w2_moving.jpg')
            combo_imgs(*flow_ratio2[:,0]).save(f'{save_dir}/flow_ratio2.jpg')
            # or the third stage flow?
            combo_imgs(*w3_moving[-1][:,0]).save(f'{save_dir}/w3_moving.jpg')
            combo_imgs(*flow_ratio3[:,0]).save(f'{save_dir}/flow_ratio3.jpg')
            combo_imgs(*w_template[-1][[1,2,5,6],0]).save(f'{save_dir}/w_template.jpg')
            combo_imgs(*w_template_seg[:,0]).save(f'{save_dir}/w_template_seg.jpg')
            combo_imgs(*wwseg2[:,0]).save(f'{save_dir}/wwseg2.jpg')
            # combo_imgs(*mask_fixing[:,0]).save(f'{save_dir}/mask_fixing.jpg')
            combo_imgs(*mask_moving[:,0]).save(f'{save_dir}/mask_moving.jpg')
            combo_imgs(*moving[:,0]).save(f'{save_dir}/moving.jpg')
            combo_imgs(*seg2[:,0]).save(f'{save_dir}/seg2.jpg')
            combo_imgs(*flow_ratio[:,0]).save(f'{save_dir}/flow_ratio.jpg')
            # combo_imgs(*hard_mask[:,0]).save(f'{save_dir}/hard_mask.jpg')
            # import debugpy; debugpy.listen(5678); print('Waiting for debugger attach'); debugpy.wait_for_client(); debugpy.breakpoint()
        if training:
            # move all to cpu
            log_scalars = {k: v.item() for k, v in log_scalars.items()}
            img_dict = {k: v.cpu() for k, v in img_dict.items()}
            return *returns, log_scalars, img_dict
        else:
            return returns



    def template_registration(self, stage1_inputs):
        """
        Register template_input to stage1_inputs and return the registered organ mask.
        """
        stage1_model = self.RCN
        template_image = self.template_input
        template_seg = self.template_seg
        template_input = template_image.expand_as(stage1_inputs)
        # achieve organ mask via regristration
        _, _, s1_agg_flows = stage1_model(stage1_inputs, template_input)
        s1_flow = s1_agg_flows[-1]
        w_template_seg = stage1_model.reconstruction(template_seg.expand_as(template_input), s1_flow)
        return w_template_seg
