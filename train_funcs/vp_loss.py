def orig_vp_loss():
    # to decide the voxels to be preserved
    if args.size_type=='organ':
        # keep the volume of the whole organ
        vp_tum_loc = vp_seg_mask>0.5
    elif args.size_type=='tumor':
        # keep the volume of the tumor region
        vp_tum_loc = vp_seg_mask>1.5
    elif args.size_type=='tumor_gt':
        # use warped gt mask to know exactly the location of tumors
        vp_tum_loc = w_seg2>1.5
        vp_seg_mask = w_seg2
    elif args.size_type=='dynamic':
        # the vp_loc is a weighted mask that is calculated from the ratio of the tumor region
        vp_tum_loc = warped_soft_mask # B, C, S, H, W
    img_dict['vp_loc'] = visualize_3d(vp_tum_loc[0,0]).cpu()
    img_dict['vp_loc_on_wseg2'] = visualize_3d(draw_seg_on_vol(vp_tum_loc[0,0],
                                                                w_seg2[0,0].round().long(),
                                                                to_onehot=True, inter_dst=5), inter_dst=1).cpu()
    bs = agg_flows[-1].shape[0]
    # the following line uses mask_fixing and mask_moving to calculate the ratio
    # ratio = (mask_fixing>.5).view(bs,-1).sum(1)/(mask_moving>.5).view(bs, -1).sum(1);
    # now I try to use the warped mask as the reference
    ratio = (vp_seg_mask>0.5).view(bs,-1).sum(1) / (mask_moving>0.5).view(bs, -1).sum(1)
    log_scalars['target_warped_ratio'] = ratio.mean().item()
    ratio=ratio.view(-1,1,1,1)
    k_sz = 0
    ### single voxel ratio
    if args.w_ks_voxel>0:
        # Notice!! It should be *ratio instead of /ratio
        det_flow = (jacobian_det(agg_flows[-1], return_det=True).abs()*ratio).clamp(min=1/3, max=3)
        vp_mask = F.interpolate(vp_tum_loc.float(), size=det_flow.shape[-3:], mode='trilinear', align_corners=False).float().squeeze() # B, S, H, W
        with torch.no_grad():
            vp_seg = F.interpolate(w_seg2, size=det_flow.shape[-3:], mode='trilinear', align_corners=False).float().squeeze().round().long() # B, S, H, W
            img_dict['det_flow'] = visualize_3d(det_flow[0]).cpu()
        if args.ks_norm=='voxel':
            adet_flow = torch.where(det_flow>1, det_flow, (1/det_flow))
            k_sz_voxel = (adet_flow*vp_mask).sum()/vp_mask.sum()
            img_dict['vp_det_flow'] = visualize_3d(adet_flow[0]).cpu()
            img_dict['vpdetflow_on_seg2'] = visualize_3d(draw_seg_on_vol(adet_flow[0], vp_seg[0], to_onehot=True, inter_dst=5), inter_dst=1).cpu()
        elif args.ks_norm=='image':
            # normalize loss for every image
            k_sz_voxel = (torch.where(det_flow>1, det_flow, 1/det_flow)*(vp_mask)).sum(dim=(-1,-2,-3))
            nonzero_idx = k_sz_voxel.nonzero()
            k_sz_voxel = k_sz_voxel[nonzero_idx]/vp_mask[:,0][nonzero_idx].sum(dim=(-1,-2,-3))
            k_sz_voxel = k_sz_voxel.mean()
        else: raise NotImplementedError
        k_sz = k_sz + args.w_ks_voxel*k_sz_voxel
        log_scalars['k_sz_voxel'] = k_sz_voxel.item()
    if args.w_ks_voxel<1:
        det_flow = jacobian_det(agg_flows[-1], return_det=True).abs()
        vp_mask = F.interpolate(vp_tum_loc.float(), size=det_flow.shape[-3:], mode='trilinear', align_corners=False) > 0.5
        det_flow[~vp_mask[:,0]]=0
        k_sz_volume = (det_flow.sum(dim=(-1,-2,-3))/vp_mask[:,0].sum(dim=(-1,-2,-3))/ratio.squeeze()).clamp(1/3, 3)
        k_sz_volume = torch.where(k_sz_volume>1, k_sz_volume, 1/k_sz_volume)
        k_sz_volume = k_sz_volume[k_sz_volume.nonzero()].mean()
        k_sz = k_sz + (1-args.w_ks_voxel)*k_sz_volume
        log_scalars['k_sz_volume'] = k_sz_volume.item()

def norm_img():
    # normalize loss for every image
    k_sz_voxel = (torch.where(det_flow>1, det_flow, 1/det_flow)*(vp_mask)).sum(dim=(-1,-2,-3))
    nonzero_idx = k_sz_voxel.nonzero()
    k_sz_voxel = k_sz_voxel[nonzero_idx]/vp_mask[:,0][nonzero_idx].sum(dim=(-1,-2,-3))
    k_sz_voxel = k_sz_voxel.mean()

def hard_mask_vp():
    ### using hard tumor mask
    # if args.w_ks_voxel<1:
    det_flow = jacobian_det(agg_flows[-1], return_det=True).abs()
    vp_mask = F.interpolate(vp_tum_loc.float(), size=det_flow.shape[-3:], mode='trilinear', align_corners=False) > 0.5
    det_flow[~vp_mask[:,0]]=0
    k_sz_volume = (det_flow.sum(dim=(-1,-2,-3))/vp_mask[:,0].sum(dim=(-1,-2,-3))/ratio.squeeze()).clamp(1/3, 3)
    k_sz_volume = torch.where(k_sz_volume>1, k_sz_volume, 1/k_sz_volume)
    k_sz_volume = k_sz_volume[k_sz_volume.nonzero()].mean()
    k_sz = k_sz + (1-args.w_ks_voxel)*k_sz_volume
    log_scalars['k_sz_volume'] = k_sz_volume.item()