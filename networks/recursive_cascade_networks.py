from .transform import sample_power, free_form_fields
from .base_networks import *
from .layers import SpatialTransformer

class RecursiveCascadeNetwork(nn.Module):
    def __init__(self, n_cascades, im_size=(512, 512), base_network='VTN', in_channels=2, cr_aff=False):
        super(RecursiveCascadeNetwork, self).__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stems = nn.ModuleList()
        # See note in base_networks.py about the assumption in the image shape
        print('using AffineStem with correct: {}'.format(cr_aff))
        self.stems.append(VTNAffineStem(dim=len(im_size), im_size=im_size[0], flow_correct=cr_aff, in_channels=in_channels))
        assert base_network in BASE_NETWORK
        base = eval(base_network)
        for i in range(n_cascades):
            self.stems.append(base(im_size=im_size, flow_multiplier=1.0 / n_cascades, in_channels=in_channels))

        # Parallelize across all available GPUs
        # if torch.cuda.device_count() > 1:
        #     self.stems = [nn.DataParallel(model) for model in self.stems]

        for model in self.stems:
            model.to(device)

        self.reconstruction = SpatialTransformer(im_size)
        # self.reconstruction = nn.DataParallel(self.reconstruction)
        self.reconstruction.to(device)

        # add initialization
        for module in self.modules():
            if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d)) and getattr(module, 'initialized', False):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)
            # affine init fc[-1]
            # elif isinstance(module, nn.Linear):
            #     torch.nn.init.xavier_uniform_(module.weight)
            #     torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, fixed, moving, return_affine=False, return_neg=False):
        flows = []
        stem_results = []
        agg_flows = []
        # Affine registration
        flow, affine_params = self.stems[0](fixed, moving)
        stem_results.append(self.reconstruction(moving, flow))
        flows.append(flow)
        agg_flows.append(flow)
        if return_neg:
            # neg_flow = self.stems[0].neg_flow(affine_params['theta'], moving.size())
            neg_flow = flow.new_zeros(flow.shape)
        for model in self.stems[1:]: # cascades
            # registration between the fixed and the warped from last cascade
            flow = model(fixed, stem_results[-1], return_neg=return_neg)
            if return_neg:
                neg_fl = flow[1]
                neg_flow = self.reconstruction(neg_flow, neg_fl) + neg_fl
                flow = flow[0]
            flows.append(flow)
            if len(agg_flows) == 1:
                theta = affine_params['theta']
                A = theta[:, :3, :3]
                agg_flow = flows[0] + torch.einsum('bij, bjxyz -> bixyz', A, flow)
            else:
                agg_flow = self.reconstruction(agg_flows[-1], flow) + flow
            agg_flows.append(agg_flow)
            stem_results.append(self.reconstruction(moving, agg_flow))
        if return_neg:
            agg_flows.append(neg_flow)
        returns = [stem_results, flows, agg_flows]
        if return_affine:
            returns.append(affine_params)
        return returns

    def augment(self, img2: torch.Tensor, seg2: torch.Tensor):
        bs, c, h, w, s = img2.shape
        control_fields = sample_power(-.4, .4, 3, (bs, 3, 5, 5, 5)).to(img2.device) * (img2.new_tensor([h, w, s])//4)[..., None, None, None]
        aug_flow = free_form_fields((h, w, s), control_fields)
        aug_img2 = self.reconstruction(img2, aug_flow)
        aug_seg2 = self.reconstruction(seg2, aug_flow)
        return aug_img2, aug_seg2


