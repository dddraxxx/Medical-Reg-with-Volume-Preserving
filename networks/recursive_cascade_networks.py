from .transform import sample_power, free_form_fields
from .base_networks import *
from .spatial_transformer import SpatialTransform

class RecursiveCascadeNetwork(nn.Module):
    def __init__(self, n_cascades, im_size=(512, 512)):
        super(RecursiveCascadeNetwork, self).__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stems = nn.ModuleList()
        # See note in base_networks.py about the assumption in the image shape
        # to do: get same network as in official tf implementation
        self.stems.append(VTNAffineStem(dim=len(im_size), im_size=im_size[0]))
        for i in range(n_cascades):
            self.stems.append(VTN(dim=len(im_size), flow_multiplier=1.0 / n_cascades))

        # Parallelize across all available GPUs
        # if torch.cuda.device_count() > 1:
        #     self.stems = [nn.DataParallel(model) for model in self.stems]

        for model in self.stems:
            model.to(device)

        self.reconstruction = SpatialTransform(im_size)
        # self.reconstruction = nn.DataParallel(self.reconstruction)
        self.reconstruction.to(device)

        # add initialization
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
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

    def forward(self, fixed, moving, return_affine=False):
        flows = []
        stem_results = []
        agg_flows = []
        # Affine registration
        flow, affine_params = self.stems[0](fixed, moving)
        stem_results.append(self.reconstruction(moving, flow))
        flows.append(flow)
        agg_flows.append(flow)
        for model in self.stems[1:]: # cascades
            # registration between the fixed and the warped from last cascade
            flow = model(fixed, stem_results[-1])
            flows.append(flow)
            if len(agg_flows) == 1:
                theta = affine_params['theta']
                A = theta[:, :3, :3]
                agg_flow = flows[0] + torch.einsum('bij, bjxyz -> bixyz', A, flow)
            else:
                agg_flow = self.reconstruction(agg_flows[-1], flow) + flow
            agg_flows.append(agg_flow)
            stem_results.append(self.reconstruction(moving, agg_flow))
        if return_affine:
            return stem_results, flows, agg_flows, affine_params
        return stem_results, flows, agg_flows

    def augment(self, img2: torch.Tensor, seg2: torch.Tensor):
        bs, c, h, w, s = img2.shape
        control_fields = sample_power(-.4, .4, 3, (bs, 3, 5, 5, 5)).to(img2.device) * (img2.new_tensor([h, w, s])//4)[..., None, None, None]
        aug_flow = free_form_fields((h, w, s), control_fields)
        aug_img2 = self.reconstruction(img2, aug_flow)
        aug_seg2 = self.reconstruction(seg2, aug_flow)
        return aug_img2, aug_seg2


