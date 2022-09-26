from .transform import sample_power, free_form_fields
from .base_networks import *
from .spatial_transformer import SpatialTransform

class RecursiveCascadeNetwork(nn.Module):
    def __init__(self, n_cascades, im_size=(512, 512)):
        super(RecursiveCascadeNetwork, self).__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stems = nn.ModuleList()
        # See note in base_networks.py about the assumption in the image shape
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

    def forward(self, fixed, moving):
        flows = []
        stem_results = []
        # Affine registration
        flow = self.stems[0](fixed, moving)
        stem_results.append(self.reconstruction(moving, flow))
        flows.append(flow)
        for model in self.stems[1:]: # cascades
            # registration between the fixed and the warped from last cascade
            flow = model(fixed, stem_results[-1])
            stem_results.append(self.reconstruction(stem_results[-1], flow))
            flows.append(flow)

        return stem_results, flows

    def augment(self, img2: torch.Tensor, seg2: torch.Tensor):
        bs, c, h, w, s = img2.shape
        control_fields = sample_power(-.4, .4, 3, (bs, 3, 5, 5, 5)).to(img2.device) * (img2.new_tensor([h, w, s])//4)[..., None, None, None]
        aug_flow = free_form_fields((h, w, s), control_fields)
        aug_img2 = self.reconstruction(img2, aug_flow)
        aug_seg2 = self.reconstruction(seg2, aug_flow)
        return aug_img2, aug_seg2


