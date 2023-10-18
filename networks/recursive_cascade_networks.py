import sys
sys.path.append("..")
from tools.utils import load_model_from_dir
from .transform import sample_power, free_form_fields
from .base_networks import *
from .layers import SpatialTransformer
from .pre_register import PreRegister
from .hyper_net import HyperModel

class RecursiveCascadeNetwork(nn.Module):
    """
    A PyTorch module that implements the recursive cascaded network.

    Args:
        n_cascades (int): The number of cascades.
        im_size (tuple): The size of the input image.
        base_network (str): The base network to use. Can be 'VTN' or 'VXM'.
        in_channels (int): The number of input channels.
        compute_mask (bool): Whether to compute the mask before registration by using the pre-registration module.
    """
    def __init__(self, n_cascades, im_size=(512, 512), base_network='VTN', in_channels=2, use_affine=True, hyper_net=False):
        super(RecursiveCascadeNetwork, self).__init__()
        self.n_casescades = n_cascades
        self.im_size = im_size
        self.use_affine = use_affine
        self.base_network = base_network
        self.in_channels = in_channels

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if hyper_net:
            self.hypernet = HyperModel(nb_hyp_params=1, nb_hyp_layers=6, nb_hyp_units=128)
        self.stems = nn.ModuleList()
        # See note in base_networks.py about the assumption in the image shape
        if use_affine:
            if base_network != 'TSM':
                self.stems.append(VTNAffineStem(dim=len(im_size), im_size=im_size[0], in_channels=in_channels))
            else:
                self.stems.append(TSMAffineStem(dim=len(im_size), im_size=im_size[0], in_channels=in_channels))
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

        def load_state_dict(self, state_dict, strict=True):
            """
            Copies parameters and buffers from :attr:`state_dict`
            into this module and its descendants. If :attr:`strict` is ``True``,
            then the keys of :attr:`state_dict` must exactly match the keys returned

            Modified: To use pre-trained model with different in-channel for input
            """
            own_state = self.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    continue
                if '.conv1.1.weight' in name and param.size(1)==2 and own_state[name].size(1)==3: # in case we want to use model with in-channel 2 whiel the current in-channel is 3
                    own_state[name][:,:2].copy_(param)
                    print("In_channel is not the same, only copy the first two channels")
                    continue
                if isinstance(param, nn.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                own_state[name].copy_(param)
        self.stems.load_state_dict = load_state_dict.__get__(self.stems, nn.ModuleList)

    def build_stage1_model(self, state_path='/home/hynx/regis/recursive-cascaded-networks/logs/Jan08_180325_normal-vtn', base_network='VTN'):
        """
        Build the stage 1 model from the saved state

        Returns:
            stage1_model (RecursiveCascadeNetwork | nn.Module): The stage 1 model
        """
        print('Building stage 1 model from', state_path)
        stage1_model = RecursiveCascadeNetwork(n_cascades=self.n_casescades, im_size = self.im_size, base_network=base_network, in_channels=2)
        load_model_from_dir(state_path, stage1_model)
        return stage1_model

    def build_preregister(self, template_input, template_seg, state_path=None, base_network='VTN'):
        """
        Build the pre-register module. The module is currently built on the stage1 model.
        """
        state_path = state_path or '/home/hynx/regis/recursive_cascaded_networks/logs/Jan08_180325_normal-vtn'
        self.pre_register: PreRegister = PreRegister(self.build_stage1_model(state_path, base_network), template_input, template_seg)
        return self.pre_register

    def forward(self, fixed, moving, return_affine=False, return_neg=False, hyp_input=None):
        # think about how to migrate the pre-computing into the model
        # if self.compute_mask:
        #     moving_seg = self.pre_register(moving)
        #     moving = torch.cat([moving, moving_seg], dim=1)
        flows = []
        stem_results = []
        agg_flows = []
        neg_flow = None
        hyp_tensor = None
        if hyp_input is not None and hasattr(self, 'hypernet'):
            hyp_tensor = self.hypernet(hyp_input)

        if self.use_affine:
            # Affine registration
            flow, affine_params = self.stems[0](fixed, moving)
            # affine_params['theta'] = flow.new_tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]).float()[None].repeat(fixed.size(0), 1, 1)
            stem_results.append(self.reconstruction(moving, flow))
            aflow = flow
            flows.append(flow)
            agg_flows.append(flow)
        else:
            flow = self.stems[0](fixed, moving, return_neg=return_neg)
            stem_results.append(self.reconstruction(moving, flow))
            aflow = flow
            flows.append(flow)
        if return_neg:
            neg_flow = self.stems[0].neg_flow(affine_params['theta'], moving.size())
        for model in self.stems[1:]: # cascades
            # registration between the fixed and the warped from last cascade
            flow = model(fixed, stem_results[-1], return_neg=return_neg)
            if return_neg:
                neg_fl = flow[1]
                ## reverse flow cannot utilize the affine params
                # if len(agg_flows) == 1:
                #     neg_theta = neg_affine['theta']
                #     neg_A = neg_theta[:, :3, :3]
                #     neg_flow = neg_fl + torch.einsum('bij, bjxyz -> bixyz', neg_A, neg_flow)
                # else:
                neg_flow = self.reconstruction(neg_fl, neg_flow) + neg_flow
                del neg_fl
                flow = flow[0]
            flows.append(flow)
            if len(agg_flows) == 1 and self.use_affine:
                theta = affine_params['theta']
                A = theta[:, :3, :3]
                agg_flow = flows[0] + torch.einsum('bij, bjxyz -> bixyz', A, flow)
            else:
                # agg_flow = self.reconstruction(agg_flows[-1], flow) + flow
                agg_flow = self.reconstruction(aflow, flow) + flow
            agg_flows.append(agg_flow)
            # stem_results.append(self.reconstruction(moving, agg_flow))
            stem_results[-1] = self.reconstruction(moving, agg_flow)
            aflow = agg_flow
        ## to keep format as previous [results, flows, agg_flows] and also reduce memory usage
        # returns = [stem_results, flows, [aflow]]
        returns = [stem_results, flows, agg_flows]
        if return_neg:
            returns.append([neg_flow])
        if return_affine:
            returns.append(affine_params)
        return returns

    def composite_flow(self, old_flow, new_flow):
        """
        Composite two flows into one
        """
        return self.reconstruction(old_flow, new_flow) + new_flow

    def augment(self, img2: torch.Tensor, seg2: torch.Tensor):
        """
        Augment the image and segmentation with random free-form deformation. The current deforming range is [-.4, .4] pixels.
        """
        bs, c, h, w, s = img2.shape
        control_fields = sample_power(-.4, .4, 3, (bs, 3, 5, 5, 5)).to(img2.device) * (img2.new_tensor([h, w, s])//4)[..., None, None, None]
        aug_flow = free_form_fields((h, w, s), control_fields)
        aug_img2 = self.reconstruction(img2, aug_flow)
        aug_seg2 = self.reconstruction(seg2, aug_flow)
        return aug_img2, aug_seg2
