import torch
import torch.nn as nn

class PreRegister(nn.Module):
    """
    A module to pre-register the template image to the input image and get the organ mask. The weight of the model for pre-registration is fixed.
    """
    def __init__(self, RCN, template_input, template_seg):
        super(PreRegister, self).__init__()
        self.RCN = RCN
        self.register_buffer('template_input', template_input)
        self.register_buffer('template_seg', template_seg)
        self.freeze()
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, stage1_inputs):
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

