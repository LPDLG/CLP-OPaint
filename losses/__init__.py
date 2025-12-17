import torch.nn as nn
from .reconstruct import ReconLoss
from .perceptual import PerceptualLoss
import torch
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .edge import SimpleEdgeLoss
class SetCriterion(nn.Module):
    def __init__(self,opts):
        super().__init__()
        self.recon_loss = ReconLoss()
        self.perceptual_loss = PerceptualLoss()
        
        self.gen_weight_dict = {
            'loss_g_recon': 5, 
            'loss_g_adversarial': 1, 
            'loss_g_perceptual': 10, 
            'loss_g_edge': 1
        }
        self.dis_weight_dict = {'loss_d_adversarial': 1}
        
        self.imagenet_normalize = transforms.Normalize(
            mean=torch.tensor(IMAGENET_DEFAULT_MEAN),
            std=torch.tensor(IMAGENET_DEFAULT_STD)
        )
        self.patch_mean = opts.patch_mean
        self.patch_std = opts.patch_std
        
        self.edge_loss = SimpleEdgeLoss(
            edge_weight=1.0
        )
        
        self.current_stage = 1
        self.stage_weights = {
            1: {'recon': 5, 'edge': 1},
            2: {'recon': 3, 'edge': 2},
            3: {'recon': 1, 'edge': 3}  
        }


    def update_stage(self, new_stage):
        self.current_stage = new_stage
        self.edge_loss.update_stage(new_stage)
        
        if new_stage == 1:
            self.gen_weight_dict['loss_g_recon'] = 5.0
            self.gen_weight_dict['loss_g_edge'] = 0.5
            self.gen_weight_dict['loss_g_perceptual'] = 10.0 
            self.gen_weight_dict['loss_g_adversarial'] = 0.8 
        elif new_stage == 2:  
            self.gen_weight_dict['loss_g_recon'] = 3.0 
            self.gen_weight_dict['loss_g_edge'] = 1.5 
            self.gen_weight_dict['loss_g_perceptual'] = 8.0 
            self.gen_weight_dict['loss_g_adversarial'] = 1.0
            
        else: 
            self.gen_weight_dict['loss_g_recon'] = 1.0
            self.gen_weight_dict['loss_g_edge'] = 3.0
            self.gen_weight_dict['loss_g_perceptual'] = 6.0 
            self.gen_weight_dict['loss_g_adversarial'] = 1.5 
     


    def renorm(self, tensor):
        tensor = tensor * self.patch_std + self.patch_mean
        return self.imagenet_normalize(tensor)

    def get_dis_loss(self, input_fake, input_real, discriminator=None):
        assert discriminator is not None
        return {'loss_d_adversarial': discriminator.calc_dis_loss(input_fake.detach(),input_real)}

    def get_gen_loss(self, input_fake, input_real,discriminator=None, warmup=False, mask=None):
        if not warmup:
            assert discriminator is not None
            g_loss_dict={'loss_g_adversarial': discriminator.calc_gen_loss(input_fake,input_real)}
            g_loss_dict['loss_g_recon']=self.recon_loss(input_fake,input_real)
            g_loss_dict['loss_g_perceptual']=self.perceptual_loss(self.renorm(input_fake),self.renorm(input_real))
            
            if mask is not None:
                g_loss_dict['loss_g_edge'] = self.edge_loss(input_fake, input_real, mask)
            else:
                g_loss_dict['loss_g_edge'] = self.edge_loss(input_fake, input_real)
                
            return g_loss_dict
        else:
            return {'loss_g_recon':self.recon_loss(input_fake,input_real)}





