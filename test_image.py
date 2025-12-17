import os
import torch
import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import argparse
from torchvision import transforms
from models.VITGen import TransGen
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import time

MODEL_PATH = 'path/to/your/checkpoint'
IMAGE_PATH = 'path/to/your/test_image'
OUTPUT_DIR = 'path/to/your/output_dir'
ENC_CKPT_PATH = 'path/to/your/encoder_checkpoint'

# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--eval', default=True, type=bool)
parser.add_argument('--input_size', type=int, default=128)
parser.add_argument('--output_size', type=int, default=192)
parser.add_argument('--dec_depth', type=int, default=4)
parser.add_argument('--normlize_target', default=True, type=bool)
parser.add_argument('--patch_mean', type=float, default=0.609)
parser.add_argument('--patch_std', type=float, default=0.198)
parser.add_argument('--resume', type=str, default=MODEL_PATH)
parser.add_argument('--image_path', type=str, default=IMAGE_PATH)
parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
parser.add_argument('--enc_ckpt_path', type=str, default=ENC_CKPT_PATH)
parser.add_argument('--current_stage', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=4)  
def denorm_img(tensor, opts):
    tensor = rearrange(tensor, 'b c h w -> b h w c').detach().cpu()
    tensor = tensor * torch.tensor([opts.patch_std] * 3) + torch.tensor([opts.patch_mean] * 3)
    tensor = torch.clamp(tensor, 0, 1)
    return tensor

class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, opts):
        self.opts = opts
        self.output_size = opts.output_size  
        self.input_size = opts.input_size   
        
        if os.path.isdir(image_path):
            self.image_paths = []
            valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
            for f in os.listdir(image_path):
                if f.lower().endswith(valid_extensions):
                    self.image_paths.append(os.path.join(image_path, f))
            self.image_paths.sort()
        else:
            self.image_paths = [image_path]
        self.transform = transforms.Compose([
            transforms.Resize((192, 192), interpolation=Image.BICUBIC), 
        ])
        
        self.input_norm = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        self.output_norm = transforms.Normalize([opts.patch_mean]*3, [opts.patch_std]*3)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        orig_img = Image.open(img_path).convert('RGB')
        orig_size = orig_img.size
        
        img_192 = self.transform(orig_img)
        
        center = (192 - 128) // 2
        input_img = img_192[:, center:center+128, center:center+128]
        output_gt = img_192
        
        mask = torch.zeros(1, self.output_size, self.output_size)
        mask_center = (self.output_size - self.input_size) // 2  # 32
        mask[:, mask_center:mask_center+self.input_size, mask_center:mask_center+self.input_size] = 1
        
        input_img = self.input_norm(input_img)
        output_gt = self.output_norm(output_gt)
        gt_inner = output_gt * mask
        
        return {
            'input': input_img,
            'ground_truth': output_gt,
            'gt_inner': gt_inner,
            'mask': mask,
            'name': img_name,
            'orig_size': orig_size,
            'orig_img_path': img_path
        }

def save_generated_image(generated_tensor, save_path, orig_img_path, opts, apply_enhancement=True):
    generated_img = denorm_img(generated_tensor[0:1], opts)[0].numpy()
    
    generated_pil = Image.fromarray((generated_img * 255).astype(np.uint8))
    
    if apply_enhancement:
        enhancer = ImageEnhance.Sharpness(generated_pil)
        generated_pil = enhancer.enhance(1.2)
        
        contrast_enhancer = ImageEnhance.Contrast(generated_pil)
        generated_pil = contrast_enhancer.enhance(1.1)
    
    if orig_img_path and os.path.exists(orig_img_path):
        orig_img = Image.open(orig_img_path).convert('RGB')
        orig_width, orig_height = orig_img.size
        
        if orig_width != orig_height:
            aspect_ratio = orig_width / orig_height
            
            if aspect_ratio > 1:
                new_width = orig_width
                new_height = orig_height
            else:
                new_width = orig_width
                new_height = orig_height
            
            generated_pil = generated_pil.resize((new_width, new_height), Image.LANCZOS)
        else:
            generated_pil = generated_pil.resize((orig_width, orig_height), Image.LANCZOS)
    
    generated_pil.save(save_path, quality=100, optimize=True)

if __name__ == '__main__':
    start_time = time.time()
    opts = parser.parse_args([])
    
    
    generated_dir = os.path.join(opts.output_dir)
    os.makedirs(generated_dir, exist_ok=True)
    
    gen = TransGen(opts=opts, enc_ckpt_path=opts.enc_ckpt_path).cuda()
    
    state_dict = torch.load(opts.resume, map_location='cuda')
    if isinstance(state_dict, dict) and 'gen_state_dict' in state_dict:
        gen.load_state_dict(state_dict['gen_state_dict'])
    elif isinstance(state_dict, dict) and 'gen' in state_dict:
        gen.load_state_dict(state_dict['gen'])
    else:
        gen.load_state_dict(state_dict)
    gen.update_stage(opts.current_stage)
    gen.eval()
    
    test_dataset = InferenceDataset(opts.image_path, opts)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=opts.batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    total_images = len(test_dataset)
    processed_count = 0
    
    with torch.no_grad():
        for batch_idx, test_data in enumerate(test_loader):
            batch_start = time.time()
            batch_size = len(test_data['name'])
            
            for k in test_data.keys():
                if isinstance(test_data[k], torch.Tensor):
                    test_data[k] = test_data[k].cuda()
            
            fake = gen(test_data)
            
            for i in range(batch_size):
                name = test_data['name'][i]
                orig_img_path = test_data['orig_img_path'][i]
                
                save_path = os.path.join(generated_dir, f'{name}_generated.png')
                save_generated_image(
                    fake[i:i+1], save_path, orig_img_path, opts, 
                    apply_enhancement=True
                )
                
                processed_count += 1
                
    
    total_time = time.time() - start_time