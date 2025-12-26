import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import itertools
import datetime
import json 

torch.backends.cudnn.benchmark = True
from torch.utils.data import DataLoader
from datasets import ImageDataset
from util.misc import cosine_scheduler

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='')

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--min_lr', type=float, default=1e-4)
parser.add_argument('--warnup_epoch', type=int, default=10)
parser.add_argument('--max_epoch', type=int, default=600)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=8)

parser.add_argument('--eval', default=False, type=bool)
parser.add_argument('--half_precision', default=False, type=bool)

parser.add_argument('--input_size', type=int, default=128)
parser.add_argument('--output_size', type=int, default=192)
parser.add_argument('--enc_ckpt_path', type=str, default='')
parser.add_argument('--dec_depth', type=int, default=4)

parser.add_argument('--data_root', type=str, default='')
parser.add_argument('--normlize_target', default=True, type=bool, help='normalized the target patch pixels')
parser.add_argument('--patch_mean', type=float, default=0.609)
parser.add_argument('--patch_std', type=float, default=0.198)

parser.add_argument('--is_train', default=True, type=bool, help='')

parser.add_argument('--total_stages', type=int, default=3, help='')

parser.add_argument('--resume', default='', type=str, help='')
parser.add_argument('--auto_resume', default=False, type=bool, help='')

from models.VITGen import TransGen
from models.CNNDis import MsImageDis
from losses import SetCriterion
from engine import train_one_epoch, train_one_epoch_warmup

def count_parameters(model, trainable_only=True):
    if trainable_only:
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    else:
        params = sum(p.numel() for p in model.parameters())
 
    
    return params, param_type



if __name__ == '__main__':
    opts = parser.parse_args()
    opts.is_train = not opts.eval

    if not hasattr(opts, 'name') or not opts.name:
        opts.name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    nowname = now + '_' + opts.name
    
    resume_path = None
    if opts.resume:
        resume_path = opts.resume
    elif opts.auto_resume:
        if os.path.exists(os.path.join("logs", opts.name, "checkpoints")):
            def find_latest_checkpoint(ckpt_dir):
                if not os.path.exists(ckpt_dir):
                    return None
                
                checkpoints = [f for f in os.listdir(ckpt_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
                if not checkpoints:
                    return None
                
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].replace('.pth', '')))
                return os.path.join(ckpt_dir, latest_checkpoint)
            
            resume_path = find_latest_checkpoint(os.path.join("logs", opts.name, "checkpoints"))

    if resume_path and os.path.isfile(resume_path):
        logdir = os.path.join("logs", opts.name)
    else:
        logdir = os.path.join("logs", nowname)
    
    ckptdir = os.path.join(logdir, "checkpoints")
    visdir = os.path.join(logdir, "visuals")
    for d in [logdir, ckptdir, visdir]:
        os.makedirs(d, exist_ok=True)
    
    opts.visdir = visdir
    opts.ckptdir = ckptdir
    log_path = os.path.join(logdir, "training_log.txt")
    loss_log_path = os.path.join(logdir, "loss_history.txt")

    train_dataset = ImageDataset(opts)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opts.batch_size,
                                               num_workers=opts.num_workers, persistent_workers=opts.num_workers > 0,
                                               shuffle=True, pin_memory=True)

    gen = TransGen(opts=opts, enc_ckpt_path=opts.enc_ckpt_path).cuda()
    cnn_dis = MsImageDis().cuda()

    param_info = analyze_model_parameters(gen, cnn_dis)
    
    param_log_path = os.path.join(logdir, "model_parameters.json")
    with open(param_log_path, 'w') as f:
        json.dump(param_info, f, indent=2)

    g_param_dicts = [
        {"params": [p for n, p in gen.named_parameters() if 'conv_offset_mask' not in n and not 'transformer_encoder' in n], "lr_scale": 1},
        {"params": [p for n, p in gen.named_parameters() if 'conv_offset_mask' in n], "lr_scale": 0.1},
        {"params": [p for n, p in gen.named_parameters() if 'transformer_encoder' in n], "lr_scale": 1}
    ]

    opt_g = torch.optim.Adam(g_param_dicts, lr=opts.lr, betas=(0.0, 0.99), weight_decay=1e-4)
    opt_d = torch.optim.Adam(itertools.chain(cnn_dis.parameters()), lr=opts.lr, betas=(0.0, 0.99), weight_decay=1e-4)

    start_epoch = 0
    current_stage = 1
    iteration = 1

    if resume_path and os.path.isfile(resume_path):
        print(f"=> 加载检查点 '{resume_path}'")
        checkpoint = torch.load(resume_path)
        
        start_epoch = checkpoint['epoch']
        current_stage = checkpoint.get('current_stage', 1)
        
        gen.load_state_dict(checkpoint['gen_state_dict'])
        cnn_dis.load_state_dict(checkpoint['dis_state_dict'])
        
        opt_g.load_state_dict(checkpoint['gen_optimizer_state_dict'])
        opt_d.load_state_dict(checkpoint['dis_optimizer_state_dict'])
        
        gen.update_stage(current_stage)
        if hasattr(cnn_dis, 'update_stage'):
            cnn_dis.update_stage(current_stage)
        
        
        
    else:
        
        with open(loss_log_path, 'w') as f:


    lr_schedule_values = cosine_scheduler(opts.lr, opts.min_lr, opts.max_epoch, len(train_loader),
                                          warmup_epochs=opts.warnup_epoch, warmup_steps=-1)

    if opts.half_precision:
        g_grad_scaler = torch.cuda.amp.GradScaler()
    else:
        g_grad_scaler = None

    criterion = SetCriterion(opts)

    total_stages = 3
    stage_epochs = opts.max_epoch // total_stages

    for epoch in range(start_epoch, opts.max_epoch):
        total_stages = opts.total_stages  
        stage_epochs = opts.max_epoch // total_stages
        
        if resume_path and epoch == start_epoch:
            pass  
        else:
            current_stage = min((epoch * total_stages) // opts.max_epoch + 1, total_stages)
        
        train_dataset.current_stage = current_stage
        gen.update_stage(current_stage)
        
        if hasattr(cnn_dis, 'update_stage'):
            cnn_dis.update_stage(current_stage)
            
        if epoch == start_epoch and resume_path:
            iteration = epoch * len(train_loader) + 1

        if lr_schedule_values is not None and epoch < opts.warnup_epoch:
            for i, param_group in enumerate(opt_g.param_groups):
                param_group["lr"] = lr_schedule_values[iteration] * param_group["lr_scale"]
            for i, param_group in enumerate(opt_d.param_groups):
                param_group["lr"] = lr_schedule_values[iteration]
        else:
            for i, param_group in enumerate(opt_g.param_groups):
                param_group["lr"] = opts.lr * param_group["lr_scale"]
            for i, param_group in enumerate(opt_d.param_groups):
                param_group["lr"] = opts.lr
        
        criterion.update_stage(current_stage)
        
        if epoch < opts.warnup_epoch:
            epoch_losses = train_one_epoch_warmup(
                opts, gen, criterion, train_loader, 
                opt_g, torch.device('cuda'), epoch,
                current_stage,
                g_grad_scale=g_grad_scaler
            )
        else:
            epoch_losses = train_one_epoch(
                opts, gen, cnn_dis, criterion,
                train_loader, opt_g, opt_d,
                torch.device('cuda'), epoch,
                current_stage,
                g_grad_scale=g_grad_scaler
            )
        
        with open(loss_log_path, 'a') as f:
            loss_line = f"{epoch},{current_stage}"
            for loss_name, loss_value in epoch_losses.items():
                if isinstance(loss_value, torch.Tensor):
                    loss_value = loss_value.item()
                loss_line += f",{loss_name}:{loss_value:.6f}"
            f.write(loss_line + "\n")
        
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch} 损失摘要:")
            for loss_name, loss_value in epoch_losses.items():
                if isinstance(loss_value, torch.Tensor):
                    loss_value = loss_value.item()
                print(f"  {loss_name}: {loss_value:.6f}")

        if (epoch + 1) % max(1, opts.max_epoch//10) == 0:
            checkpoint_path = os.path.join(opts.ckptdir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'gen_state_dict': gen.state_dict(),
                'dis_state_dict': cnn_dis.state_dict(),
                'gen_optimizer_state_dict': opt_g.state_dict(),
                'dis_optimizer_state_dict': opt_d.state_dict(),
                'current_stage': current_stage
            }, checkpoint_path)
            
            gen_path = os.path.join(opts.ckptdir, f'gen_epoch_{epoch+1}.pth')
            torch.save(gen.state_dict(), gen_path)
            
            print(f"Model checkpoint saved at epoch {epoch+1} (Stage {current_stage})")

    print(f"\nStage {current_stage} completed → "
          f"Next target size: {128 + 32*(current_stage)}x{128 + 32*(current_stage)}")
