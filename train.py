import argparse

import torch
from mpmath import arg
from torch import nn
import os
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets.LOL_paired import LOL_paired, LOL_patches
from models import model_dict
from tqdm import tqdm
from datetime import date
from utils.lossIJCV import LossMSElpipsCosineColor
from torchmetrics.functional.image import peak_signal_noise_ratio as PSNR

from datetime import datetime
import torchvision.transforms as transforms
import numpy as np
import bm3d
import random
import lpips




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='/home/dani/datasets')
    parser.add_argument('--model', type=str, default='IJCVProxGradMHAv2')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--stages', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0001)

    parser.add_argument('--device', type=str, default='cuda:1')

    args = parser.parse_args()

    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    train_set = LOL_paired(path=args.dataset_path, subset='train')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    validation_set = LOL_paired(path=args.dataset_path, subset='validation')
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)


    model = model_dict[args.model](channels=train_set.channels(), batch_size= args.batch_size, stages=args.stages)
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = LossMSElpipsCosineColor(device= args.device)
    loss_lpips = lpips.LPIPS(net='alex')
    loss_lpips.to('cpu')
    today = date.today()

    n_param = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(n_param)

    hour = datetime.now()
    formatted_time = hour.strftime("%H:%M:%S")

    writer = SummaryWriter(f'/home/dani/projects/IJCVProximalGradient/logsv2/LOLv2_model_ok/')
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}', unit='batch'):
            optimizer.zero_grad()
            #selected_patch = random.randint(0, 4 - 1)
            gt_h, gt_l, L_l, R_l = batch
            #gt_h, gt_l, L_l, R_l = gt_h[:, selected_patch, :, :, :], gt_l[:, selected_patch, :, :, :], L_l[:, selected_patch, :, :, :], R_l[:, selected_patch, :, :, :]
            gt_h = gt_h.to(args.device)
            gt_l = gt_l.to(args.device)
            L_l = L_l.to(args.device)
            R_l = R_l.to(args.device)
            oL, oR, oN_p, Itilla, L_stages, R_stages, N_stages = model(gt_l, L_l, R_l)
            image=oR*oL
            loss = criterion(gt_h, image)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        writer.add_scalar('Loss/train', total_loss / len(train_loader), epoch)
        total_loss = 0
        total_lpips = 0
        total_psnr = 0
        for batch in tqdm(validation_loader, unit='batch'):
            gt_h, gt_l, L_l, R_l = batch
            gt_h = gt_h.to(args.device)
            gt_l = gt_l.to(args.device)
            L_l = L_l.to(args.device)
            R_l = R_l.to(args.device)
            with torch.no_grad():
                oL, oR, oN_p, Itilla, L_stages, R_stages, N_stages = model(gt_l, L_l, R_l)
            image=oR*oL
            loss = criterion(gt_h, image)
            image_cpu= image.to('cpu')
            gt_h_cpu= gt_h.to('cpu')
            lpips_value = loss_lpips.forward(gt_h_cpu, image_cpu)
            psnr_value = PSNR(image, gt_h)
            total_lpips += lpips_value
            total_loss += loss.item()
            total_psnr += psnr_value
        writer.add_scalar('Loss/validation', total_loss / len(validation_loader), epoch)
        writer.add_scalar('LPIPS/validation', total_lpips / len(validation_loader), epoch)
        writer.add_scalar('PSNR/validation', total_psnr / len(validation_loader), epoch)

        path= f'/home/dani/projects/IJCVProximalGradient/ckptv2/LOLv2_model_ok/{args.model}_{epoch}.pth'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)



