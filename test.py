import argparse

import torch
import os
from torch.utils.data import DataLoader
from data.PairedData import LOL_paired
from model import model_dict
from tqdm import tqdm
from torchvision.utils import save_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./datasets')
    parser.add_argument('--data', type=str, default='LOL')
    parser.add_argument('--model', type=str, default='CARNet')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--stages', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda:1')

    args = parser.parse_args()
    target_dir = f"./results/{args.data}"
    weights_path=f"./ckpt/{args.data}/{args.model}.pth"
    os.makedirs(os.path.dirname(target_dir), exist_ok=True)
    validation_set = LOL_paired(path=args.dataset_path, type= args.data, subset='eval')
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)

    model = model_dict[args.model](channels=validation_set.channels(), batch_size=args.batch_size, stages=args.stages)
    model = model.to(args.device)
    ckpt = torch.load(weights_path, map_location=args.device)
    model.load_state_dict(ckpt)

    i=1

    for batch in tqdm(validation_loader, unit='batch'):
        gt, low, L, R = batch
        gt = gt.to(args.device)
        low = low.to(args.device)
        L = L.to(args.device)
        R = R.to(args.device)
        model.eval()
        with torch.no_grad():
            oL_p, oR_p = model(low, L, R)
        image_p = oL_p*oR_p
        filepathI = os.path.join(target_dir, f"{i}.png")
        os.makedirs(os.path.dirname(filepathI), exist_ok=True)
        save_image(image_p, filepathI)
        i=i+1






