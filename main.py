import config
from dataset import open_set_folds, face_dataset
from model import fetch_encoder, head
from finetune import linear_probing, weight_imprinting, fine_tune
from utils import save_dir_far_curve, save_dir_far_excel

import os
import json
import pprint
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# for boolean parser argument
def str2bool(v):
    if isinstance(v,bool):
        return v
    if v == "True":
        return True
    elif v == "False":
        return False
    else:
        raise argparse.ArgumentTypeError("'True' or 'False' expected")

def False_or_float(v):
    if v == "False":
        return False
    else:
        return float(v)

parser = argparse.ArgumentParser()
# basic arguments
parser.add_argument("--device_id", type=int, default=0)
parser.add_argument("--lr",default=1e-3,type=float)
parser.add_argument("--batch_size",default=128,type=int)
parser.add_argument("--num_epochs",default=20,type=int,help="num_epochs for fine-tuning")

# dataset arguments
parser.add_argument("--dataset", type=str, default='CASIA', help="['CASIA','IJBC']")
parser.add_argument("--num_gallery", type=int, default=3, help="number of gallery images per identity")
parser.add_argument("--num_probe", type=int, default=5, help="maximum number of probe images per identity")

# encoder arguments
parser.add_argument("--encoder", type=str, default='Res50', help="['VGG19','Res50']")
parser.add_argument("--head_type", type=str, default='cos', help="['arc', 'cos', 'mag']")

# main arguments: classifier init / finetune layers / matcher
parser.add_argument("--classifier_init", type=str, default='WI',
                    help="['Random','LP','WI']")  # Random Init. / Linear Probing / Weight Imprinting
parser.add_argument("--finetune_layers", type=str, default='None',
                    help="['None','Full','Partial','PA','BN']")  # 'None' refers to no fine-tuning
parser.add_argument("--matcher", type=str, default='NAC',
                    help="['org','NAC','EVM']")  # unused argument: refer to the results

# misc. arguments: no need to change
parser.add_argument("--arc_s",default=32,type=float,help="scale for ArcFace")
parser.add_argument("--arc_m",default=0.4,type=float,help="margin for ArcFace")
parser.add_argument("--cos_m",default=0.4,type=float,help="margin for CosFace")
parser.add_argument("--train_output",type=str2bool,default=False,
                    help="if True, train output layer")

args = parser.parse_args()



def main(args):
    # check arguments
    assert args.classifier_init in ['Random','LP','WI'], 'classifier_init must be one of ["Random","LP","WI"]'
    assert args.finetune_layers in ['None','Full','Partial','PA','BN'], \
        "finetune_layers must be one of ['None','Full','Partial','PA','BN']"

    # fix random seed
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # set device
    args.device = torch.device(args.device_id)

    # result save directory
    os.makedirs(f'results/{args.dataset}_{args.encoder}', exist_ok=True)
    if args.finetune_layers == 'None':
        exp_name = 'Pretrained'
    else:
        exp_name = f'{args.classifier_init}_{args.finetune_layers}'
    save_dir = f'results/{args.dataset}_{args.encoder}/{exp_name}'
    os.makedirs(save_dir, exist_ok=True)
    print("results are saved at: ", save_dir)

    # save arguments
    argdict = args.__dict__.copy()
    argdict['device'] = argdict['device'].type + f":{argdict['device'].index}"
    with open(save_dir + '/args.txt', 'w') as fp:
        json.dump(argdict, fp, indent=2)


    # prepare dataset: config
    data_config = config.data_config[args.dataset]
    folds = open_set_folds(data_config["image_directory"], data_config["known_list_path"],
                           data_config["unknown_list_path"], args.num_gallery, args.num_probe)

    '''
    data preparation
    '''
    num_cls = folds.num_known
    train_trf = transforms.Compose([
        transforms.RandomResizedCrop(size=112, scale=(0.7, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(torch.FloatTensor([0.5, 0.5, 0.5]), torch.FloatTensor([0.5, 0.5, 0.5])),
    ])
    eval_trf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(torch.FloatTensor([0.5, 0.5, 0.5]), torch.FloatTensor([0.5, 0.5, 0.5])),
    ])
    """
    prepare G, K, U sets for evaluation
    increase batch_size for faster inference
    """
    dataset_gallery = face_dataset(folds.G, eval_trf, img_size=112)
    dataset_probe = face_dataset(folds.P, eval_trf, img_size=112)
    data_loader_gallery = DataLoader(dataset_gallery, batch_size=256, shuffle=False, num_workers=4)
    data_loader_probe = DataLoader(dataset_probe, batch_size=256, shuffle=False, num_workers=4)


    '''
    prepare encoder
    '''
    encoder = fetch_encoder.fetch(args.device, config.encoder_config,
                                  args.encoder, args.finetune_layers, args.train_output)

    '''
    fine-tune
    '''
    if args.finetune_layers != "None":  # for 'None', no fine-tuning is done
        if args.head_type == "arc":
            classifier = head.arcface_head(args.device, 512, num_cls, s=args.arc_s, m=args.arc_m, use_amp=True)
        elif args.head_type == "cos":
            classifier = head.cosface_head(512, num_cls, s=args.arc_s, m=args.cos_m)
        elif args.head_type == "mag":
            classifier = head.magface_head(args.device, 512, num_cls, s=args.arc_s, use_amp=True)
        classifier.to(args.device)

        # classifier initialization
        if args.classifier_init == 'WI':
            prototypes = weight_imprinting(args, encoder, data_loader_gallery, num_cls, 512)
            classifier.weight = nn.Parameter(prototypes.T)
        elif args.classifier_init == 'LP':
            linear_probing(args, data_loader_gallery, encoder, classifier)
        else:
            pass  # just use random weights for classifier


        # set optimizer & LR scheduler
        optimizer = optim.Adam([{"params": encoder.parameters(), "lr": args.lr},
                                {"params": classifier.parameters(), "lr": args.lr}],
                               weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

        # load training dataset (random shuffling & augmentation)
        trainset_gallery = face_dataset(folds.G, train_trf, 112)
        train_loader_gallery = DataLoader(trainset_gallery, batch_size=args.batch_size, shuffle=True, num_workers=4)

        # finetune encoder
        fine_tune(args, train_loader_gallery, encoder, classifier, optimizer, scheduler, verbose=True)

    '''
    evaluate encoder
    '''
    encoder.eval()
    flip = transforms.RandomHorizontalFlip(p=1)
    Gfeat = torch.FloatTensor([]).to(args.device)
    Glabel = torch.LongTensor([])
    for img, label in tqdm(data_loader_gallery):
        img = img.to(args.device)
        with torch.no_grad():
            feat = 0.5 * (encoder(img) + encoder(flip(img)))
        Gfeat = torch.cat((Gfeat, feat), dim=0)
        Glabel = torch.cat((Glabel, label), dim=0)

    Pfeat = torch.FloatTensor([]).to(args.device)
    Plabel = torch.LongTensor([])
    for img, label in tqdm(data_loader_probe):
        img = img.to(args.device)
        with torch.no_grad():
            feat = 0.5 * (encoder(img) + encoder(flip(img)))
        Pfeat = torch.cat((Pfeat, feat), dim=0)
        Plabel = torch.cat((Plabel, label), dim=0)
    Gfeat = Gfeat.cpu()
    Pfeat = Pfeat.cpu()

    # save results
    save_dir_far_curve(Gfeat, Glabel, Pfeat, Plabel, save_dir)
    save_dir_far_excel(Gfeat, Glabel, Pfeat, Plabel, save_dir)


if __name__ == '__main__':
    pprint.pprint(vars(args))
    main(args)
