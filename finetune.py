import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms



def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def linear_probing(args, galleryloader, encoder, classifier,verbose=True, target_acc=95):
    CEloss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,args.num_epochs)
    print("start classifier training!")
    for epoch in range(args.num_epochs):
        train_corr, train_tot = 0, 0
        for img, label in galleryloader:
            img, label = img.to(args.device), label.to(args.device)
            with torch.no_grad():
                feat = encoder(img)
            logit, sim = classifier(feat, label)
            loss = CEloss(logit, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            corr = torch.argmax(sim, dim=1).eq(label).sum().item()
            train_corr += corr
            train_tot += label.size(0)
        train_acc = train_corr / train_tot * 100
        scheduler.step()
        if verbose:
            print("epoch:{}, loss:{:.2f}, acc:{:.2f}%,  lr:{:.2e}".format(epoch, loss.item(),
                                                                          train_acc, get_lr(optimizer)))
        if train_acc > target_acc:
            if verbose:
                print("acc:{:.2f}%, target accuracy met. training finished".format(train_acc))
            break


def weight_imprinting(args, encoder, G_loader, num_cls, feat_dim):
    flip = transforms.RandomHorizontalFlip(p=1)
    encoder.eval()
    prototypes = torch.zeros(num_cls, feat_dim).to(args.device)
    with torch.no_grad():
        for batch, (img, label) in enumerate(G_loader):
            img, label = img.to(args.device), label.to(args.device)
            feat = 0.5 * (encoder(img) + encoder(flip(img)))
            for i in range(label.size(0)):
                prototypes[label[i]] += feat[i]
    prototypes = F.normalize(prototypes, dim=1)

    return prototypes


def fine_tune(args, trainloader, encoder, classifier,
              optimizer, scheduler, verbose=True):
    CEloss = nn.CrossEntropyLoss()
    for epoch in range(args.num_epochs):
        train_corr, train_tot = 0, 0
        for img, label in trainloader:
            img, label = img.to(args.device), label.to(args.device)
            with torch.cuda.amp.autocast():
                feat = encoder(img)
                if args.head_type == "mag":  # if using MagFace head
                    logit, sim, loss_g = classifier(feat, label)
                else:
                    logit, sim = classifier(feat, label)
                loss = CEloss(logit, label)
                if args.head_type == "mag":
                    loss += loss_g
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            corr = torch.argmax(sim, dim=1).eq(label).sum().item()
            train_corr += corr
            train_tot += label.size(0)
        train_acc = train_corr / train_tot * 100
        scheduler.step()
        if verbose:
            print("epoch:{}, loss:{:.2f}, acc:{:.2f}%,  lr:{:.2e}".format(epoch, loss.item(),
                                                                          train_acc, get_lr(optimizer)))