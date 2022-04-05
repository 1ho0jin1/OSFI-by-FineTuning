import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def cosine(x,w):
    # x, w shape: [B, d], where B = batch size, d = feature dim.
    x_norm = F.normalize(x,dim=1)
    w_norm = F.normalize(w,dim=1)
    cos_sim = torch.mm(x_norm, w_norm.T).clamp(-1, 1)
    return cos_sim


def NAC(sim,k=16,s=1):
    """
    Neighborhood Aware Cosine (NAC) matcher
    args:
        sim: cosine similarity
        k: k for kNN
        s: scale (=1/T)  # In the paper scale is not used (no difference)
    returns:
        conf, pred : confidence and predicted class of shape [B,]
    """
    logit, pred = sim.topk(k,dim=1) # logit, label: [B,k]
    conf = (logit*s).softmax(1)     # norm. scale. >> use largest as conf.
    return conf[:,0],pred[:,0]      # return Top-1 confidence & prediction


def compute_dir_far(Gfeat, Glabel, Pfeat, Plabel, matcher="cos", nac_s=1, nac_k=16):
    num_cls = Plabel[-1].item()
    temp = torch.zeros(num_cls, Gfeat.size(1))
    for i in range(num_cls):
        mask = Glabel.eq(i)
        temp[i] = Gfeat[mask].mean(dim=0)  # make 1x512 vector
    Gfeat = temp.clone()

    num_cls = Plabel[-1].item()
    Umask = Plabel.eq(num_cls)
    Klabel = Plabel[~Umask]
    Kfeat = Pfeat[~Umask]
    Ufeat = Pfeat[Umask]

    # compute cosine similarity
    Kcos = cosine(Kfeat, Gfeat)
    Ucos = cosine(Ufeat, Gfeat)

    # get prediction & confidence
    if matcher == "cos":
        Kconf, Kidx = Kcos.max(1)
        Uconf, _ = Ucos.max(1)
    elif matcher == "NAC":
        Kconf, Kidx = NAC(Kcos, k=nac_k, s=nac_s)
        Uconf, _ = NAC(Ucos, k=nac_k, s=nac_s)

    corr_mask = Kidx.eq(Klabel)
    dir_far_tensor = torch.zeros(1000, 3)  # intervals: 1000
    for i, th in enumerate(torch.linspace(Uconf.min(), Uconf.max(), 1000)):
        mask = (corr_mask) & (Kconf > th)
        dir = torch.sum(mask).item() / Kcos.size(0)
        far = torch.sum(Uconf > th).item() / Ucos.size(0)
        dir_far_tensor[i] = torch.FloatTensor([th, dir, far])  # [threshold, DIR, FAR] for each row
    return dir_far_tensor


def dir_at_far(dir_far_tensor,far):
    # deal with exceptions: there can be multiple thresholds that meets the given FAR (e.g., FAR=1.000)
    # if so, we must choose maximum DIR value among those cases
    abs_diff = torch.abs(dir_far_tensor[:,2]-far)
    minval = abs_diff.min()
    mask = abs_diff.eq(minval)
    dir_far = dir_far_tensor[mask]
    dir = dir_far[:,1].max().item()
    return dir


# area under DIR@FAR curve
def AUC(dir_far_tensor):
    auc = 0
    eps = 1e-5
    for i in range(dir_far_tensor.size(0)-1):
        if dir_far_tensor[i,1].ge(eps) and dir_far_tensor[i,2].ge(eps)\
                and dir_far_tensor[i+1,1].ge(eps) and dir_far_tensor[i+1,2].ge(eps):
            height = (dir_far_tensor[i,1] + dir_far_tensor[i+1,1])/2
            width = torch.abs(dir_far_tensor[i,2] - dir_far_tensor[i+1,2])
            auc += (height*width).item()
    return auc


def save_dir_far_curve(Gfeat, Glabel, Pfeat, Plabel, save_dir):
    cos_tensor = compute_dir_far(Gfeat, Glabel, Pfeat, Plabel, matcher='cos')
    nac_tensor = compute_dir_far(Gfeat, Glabel, Pfeat, Plabel, matcher='NAC')
    cos_auc = AUC(cos_tensor)
    nac_auc = AUC(nac_tensor)
    fig,ax = plt.subplots(1,1)
    ax.plot(cos_tensor[:,2], cos_tensor[:,1])
    ax.plot(nac_tensor[:,2], nac_tensor[:,1])
    ax.set_xscale('log')
    ax.set_xlabel('FAR')
    ax.set_ylabel('DIR')
    ax.legend(['cos-AUC: {:.3f}'.format(cos_auc),
               'NAC-AUC: {:.3f}'.format(nac_auc)])
    ax.grid()
    fig.savefig(save_dir+'/DIR_FAR_curve.png', bbox_inches='tight')


def save_dir_far_excel(Gfeat, Glabel, Pfeat, Plabel, save_dir):
    cos_tensor = compute_dir_far(Gfeat, Glabel, Pfeat, Plabel, matcher='cos')
    nac_tensor = compute_dir_far(Gfeat, Glabel, Pfeat, Plabel, matcher='NAC')
    cos_list, nac_list = [], []
    for far in [0.001, 0.01, 0.1, 1.0]:
        cos_list.append('{:.2f}%'.format(dir_at_far(cos_tensor, far) * 100))
        nac_list.append('{:.2f}%'.format(dir_at_far(nac_tensor, far) * 100))
    cos_list = np.array(cos_list).reshape(1,4)
    nac_list = np.array(nac_list).reshape(1,4)
    dir_far = np.concatenate((cos_list, nac_list),axis=0)

    # save as excel
    columns = ['{:.1f}'.format(far) for far in [0.1, 1, 10, 100]]
    index = ['cos', 'NAC']
    df = pd.DataFrame(dir_far, index=index, columns=columns)
    df.to_excel(save_dir+'/DIR_FAR.xls')