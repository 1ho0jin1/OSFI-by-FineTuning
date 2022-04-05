import os
import pickle
from PIL import Image, ImageFile
from torch.utils.data import Dataset

# to avoid ValueError: Decompressed Data Too Large
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Prep_CASIA_IJBC(object):
    """
    interval: interval of images to be used
       ex) [9,12] means we'll be using 9,10,11-th image of that person
    """
    def __init__(self, G_data_dir, K_data_dir, U_data_dir,
                 G_interval, K_interval, U_interval, known_pkl, unknown_pkl):
        with open(known_pkl,"rb") as fp:
            self.knowns = pickle.load(fp)
        with open(unknown_pkl,"rb") as fp:
            self.unknowns = pickle.load(fp)
        self.num_known_classes = len(self.knowns)
        self.num_unknown_classes = len(self.unknowns)
        self.G, self.K, self.U = [], [], []
        Gs, Ge = G_interval
        Ks, Ke = K_interval
        Us, Ue = U_interval
        for Gid, name in enumerate(self.knowns):
            name_dir = os.path.join(G_data_dir, name)
            img_list = sorted(os.listdir(name_dir))
            for cnt, img_name in enumerate(img_list):
                if cnt in range(Gs, Ge):
                    img_dir = os.path.join(name_dir, img_name)
                    self.G.append((img_dir,Gid))
        for Kid, name in enumerate(self.knowns):
            name_dir = os.path.join(K_data_dir, name)
            img_list = sorted(os.listdir(name_dir))
            for cnt, img_name in enumerate(img_list):
                if cnt in range(Ks, Ke):
                    img_dir = os.path.join(name_dir, img_name)
                    self.K.append((img_dir,Kid))
        for name in self.unknowns:
            name_dir = os.path.join(U_data_dir, name)
            img_list = sorted(os.listdir(name_dir))
            for cnt, img_name in enumerate(img_list):
                if cnt in range(Us, Ue):
                    img_dir = os.path.join(name_dir, img_name)
                    self.U.append((img_dir,self.num_known_classes))
        self.P = self.K + self.U



class face_dataset(Dataset):
    def __init__(self, data_fold, transform, img_size):
        self.data_fold = data_fold
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.data_fold)

    def __getitem__(self, index):
        image_path, label = self.data_fold[index]
        image = Image.open(image_path).resize((self.img_size, self.img_size))
        if image.mode == 'L':
            image = image.convert("RGB")
        image = self.transform(image)
        return image, label