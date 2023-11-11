import os
import pickle
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset

# to avoid ValueError: Decompressed Data Too Large
ImageFile.LOAD_TRUNCATED_IMAGES = True


class open_set_folds():
    def __init__(self, image_directory, known_list_path, unknown_list_path, num_gallery, num_probe):
        with open(known_list_path, 'rb') as fp:
            known_list = pickle.load(fp)
        with open(unknown_list_path, 'rb') as fp:
            unknown_list = pickle.load(fp)
        num_known = len(known_list)

        # Gallery, Known Probe, Unknown Probe set
        self.G, self.K, self.U = [], [], []

        known_id = 0  # assign id to known identities
        for name in known_list:
            image_list = os.listdir(os.path.join(image_directory, name))
            if len(image_list) < num_gallery + 1:  # cannot be used as known identity
                num_known -= 1
            else:
                image_list = np.random.permutation(image_list).tolist()  # randomly shuffle
                for i in range(num_gallery):
                    image_path = os.path.join(image_directory, name, image_list[i])
                    self.G.append((image_path, known_id))
                for i in range(num_gallery, min(len(image_list), num_gallery+num_probe)):
                    image_path = os.path.join(image_directory, name, image_list[i])
                    self.K.append((image_path, known_id))
                known_id += 1

        for name in unknown_list:
            image_list = os.listdir(os.path.join(image_directory, name))
            image_list = np.random.permutation(image_list).tolist()  # randomly shuffle
            for i in range(min(len(image_list), num_probe)):
                image_path = os.path.join(image_directory, name, image_list[i])
                self.U.append((image_path, num_known))

        self.P = self.K + self.U    # Probe set: Known Probe + Unknown Probe
        self.num_known = num_known  # Number of known identities



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
