import os
import torch
import torchvision.transforms.functional as TF
from PIL import Image


def get_data(opt):
    if opt['task'] == 'visualization':
        # directory is image
        img_q = Image.open(opt['dataset']['query_directory'])
        img_k = Image.open(opt['dataset']['key_directory'])

        img_q = TF.to_tensor(img_q)
        img_k = TF.to_tensor(img_k)

        if opt['dataset']['normalize_preprocess']:
            img_q = TF.normalize(img_q, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            img_k = TF.normalize(img_k, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        img_q = img_q.unsqueeze(dim=0)
        img_k = img_k.unsqueeze(dim=0)

        return img_q, img_k

    elif opt['task'] == 'numeric':
        # directory is folder
        img_q_name_list = sorted(os.listdir(opt['dataset']['query_directory']))
        img_k_name_list = sorted(os.listdir(opt['dataset']['key_directory']))

        assert len(img_q_name_list) == len(img_k_name_list), "Total number of images are different"

        img_q_list = list()
        img_k_list = list()
        for img_q_name, img_k_name in zip(img_q_name_list, img_k_name_list):
            img_q = Image.open(os.path.join(opt['dataset']['query_directory'], img_q_name))
            img_k = Image.open(os.path.join(opt['dataset']['key_directory'], img_k_name))

            img_q = TF.to_tensor(img_q)
            img_k = TF.to_tensor(img_k)

            if opt['dataset']['normalize_preprocess']:
                img_q = TF.normalize(img_q, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                img_k = TF.normalize(img_k, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            img_q_list.append(img_q.unsqueeze(dim=0))
            img_k_list.append(img_k.unsqueeze(dim=0))

        return img_q_list, img_k_list, img_q_name_list, img_k_name_list




