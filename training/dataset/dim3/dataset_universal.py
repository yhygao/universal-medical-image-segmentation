import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk
import yaml
import math
import random
import pdb
from training import augmentation
from .dataset_config import train_test_split, dataset_lab_map, dataset_modality_map, dataset_sample_weight, dataset_aug_prob


class UniversalDataset(Dataset):
    def __init__(self, 
                args, 
                dataset_list = ['lits', 'kits19'], 
                mode='train'
            ):
        
        self.mode = mode
        self.args = args

        assert mode in ['train', 'val', 'test']

        self.max_classes = max(args.dataset_classes_list) # max number of classes among all datasets

        all_name_list = []        

        for dataset in dataset_list:
            all_name_list.append(train_test_split[f"{dataset}_{mode}"])

        
        
        path = args.data_root

        self.img_list = []
        self.lab_list = []
        self.spacing_list = []
        self.tgt_list = []
        self.mod_list = []
        self.dataset_name_list = []
        self.weight_list = []

        for idx in range(len(dataset_list)):
            dataset_name = dataset_list[idx]
            print(f"Start loading {dataset_name} {mode} set")
            for i in range(len(all_name_list[idx])):
                if dataset_name == 'chaos':
                    for prefix in ['t1_in', 't1_out', 't2']:
                        img_path = os.path.join(path, dataset_name, f"{all_name_list[idx][i]}_{prefix}.npy")
                        lab_path = os.path.join(path, dataset_name, f"{all_name_list[idx][i]}_{prefix}_gt.npy")

                        self.spacing_list.append((1.5, 1.5, 1.5))  # itk axis order is inverse of numpy axis order

                        self.img_list.append(img_path)
                        self.lab_list.append(lab_path)
                        self.weight_list.append(dataset_sample_weight[dataset_name])
                        self.mod_list.append(dataset_modality_map[dataset_name+'_'+prefix])
                        self.tgt_list.append(np.array(dataset_lab_map[dataset_name]))
                        self.dataset_name_list.append(dataset_name)
                elif dataset_name == 'mnm':
                    for prefix in [0, 1]:
                        img_path = os.path.join(path, dataset_name, f"{all_name_list[idx][i]}_{prefix}.npy")
                        lab_path = os.path.join(path, dataset_name, f"{all_name_list[idx][i]}_{prefix}_gt.npy")

                        self.spacing_list.append((1.5, 1.5, 1.5))  # itk axis order is inverse of numpy axis order

                        self.img_list.append(img_path)
                        self.lab_list.append(lab_path)
                        self.weight_list.append(dataset_sample_weight[dataset_name])
                        self.mod_list.append(dataset_modality_map[dataset_name])
                        self.tgt_list.append(np.array(dataset_lab_map[dataset_name]))
                        self.dataset_name_list.append(dataset_name)
                else: 
                    img_path = os.path.join(path, dataset_name, f"{all_name_list[idx][i]}.npy")
                    lab_path = os.path.join(path, dataset_name, f"{all_name_list[idx][i]}_gt.npy")

                    self.spacing_list.append((1.5, 1.5, 1.5))  # itk axis order is inverse of numpy axis order

                    self.img_list.append(img_path)
                    self.lab_list.append(lab_path)
                    self.weight_list.append(dataset_sample_weight[dataset_name])
                    self.mod_list.append(dataset_modality_map[dataset_name])
                    self.tgt_list.append(np.array(dataset_lab_map[dataset_name]))
                    self.dataset_name_list.append(dataset_name)
            
            print(f"Finish loading {dataset_name}")
        self.weight_list = self.weight_list# this need to be the same as in __len__
        print('All datasets load done, length of dataset:', len(self.img_list))

    def __len__(self):
        if self.mode == 'train':
            return len(self.img_list) * 10
        else:
            return len(self.img_list)



    def __getitem__(self, idx):
        
        idx = idx % len(self.img_list)

        np_img = np.load(self.img_list[idx], mmap_mode='r', allow_pickle=False)
        np_lab = np.load(self.lab_list[idx], mmap_mode='r', allow_pickle=False)
        
        '''
        np_img = np.random.rand(320, 367, 387)
        np_lab = np.random.rand(320, 367, 387)
        np_lab[np_lab>0.5] = 1
        np_lab[np_lab<0.5] = 0
        np_lab = np_lab.astype(np.int8)
        '''

        if self.mode == 'train':
            d, h, w= self.args.training_size

            np_img, np_lab = augmentation.np_crop_3d(np_img, np_lab, [d+30, h+30, w+30], mode='random')

            tensor_img = torch.from_numpy(np_img).unsqueeze(0).unsqueeze(0)
            tensor_lab = torch.from_numpy(np_lab).unsqueeze(0).unsqueeze(0)
            # 1, C, D, H, W

            if self.args.aug_device == 'gpu':
                tensor_img = tensor_img.cuda(self.args.proc_idx).float()
                tensor_lab = tensor_lab.cuda(self.args.proc_idx).long()
            
            dataset_name = self.dataset_name_list[idx]
            affine_prob = 0.5
            color_prob = dataset_aug_prob[dataset_name]

            if np.random.random() < affine_prob:
                # crop trick for faster augmentation
                # crop a sub volume for scaling and rotation
                # instead of scaling and rotating the whole image
                tensor_img, tensor_lab = augmentation.random_scale_rotate_translate_3d(tensor_img, tensor_lab, self.args.scale, self.args.rotate, self.args.translate)
                tensor_img, tensor_lab = augmentation.crop_3d(tensor_img, tensor_lab, self.args.training_size, mode='center')
            else:
                 tensor_img, tensor_lab = augmentation.crop_3d(tensor_img, tensor_lab, self.args.training_size, mode='random')


            if np.random.random() < color_prob:
                tensor_img = augmentation.brightness_multiply(tensor_img, multiply_range=[0.7, 1.3]) #0.7 1.3
            if np.random.random() < color_prob:
                tensor_img = augmentation.brightness_additive(tensor_img, std=0.1)
            if np.random.random() < color_prob:
                tensor_img = augmentation.gamma(tensor_img, gamma_range=[0.7, 1.5]) #0.7 1.5
            if np.random.random() < color_prob:
                tensor_img = augmentation.contrast(tensor_img, contrast_range=[0.7, 1.3]) #0.7 1.3
            if np.random.random() < color_prob:
                tensor_img = augmentation.gaussian_blur(tensor_img, sigma_range=[0.5, 1.0]) #0.5 1.5
            if np.random.random() < color_prob:
                std = np.random.random() * 0.1 #0.2
                tensor_img = augmentation.gaussian_noise(tensor_img, std=std)

        else:
            tensor_img = torch.from_numpy(np_img).unsqueeze(0).unsqueeze(0).float()
            tensor_lab = torch.from_numpy(np_lab).unsqueeze(0).unsqueeze(0).to(torch.int8)
 

        tensor_img = tensor_img.squeeze(0)
        tensor_lab = tensor_lab.squeeze(0)
        
        assert tensor_img.shape == tensor_lab.shape

        tensor_lab = self.label2binary(tensor_lab.long())
        tgt = self.tgt_list[idx]
        
        # For datasets with less classes, pad there task priors with null class (-1), such that a batch can include images from different datasets
        diff = self.max_classes + 1 - len(tgt) 
        tgt = np.pad(tgt, (0, diff), constant_values=-1)

        if self.mode == 'train':
            return tensor_img, tensor_lab.to(torch.int8), torch.from_numpy(tgt).to(tensor_img.device), torch.from_numpy(np.array(self.mod_list[idx])).to(tensor_img.device)
        else:
            return tensor_img, tensor_lab, torch.from_numpy(tgt), torch.from_numpy(np.array(self.mod_list[idx])), np.array(self.spacing_list[idx])

    
    def label2binary(self, tensor_lab):
        # map the original label (0, 1, 2, ...)  to binary label maps
        # need modification. Can't handle label index that doesn't follow accent order

        _, D, H, W = tensor_lab.shape
        
        class_mask = torch.zeros([self.max_classes+2, D, H, W]).to(torch.int8).to(tensor_lab.device) # max_classes + 2 (background and one null class)
        class_mask.scatter_(0, tensor_lab, 1)
        class_mask = class_mask[1:, :, :, :] # exclude background
        return class_mask.to(torch.int8)

           


if __name__ == '__main__':
    
    class Args(object):
        def __init__(self):
            self.data_root = '/research/cbim/vast/yg397/universal_model/dataset/'
    
    args = Args()
    print('testing')
    dataset = UniversalDataset(args)

    print('done')

