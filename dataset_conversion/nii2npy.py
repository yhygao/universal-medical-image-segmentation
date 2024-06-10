import SimpleITK as sitk
import numpy as np
import os
import shutil
import math

def pad(img, lab):
    z, y, x = img.shape
    # pad if the image size is smaller than trainig size
    if z < 128:
        diff = int(math.ceil((128. - z) / 2)) 
        img = np.pad(img, ((diff, diff), (0,0), (0,0)))
        lab = np.pad(lab, ((diff, diff), (0,0), (0,0)))
    if y < 128:
        diff = int(math.ceil((128. - y) / 2)) 
        img = np.pad(img, ((0,0), (diff,diff), (0,0)))
        lab = np.pad(lab, ((0,0), (diff, diff), (0,0)))
    if x < 128:
        diff = int(math.ceil((128. - x) / 2)) 
        img = np.pad(img, ((0,0), (0,0), (diff, diff)))
        lab = np.pad(lab, ((0,0), (0,0), (diff, diff)))

    return img, lab

dataset_list = [
            ('amos_ct', 'ct'),
            ('amos_mr', 'mr'),
            ('bcv', 'ct'),
            ('structseg_oar', 'ct'),
            ('lits', 'ct'),
            ('kits', 'ct'),
            ('chaos', 'mr'),
            ('structseg_head_oar', 'ct'),
            ('mnm', 'mr'),
            ('brain_structure', 'mr'),
            ('autopet', 'pet'),
            ]

source_path = './'
target_path = 'npy'

for dataset, modality in dataset_list:
    
    shutil.copytree(os.path.join(source_path, dataset, 'list'), os.path.join(target_path, dataset, 'list'))
    for name in os.listdir(os.path.join(source_path, dataset)):
        if 'gt' in name:
            idx = name.split('_gt')[0]

            img = sitk.ReadImage(os.path.join(source_path, dataset, f"{idx}.nii.gz"))
            img = sitk.GetArrayFromImage(img).astype(np.float32)
            lab = sitk.ReadImage(os.path.join(source_path, dataset, f"{idx}_gt.nii.gz"))
            lab = sitk.GetArrayFromImage(lab).astype(np.int8)
           
            if modality == 'ct':
                img = np.clip(img, -991, 500)
            else:
                percentile_2 = np.percentile(img, 2, axis=None)
                percentile_98 = np.percentile(img, 98, axis=None)
                img = np.clip(img, percentile_2, percentile_98)
                
            mean = np.mean(img)
            std = np.std(img)

            img -= mean
            img /= std

            img, lab = pad(img, lab)
            
            if dataset == 'amos_mr':
                lab[lab == 14] = 0
                lab[lab == 15] = 0

            img, lab = img.astype(np.float32), lab.astype(np.int8)
            
            np.save(os.path.join(target_path, dataset, f"{idx}.npy", img)
            np.save(os.path.join(target_path, dataset, f"{idx}_gt.npy", lab)

            print(name)

