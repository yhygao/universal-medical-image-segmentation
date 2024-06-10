import numpy as np
import SimpleITK as sitk
from utils import ResampleXYZAxis, ResampleLabelToRef, ITKReDirection
import os
import random
import yaml
import copy
import pdb
from skimage.measure import regionprops

def CropForeground(imImage, imLabel, context_size=[20, 20, 20]):                                                                  
    # the context_size is in numpy indice order: z, y, x
    # Note that SimpleITK use the indice order of: x, y, z
    
    npImg = sitk.GetArrayFromImage(imImage)
    npLab = sitk.GetArrayFromImage(imLabel)

    mask = (npLab>0).astype(np.uint8) # foreground mask
    
    regions = regionprops(mask)
    assert len(regions) == 1 or len(regions) == 0

    if len(regions) == 0:
        return imImage, imLabel

    else:

        zz, yy, xx = npImg.shape

        z, y, x = regions[0].centroid

        z_min, y_min, x_min, z_max, y_max, x_max = regions[0].bbox
        print(zz, yy, xx) 
        print('forground size:', z_max-z_min, y_max-y_min, x_max-x_min)
        
        if z_max - z_min < 128:
            context_size[0] = context_size[0] + (128 - (z_max - z_min)) // 2
        if y_max - y_min < 128:
            context_size[1] = context_size[1] + (128 - (y_max - y_min)) // 2
        if x_max - x_min < 128:
            context_size[2] = context_size[2] + (128 - (x_max - x_min)) // 2

        z, y, x = int(z), int(y), int(x)

        z_min = max(0, z_min-context_size[0])
        z_max = min(zz, z_max+context_size[0])
        y_min = max(0, y_min-context_size[1])
        y_max = min(yy, y_max+context_size[1])
        x_min = max(0, x_min-context_size[2])
        x_max = min(xx, x_max+context_size[2])

        img = npImg[z_min:z_max, y_min:y_max, x_min:x_max]
        lab = npLab[z_min:z_max, y_min:y_max, x_min:x_max]
        
        print('after crop size', lab.shape)

        croppedImage = sitk.GetImageFromArray(img)
        croppedLabel = sitk.GetImageFromArray(lab)


        croppedImage.SetSpacing(imImage.GetSpacing())
        croppedLabel.SetSpacing(imImage.GetSpacing())
        
        croppedImage.SetDirection(imImage.GetDirection())
        croppedLabel.SetDirection(imImage.GetDirection())

        return croppedImage, croppedLabel



def ResampleImage(imImage, imLabel, save_path, name, target_spacing=(1., 1., 1.)):

    assert imImage.GetSpacing() == imLabel.GetSpacing()
    assert imImage.GetSize() == imLabel.GetSize()

    #imImage = ITKReDirection(imImage, target_direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), interp=sitk.sitkBSpline)
    #imLabel = ITKReDirection(imLabel, target_direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), interp=sitk.sitkNearestNeighbor)

    spacing = imImage.GetSpacing()
    origin = imImage.GetOrigin()


    npimg = sitk.GetArrayFromImage(imImage).astype(np.int32)
    nplab = sitk.GetArrayFromImage(imLabel).astype(np.uint8)
    z, y, x = npimg.shape

    if not os.path.exists('%s'%(save_path)):
        os.mkdir('%s'%(save_path))


    re_img_xyz = ResampleXYZAxis(imImage, space=(target_spacing[0], target_spacing[1], spacing[2]), interp=sitk.sitkBSpline)
    re_lab_xyz = ResampleLabelToRef(imLabel, re_img_xyz, interp=sitk.sitkNearestNeighbor)


    cropped_img, cropped_lab = CropForeground(re_img_xyz, re_lab_xyz, context_size=[20, 30, 30])

    sitk.WriteImage(cropped_img, '%s/%s.nii.gz'%(save_path, name))
    sitk.WriteImage(cropped_lab, '%s/%s_gt.nii.gz'%(save_path, name))


if __name__ == '__main__':


    src_path = '/research/cbim/medical/yg397/autoPET/autoPET_nii/'
    tgt_path = '/research/cbim/medical/yg397/universal_model/autopet/'


    name_list = []
    name_path_dict = {}

    for name in os.listdir(src_path):
        count = 0
        for study in os.listdir(f"{src_path}{name}"):
            name_list.append(f"{name}_{count}")
            name_path_dict[f"{name}_{count}"] = f"{src_path}{name}/{study}/"
            count += 1

    if not os.path.exists(tgt_path+'list'):
        os.mkdir('%slist'%(tgt_path))
    with open("%slist/dataset.yaml"%tgt_path, "w",encoding="utf-8") as f:
        yaml.dump(name_list, f)

    os.chdir(src_path)

    for name in name_list[::-1]:
        path = name_path_dict[name]

        img_name = path + 'SUV.nii.gz'
        lab_name = path + 'SEG.nii.gz'

        img = sitk.ReadImage(img_name)
        lab = sitk.ReadImage(lab_name)

        ResampleImage(img, lab, tgt_path, name, (1.5, 1.5, 1.5))
        print(name, 'done')


