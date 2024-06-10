import numpy as np
import SimpleITK as sitk
from utils import ResampleXYZAxis, ResampleLabelToRef, CropForeground, ITKReDirection
import os
import random
import yaml
import copy
import pdb

from matplotlib import image

def ResampleImage(imImage, imLabel, save_path, name, target_spacing=(1., 1., 1.)):

    assert round(imImage.GetSpacing()[0], 2) == round(imLabel.GetSpacing()[0], 2)
    assert round(imImage.GetSpacing()[1], 2) == round(imLabel.GetSpacing()[1], 2)
    assert round(imImage.GetSpacing()[2], 2) == round(imLabel.GetSpacing()[2], 2)

    assert imImage.GetSize() == imLabel.GetSize()


    imLabel.CopyInformation(imImage)
    
    imImage = ITKReDirection(imImage, target_direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    imLabel = ITKReDirection(imLabel, target_direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))


    spacing = imImage.GetSpacing()
    origin = imImage.GetOrigin()
    

    npimg = sitk.GetArrayFromImage(imImage).astype(np.int32)
    nplab = sitk.GetArrayFromImage(imLabel).astype(np.uint8)
    z, y, x = npimg.shape

    if not os.path.exists('%s'%(save_path)):
        os.mkdir('%s'%(save_path))
       

    re_img_yz = ResampleXYZAxis(imImage, space=(target_spacing[0], target_spacing[1], spacing[2]), interp=sitk.sitkBSpline)
    re_lab_yz = ResampleLabelToRef(imLabel, re_img_yz, interp=sitk.sitkNearestNeighbor)
    
    re_img_xyz = ResampleXYZAxis(re_img_yz, space=(target_spacing[0], target_spacing[1], target_spacing[2]), interp=sitk.sitkNearestNeighbor)
    re_lab_xyz = ResampleLabelToRef(re_lab_yz, re_img_xyz, interp=sitk.sitkNearestNeighbor)

    np_lab = sitk.GetArrayFromImage(re_lab_xyz)
    np_lab[np_lab == 63] = 1
    np_lab[np_lab == 126] = 2
    np_lab[np_lab == 189] = 3
    np_lab[np_lab == 252] = 4

    re_lab_xyz = sitk.GetImageFromArray(np_lab.astype(np.uint8))
    re_lab_xyz.CopyInformation(re_img_xyz)


    cropped_img, cropped_lab = CropForeground(re_img_xyz, re_lab_xyz, context_size=[20, 20, 20]) # z, y, x

    sitk.WriteImage(cropped_img, '%s/%s.nii.gz'%(save_path, name))
    sitk.WriteImage(cropped_lab, '%s/%s_gt.nii.gz'%(save_path, name))


if __name__ == '__main__':


    src_path = '/research/cbim/medical/yg397/CHAOS/Train_Sets/T1/'
    tgt_path = '/research/cbim/medical/yg397/universal_model/chaos/'

    
    name_list = []
    for i in os.listdir(src_path):
        if i.endswith('gt.nii.gz'):
            name = i.split('_')[0]
            name_list.append(name)

    if not os.path.exists(tgt_path+'list'):
        os.mkdir('%slist'%(tgt_path))
    with open("%slist/dataset.yaml"%tgt_path, "w",encoding="utf-8") as f:
        yaml.dump(name_list, f)
    
    base_dir =  '/research/cbim/medical/yg397/CHAOS/Train_Sets/'
    os.chdir(base_dir)

    for name in name_list:
        img_t1_in = sitk.ReadImage(base_dir+f"T1/{name}_InPhase.nii.gz")
        img_t1_out = sitk.ReadImage(base_dir+f"T1/{name}_OutPhase.nii.gz")
        lab_t1 = sitk.ReadImage(base_dir+f"T1/{name}_gt.nii.gz")
        img_t2 = sitk.ReadImage(base_dir+f"T2/{name}.nii.gz")
        lab_t2 = sitk.ReadImage(base_dir+f"T2/{name}_gt.nii.gz")

        ResampleImage(img_t1_in, lab_t1, tgt_path, f"{name}_t1_in", (1.5, 1.5, 1.5))
        ResampleImage(img_t1_out, lab_t1, tgt_path, f"{name}_t1_out", (1.5, 1.5, 1.5))
        ResampleImage(img_t2, lab_t2, tgt_path, f"{name}_t2", (1.5, 1.5, 1.5))



        #ResampleImage(img, lab, tgt_path, name, (1, 1, 1))
        print(name, 'done')


