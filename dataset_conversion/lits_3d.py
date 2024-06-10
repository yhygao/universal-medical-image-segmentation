import numpy as np
import SimpleITK as sitk
from utils import ResampleXYZAxis, ResampleLabelToRef, CropForeground, ITKReDirection
import os
import random
import yaml
import copy
import pdb

def ResampleImage(imImage, imLabel, save_path, name, target_spacing=(1., 1., 1.)):

    #assert round(imImage.GetSpacing()[0], 2) == round(imLabel.GetSpacing()[0], 2)
    #assert round(imImage.GetSpacing()[1], 2) == round(imLabel.GetSpacing()[1], 2)
    #assert round(imImage.GetSpacing()[2], 2) == round(imLabel.GetSpacing()[2], 2)

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


    re_img_xy = ResampleXYZAxis(imImage, space=(target_spacing[0], target_spacing[1], spacing[2]), interp=sitk.sitkBSpline)
    re_lab_xy = ResampleLabelToRef(imLabel, re_img_xy, interp=sitk.sitkNearestNeighbor)
    
    re_img_xyz = ResampleXYZAxis(re_img_xy, space=(target_spacing[0], target_spacing[1], target_spacing[2]), interp=sitk.sitkNearestNeighbor)
    re_lab_xyz = ResampleLabelToRef(re_lab_xy, re_img_xyz, interp=sitk.sitkNearestNeighbor)


    

    cropped_img, cropped_lab = CropForeground(re_img_xyz, re_lab_xyz, context_size=[20, 20, 20])

    sitk.WriteImage(cropped_img, '%s/%s.nii.gz'%(save_path, name))
    sitk.WriteImage(cropped_lab, '%s/%s_gt.nii.gz'%(save_path, name))


if __name__ == '__main__':


    src_path = '/research/cbim/medical/yg397/LiTS/data/'
    tgt_path = '/research/cbim/medical/yg397/universal_model/lits/'

    
    name_list = []
    for i in range(0, 131):
        name_list.append(i)

    if not os.path.exists(tgt_path+'list'):
        os.mkdir('%slist'%(tgt_path))
    with open("%slist/dataset.yaml"%tgt_path, "w",encoding="utf-8") as f:
        yaml.dump(name_list, f)

    os.chdir(src_path)

    count = 0
    spacing = [1.7, 2.5, 1.5, 1.8, 1.7, 1.9, 2, 1.7, 1.7, 1.7, 1.8, 1.6, 1.8, 2, 2.5, 2.5, 1.7]
    #for name in [28, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47]:
    for name in name_list:
        img_name = 'volume-%d.nii'%name
        lab_name = 'segmentation-%d.nii'%name

        img = sitk.ReadImage(src_path+img_name)
        lab = sitk.ReadImage(src_path+lab_name)
        
        
        if name in [28, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47]:
            img.SetSpacing([1, 1, spacing[count]])
            lab.SetSpacing([1, 1, spacing[count]])
            count += 1
        ResampleImage(img, lab, tgt_path, name, (1.5, 1.5, 1.5))
        #ResampleImage(img, lab, tgt_path, name, (1, 1, 1))
        print(name, 'done')


