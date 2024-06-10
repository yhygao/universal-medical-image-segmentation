import numpy as np
import SimpleITK as sitk
from utils import ResampleXYZAxis, ResampleLabelToRef, CropForeground_TwoLab, ITKReDirection
import os
import random
import yaml
import copy
import pdb

from matplotlib import image

def ResampleImage(imImage, imLabel, imGTVLabel, save_path, name, target_spacing=(1., 1., 1.)):

    assert round(imImage.GetSpacing()[0], 2) == round(imLabel.GetSpacing()[0], 2)
    assert round(imImage.GetSpacing()[1], 2) == round(imLabel.GetSpacing()[1], 2)
    assert round(imImage.GetSpacing()[2], 2) == round(imLabel.GetSpacing()[2], 2)

    assert imImage.GetSize() == imLabel.GetSize()


    imLabel.CopyInformation(imImage)
    imGTVLabel.CopyInformation(imImage)
    
    imImage = ITKReDirection(imImage, target_direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    imLabel = ITKReDirection(imLabel, target_direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    imGTVLabel = ITKReDirection(imGTVLabel, target_direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))


    spacing = imImage.GetSpacing()
    origin = imImage.GetOrigin()
    

    npimg = sitk.GetArrayFromImage(imImage).astype(np.int32)
    nplab = sitk.GetArrayFromImage(imLabel).astype(np.uint8)
    z, y, x = npimg.shape

    if not os.path.exists('%s'%(save_path)):
        os.mkdir('%s'%(save_path))
       

    re_img_yz = ResampleXYZAxis(imImage, space=(target_spacing[0], target_spacing[1], spacing[2]), interp=sitk.sitkBSpline)
    re_lab_yz = ResampleLabelToRef(imLabel, re_img_yz, interp=sitk.sitkNearestNeighbor)
    re_gtv_lab_yz = ResampleLabelToRef(imGTVLabel, re_img_yz, interp=sitk.sitkNearestNeighbor)

    
    re_img_xyz = ResampleXYZAxis(re_img_yz, space=(target_spacing[0], target_spacing[1], target_spacing[2]), interp=sitk.sitkNearestNeighbor)
    re_lab_xyz = ResampleLabelToRef(re_lab_yz, re_img_xyz, interp=sitk.sitkNearestNeighbor)
    re_gtv_lab_xyz = ResampleLabelToRef(re_gtv_lab_yz, re_img_xyz, interp=sitk.sitkNearestNeighbor)


    cropped_img, cropped_lab, cropped_gtv_lab = CropForeground_TwoLab(re_img_xyz, re_lab_xyz, re_gtv_lab_xyz, context_size=[20, 20, 20]) # z, y, x

    sitk.WriteImage(cropped_img, '%s/structseg_head_oar/%s.nii.gz'%(save_path, name))
    sitk.WriteImage(cropped_lab, '%s/structseg_head_oar/%s_gt.nii.gz'%(save_path, name))
    sitk.WriteImage(cropped_img, '%s/structseg_head_gtv/%s.nii.gz'%(save_path, name))
    sitk.WriteImage(cropped_gtv_lab, '%s/structseg_head_gtv/%s_gt.nii.gz'%(save_path, name))

if __name__ == '__main__':


    src_path = '/research/cbim/medical/yg397/StructSeg/'
    tgt_path = '/research/cbim/medical/yg397/universal_model/'

    
    name_list = []
    for i in range(1, 51):
        name_list.append(i)

    if not os.path.exists(tgt_path+'structseg_head_oar/list'):
        os.mkdir('%sstructseg_head_oar/list'%(tgt_path))
    with open("%sstructseg_head_oar/list/dataset.yaml"%tgt_path, "w",encoding="utf-8") as f:
        yaml.dump(name_list, f)
    
    if not os.path.exists(tgt_path+'structseg_head_gtv/list'):
        os.mkdir('%sstructseg_head_gtv/list'%(tgt_path))
    with open("%sstructseg_head_gtv/list/dataset.yaml"%tgt_path, "w",encoding="utf-8") as f:
        yaml.dump(name_list, f)

    os.chdir(src_path)

    for name in name_list:
        img = sitk.ReadImage(src_path+f"HaN_OAR/{name}/data.nii.gz")
        lab = sitk.ReadImage(src_path+f"HaN_OAR/{name}/label.nii.gz")
        gtv_lab = sitk.ReadImage(src_path+f"Naso_GTV/{name}/label.nii.gz")

        ResampleImage(img, lab, gtv_lab, tgt_path, name, (1.5, 1.5, 1.5))
        #ResampleImage(img, lab, tgt_path, name, (1, 1, 1))
        print(name, 'done')


