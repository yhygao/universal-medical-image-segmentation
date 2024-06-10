import numpy as np
import SimpleITK as sitk
from utils import ResampleXYZAxis, ResampleLabelToRef, CropForeground, ITKReDirection
import os
import random
import yaml
import copy
import pdb

from matplotlib import image

def ResampleImage(imImage, imLabel_gm, imLabel_wm, imLabel_csf, save_path, name, target_spacing=(1., 1., 1.)):

    assert imImage.GetSize() == imLabel_gm.GetSize()


    #imImage = ITKReDirection(imImage, target_direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    #imLabel_gm = ITKReDirection(imLabel_gm, target_direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    #imLabel_wm = ITKReDirection(imLabel_wm, target_direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    #imLabel_csf = ITKReDirection(imLabel_csf, target_direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))


    spacing = imImage.GetSpacing()
    origin = imImage.GetOrigin()
    

    npimg = sitk.GetArrayFromImage(imImage).astype(np.int32)
    z, y, x = npimg.shape

    if not os.path.exists('%s'%(save_path)):
        os.mkdir('%s'%(save_path))
       

    re_img_xyz = ResampleXYZAxis(imImage, space=(target_spacing[0], target_spacing[1], target_spacing[2]), interp=sitk.sitkBSpline)
    re_lab_gm = ResampleLabelToRef(imLabel_gm, re_img_xyz, interp=sitk.sitkLinear)
    re_lab_wm = ResampleLabelToRef(imLabel_wm, re_img_xyz, interp=sitk.sitkLinear)
    re_lab_csf = ResampleLabelToRef(imLabel_csf, re_img_xyz, interp=sitk.sitkLinear)



    #sitk.sitkNearestNeighbor)
    
    nplab_gm = sitk.GetArrayFromImage(re_lab_gm).astype(np.float32)
    nplab_wm = sitk.GetArrayFromImage(re_lab_wm).astype(np.float32)
    nplab_csf = sitk.GetArrayFromImage(re_lab_csf).astype(np.float32)

    nplab_bg = np.ones(nplab_gm.shape)
    nplab_bg[nplab_gm > 0] = 0
    nplab_bg[nplab_wm > 0] = 0
    nplab_bg[nplab_csf > 0] = 0

    nplab = np.stack([nplab_bg, nplab_gm, nplab_wm, nplab_csf])
    nplab = np.argmax(nplab, axis=0).astype(np.uint8)


    imLabel = sitk.GetImageFromArray(nplab)
    imLabel.CopyInformation(re_img_xyz)


    #cropped_img, cropped_lab = CropForeground(re_img_xyz, re_lab_xyz, context_size=[10, 30, 30]) # z, y, x

    sitk.WriteImage(re_img_xyz, '%s%s.nii.gz'%(save_path, name))
    sitk.WriteImage(imLabel, '%s%s_gt.nii.gz'%(save_path, name))


if __name__ == '__main__':


    src_path = '/research/cbim/medical/yg397/brain_structure/'
    tgt_path = '/research/cbim/medical/yg397/universal_model/brain_structure/'


    print('Start processing training set')
    name_list = []
    train_name_list = []
    val_name_list = []
    test_name_list = []

    dataset_name_list = ['dlbs', 'sald', 'ixi']
    
    for phase in ['train', 'valid', 'test']:
        for dataset_name in dataset_name_list:
            for file_name in os.listdir(f"{src_path}{phase}/image"):
                if dataset_name in file_name:
                    name_id = file_name.split('.')[0].replace('_img', '')

                    name_list.append(name_id)
                    if phase == 'train':
                        train_name_list.append(name_id)
                    elif phase == 'valid':
                        val_name_list.append(name_id)
                    elif phase == 'test':
                        test_name_list.append(name_id)


    if not os.path.exists(tgt_path+'list'):
        os.mkdir('%slist'%(tgt_path))
    with open("%slist/dataset.yaml"%tgt_path, "w",encoding="utf-8") as f:
        yaml.dump(name_list, f)
    with open("%slist/dataset_train.yaml"%tgt_path, "w",encoding="utf-8") as f:
        yaml.dump(train_name_list, f)
    with open("%slist/dataset_val.yaml"%tgt_path, "w",encoding="utf-8") as f:
        yaml.dump(val_name_list, f)
    with open("%slist/dataset_test.yaml"%tgt_path, "w",encoding="utf-8") as f:
        yaml.dump(test_name_list, f)
 
    
    os.chdir(src_path)
    
    for (phase, name_list) in zip(['train', 'valid', 'test'], [train_name_list, val_name_list, test_name_list]):
        
        
        for name in name_list:
            img = sitk.ReadImage(src_path+f"{phase}/image/{name}_img.nii")
            lab_gm = sitk.ReadImage(src_path+f"{phase}/mask/{name}_probmask_graymatter.nii")
            lab_wm = sitk.ReadImage(src_path+f"{phase}/mask/{name}_probmask_whitematter.nii")
            lab_csf = sitk.ReadImage(src_path+f"{phase}/mask/{name}_probmask_csf.nii")


            ResampleImage(img, lab_gm, lab_wm, lab_csf, tgt_path, name, (1.5, 1.5, 1.5))
            print(name, 'done')
    
