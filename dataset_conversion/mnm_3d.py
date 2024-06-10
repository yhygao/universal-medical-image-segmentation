import numpy as np
import SimpleITK as sitk
from utils import ResampleXYZAxis, ResampleLabelToRef, CropForeground
import os
import random
import yaml

def extract_frame(path, name):
    img_name = f"{path}{name}_sa.nii.gz"
    lab_name = f"{path}{name}_sa_gt.nii.gz"

    img = sitk.ReadImage(img_name)
    lab = sitk.ReadImage(lab_name)

    npimg = sitk.GetArrayFromImage(img)
    nplab = sitk.GetArrayFromImage(lab)
    
    img_frame_list = []
    lab_frame_list = []

    for i in range(nplab.shape[0]):
        tmp_lab = nplab[i]
        if tmp_lab.max() > 0:
            tmp_img = npimg[i]
        else:
            continue

        itkImg = sitk.GetImageFromArray(tmp_img)
        itkLab = sitk.GetImageFromArray(tmp_lab)

        spacing = img.GetSpacing()[:3]
        origin = img.GetOrigin()[:3]
        #direction = np.array(img.GetDirection()).reshape(4,4)
        #direction = direction[:3, :3]
        #direction = tuple(direction.reshape(-1))

        itkImg.SetSpacing(spacing)
        itkLab.SetSpacing(spacing)
        itkImg.SetOrigin(origin)
        itkLab.SetOrigin(origin)
        #itkImg.SetDirection(direction)
        #itkLab.SetDirection(direction)
        img_frame_list.append(itkImg)
        lab_frame_list.append(itkLab)
        
    assert len(img_frame_list) == 2
    assert len(lab_frame_list) == 2

    return img_frame_list[0], lab_frame_list[0], img_frame_list[1], lab_frame_list[1]


def ResampleCMRImage(imImage, imLabel, save_path, patient_name, count, target_spacing=(1., 1., 1.)):

    assert imImage.GetSpacing() == imLabel.GetSpacing()
    assert imImage.GetSize() == imLabel.GetSize()


    spacing = imImage.GetSpacing()
    origin = imImage.GetOrigin()


    npimg = sitk.GetArrayFromImage(imImage)
    nplab = sitk.GetArrayFromImage(imLabel)
    z, y, x = npimg.shape

    if not os.path.exists('%s'%(save_path)):
        os.mkdir('%s'%(save_path))
    
    re_img_xy = ResampleXYZAxis(imImage, space=(target_spacing[0], target_spacing[1], spacing[2]), interp=sitk.sitkBSpline)
    re_lab_xy = ResampleLabelToRef(imLabel, re_img_xy, interp=sitk.sitkNearestNeighbor)

    re_img_xyz = ResampleXYZAxis(re_img_xy, space=(target_spacing[0], target_spacing[1], target_spacing[2]), interp=sitk.sitkNearestNeighbor)
    re_lab_xyz = ResampleLabelToRef(re_lab_xy, re_img_xyz, interp=sitk.sitkNearestNeighbor)


    cropped_img, cropped_lab = CropForeground(re_img_xyz, re_lab_xyz, context_size=[20, 40, 40])

    sitk.WriteImage(cropped_img, '%s/%s_%d.nii.gz'%(save_path, patient_name, count))
    sitk.WriteImage(cropped_lab, '%s/%s_%d_gt.nii.gz'%(save_path, patient_name, count))



if __name__ == '__main__':


    src_path = '/research/cbim/medical/medical-share/public/MM_challenge/'
    tgt_path = '/research/cbim/medical/yg397/universal_model/mnm/'


    phase_list = ['Training/Labeled', 'Validation', 'Testing']

    training_name_list = os.listdir(f"{src_path}{phase_list[0]}/")
    validation_name_list = os.listdir(f"{src_path}{phase_list[1]}/")
    testing_name_list = os.listdir(f"{src_path}{phase_list[2]}/")

    all_name_list = training_name_list + validation_name_list + testing_name_list


    if not os.path.exists(tgt_path+'list'):
        os.mkdir('%slist'%(tgt_path))
    with open("%slist/dataset.yaml"%tgt_path, "w",encoding="utf-8") as f:
        yaml.dump(all_name_list, f)
    with open("%slist/dataset_train.yaml"%tgt_path, "w",encoding="utf-8") as f:
        yaml.dump(training_name_list, f)
    with open("%slist/dataset_test.yaml"%tgt_path, "w",encoding="utf-8") as f:
        yaml.dump(validation_name_list + testing_name_list, f)

    os.chdir(src_path)
    
    for phase in phase_list:
        if phase == 'Training/Labeled':
            name_list = training_name_list
        elif phase == 'Validation':
            name_list = validation_name_list
        elif phase == 'Testing':
            name_list = testing_name_list
        
        for name in name_list:
            itk_img0, itk_gt0, itk_img1, itk_gt1 = extract_frame(f"{phase}/{name}/", name)

            ResampleCMRImage(itk_img0, itk_gt0, tgt_path, name, 0, (1.5, 1.5, 1.5))
            ResampleCMRImage(itk_img1, itk_gt1, tgt_path, name, 1, (1.5, 1.5, 1.5))           
            
            print(f"{phase}/{name}", 'done')



