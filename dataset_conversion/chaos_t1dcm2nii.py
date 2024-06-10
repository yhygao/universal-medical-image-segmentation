import os
import SimpleITK as sitk
import numpy as np
from PIL import Image

import pdb


reader = sitk.ImageSeriesReader()

os.chdir('./MR')
for name in os.listdir('.'):
    for phase in ['InPhase', 'OutPhase']:
        dcm_series = reader.GetGDCMSeriesFileNames(f"./{name}/T1DUAL/DICOM_anon/{phase}/")
        reader.SetFileNames(dcm_series)
        img = reader.Execute()

        sitk.WriteImage(img, f"/research/cbim/medical/yg397/CHAOS/Train_Sets/T1/{name}_{phase}.nii.gz")
    
    lab_list = []
    for lab_name in os.listdir(f"./{name}/T1DUAL/Ground/"):
        lab = Image.open(f"./{name}/T1DUAL/Ground/{lab_name}")
        lab_list.append(np.array(lab))
    
    npimg = sitk.GetArrayFromImage(img)
    lab_arr = np.stack(lab_list)
    itk_lab = sitk.GetImageFromArray(lab_arr)
    itk_lab.CopyInformation(img)
    sitk.WriteImage(itk_lab, f"/research/cbim/medical/yg397/CHAOS/Train_Sets/T1/{name}_gt.nii.gz")
    
    print(name, 'done')

