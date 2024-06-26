import numpy as np
import SimpleITK as sitk
from skimage.measure import regionprops

import os
import pdb

def ResampleXYZAxis(imImage, space=(1., 1., 1.), interp=sitk.sitkLinear):
    identity1 = sitk.Transform(3, sitk.sitkIdentity)
    sp1 = imImage.GetSpacing()
    sz1 = imImage.GetSize()

    sz2 = (int(round(sz1[0]*sp1[0]*1.0/space[0])), int(round(sz1[1]*sp1[1]*1.0/space[1])), int(round(sz1[2]*sp1[2]*1.0/space[2])))

    imRefImage = sitk.Image(sz2, imImage.GetPixelIDValue())
    imRefImage.SetSpacing(space)
    imRefImage.SetOrigin(imImage.GetOrigin())
    imRefImage.SetDirection(imImage.GetDirection())
    #imRefImage.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    
    imOutImage = sitk.Resample(imImage, imRefImage, identity1, interp)

    return imOutImage

def ResampleLabelToRef(imLabel, imRef, interp=sitk.sitkNearestNeighbor):
    identity1 = sitk.Transform(3, sitk.sitkIdentity)

    imRefImage = sitk.Image(imRef.GetSize(), imLabel.GetPixelIDValue())
    imRefImage.SetSpacing(imRef.GetSpacing())
    imRefImage.SetOrigin(imRef.GetOrigin())
    imRefImage.SetDirection(imRef.GetDirection())
        
    ResampledLabel = sitk.Resample(imLabel, imRefImage, identity1, interp)
    
    return ResampledLabel

def ITKReDirection_bkp(itkimg, target_direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), interp=sitk.sitkBSpline):
    # target direction should be orthognal, i.e. (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    target_direction = np.array(target_direction).reshape(3, 3)
    original_direction = np.array(itkimg.GetDirection()).reshape(3, 3)
    
    
    if not np.array_equal(target_direction, original_direction):
        rotation_matrix = np.linalg.inv(original_direction) @ target_direction

        # Create a resampler
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(itkimg)
        resampler.SetSize(itkimg.GetSize())

        # Set the transformation (use a combination of rotation and translation)
        transformation = sitk.AffineTransform(3)
        transformation.SetMatrix(rotation_matrix.ravel())
        resampler.SetTransform(transformation)
        resampler.SetInterpolator(interp)

        # Resample the image
        transformed_img = resampler.Execute(itkimg)
        transformed_img.SetDirection(target_direction.ravel())
        
        return transformed_img
    else:
        return itkimg

def ITKReDirection(itkimg, target_direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), interp=sitk.sitkBSpline):
    # target direction should be orthognal, i.e. (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    target_direction = np.array(target_direction).reshape(3, 3)
    original_direction = np.array(itkimg.GetDirection()).reshape(3, 3)
    
    
    if not np.array_equal(target_direction, original_direction):
        rotation_matrix = np.linalg.inv(original_direction) @ target_direction

        # Create a resampler
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(itkimg.GetSize())
        resampler.SetOutputDirection(target_direction.ravel())
        resampler.SetOutputSpacing(itkimg.GetSpacing())
        
        # Compute the new origin such that the physical center of the image remains the same
        center_index = [(dim-1)*0.5 for dim in itkimg.GetSize()]
        center_physical = itkimg.TransformContinuousIndexToPhysicalPoint(center_index)
        
        half_physical_size = [0.5*spacing*(size-1) for spacing, size in zip(itkimg.GetSpacing(), itkimg.GetSize())]
        new_origin = center_physical - np.dot(half_physical_size, target_direction) 
        resampler.SetOutputOrigin(tuple(new_origin))


        # Set the transformation (use a combination of rotation and translation)
        transformation = sitk.AffineTransform(3)
        transformation.SetMatrix(rotation_matrix.ravel())
        transformation.SetCenter(center_physical)

        resampler.SetTransform(transformation)
        resampler.SetInterpolator(interp)

        # Resample the image
        transformed_img = resampler.Execute(itkimg)
        
        return transformed_img
    else:
        return itkimg



def ITKReDirection_bkp(itkimg, target_direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)):
    # target direction should be orthognal, i.e. (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    # permute axis
    print(itkimg.GetDirection())
    tmp_target_direction = np.abs(np.round(np.array(target_direction))).reshape(3,3).T
    current_direction = np.abs(np.round(itkimg.GetDirection())).reshape(3,3).T
    
    permute_order = []
    if not np.array_equal(tmp_target_direction, current_direction):
        for i in range(3):
            for j in range(3):
                if np.array_equal(tmp_target_direction[i], current_direction[j]):
                    permute_order.append(j)
                    #print(i, j)
                    #print(permute_order)
                    break
        pdb.set_trace()
        redirect_img = sitk.PermuteAxes(itkimg, permute_order)
    else:
        redirect_img = itkimg
    # flip axis
    current_direction = np.round(np.array(redirect_img.GetDirection())).reshape(3,3).T
    current_direction = np.max(current_direction, axis=1)

    tmp_target_direction = np.array(target_direction).reshape(3,3).T 
    tmp_target_direction = np.max(tmp_target_direction, axis=1)
    flip_order = ((tmp_target_direction * current_direction) != 1)
    fliped_img = sitk.Flip(redirect_img, [bool(flip_order[0]), bool(flip_order[1]), bool(flip_order[2])])
    return fliped_img

def CropForeground(imImage, imLabel, context_size=[10, 30, 30]):
    # the context_size is in numpy indice order: z, y, x
    # Note that SimpleITK use the indice order of: x, y, z
    
    npImg = sitk.GetArrayFromImage(imImage)
    npLab = sitk.GetArrayFromImage(imLabel)

    mask = (npLab>0).astype(np.uint8) # foreground mask
    
    regions = regionprops(mask)
    assert len(regions) == 1

    zz, yy, xx = npImg.shape

    z, y, x = regions[0].centroid

    z_min, y_min, x_min, z_max, y_max, x_max = regions[0].bbox
    print(zz, yy, xx)
    print('forground size:', z_max-z_min, y_max-y_min, x_max-x_min)

    z, y, x = int(z), int(y), int(x)

    z_min = max(0, z_min-context_size[0])
    z_max = min(zz, z_max+context_size[0])
    y_min = max(0, y_min-context_size[1])
    y_max = min(yy, y_max+context_size[1])
    x_min = max(0, x_min-context_size[2])
    x_max = min(xx, x_max+context_size[2])

    img = npImg[z_min:z_max, y_min:y_max, x_min:x_max]
    lab = npLab[z_min:z_max, y_min:y_max, x_min:x_max]

    croppedImage = sitk.GetImageFromArray(img)
    croppedLabel = sitk.GetImageFromArray(lab)


    croppedImage.SetSpacing(imImage.GetSpacing())
    croppedLabel.SetSpacing(imImage.GetSpacing())
    
    croppedImage.SetDirection(imImage.GetDirection())
    croppedLabel.SetDirection(imImage.GetDirection())

    return croppedImage, croppedLabel


def CropForeground_TwoLab(imImage, imLabel, imGTVLabel, context_size=[10, 30, 30]):
    # the context_size is in numpy indice order: z, y, x
    # Note that SimpleITK use the indice order of: x, y, z
    
    npImg = sitk.GetArrayFromImage(imImage)
    npLab = sitk.GetArrayFromImage(imLabel)
    npGTVLab = sitk.GetArrayFromImage(imGTVLabel)

    mask = (npLab>0).astype(np.uint8) # foreground mask
    
    regions = regionprops(mask)
    assert len(regions) == 1

    zz, yy, xx = npImg.shape

    z, y, x = regions[0].centroid

    z_min, y_min, x_min, z_max, y_max, x_max = regions[0].bbox
    print('forground size:', z_max-z_min, y_max-y_min, x_max-x_min)

    z, y, x = int(z), int(y), int(x)

    z_min = max(0, z_min-context_size[0])
    z_max = min(zz, z_max+context_size[0])
    y_min = max(0, y_min-context_size[1])
    y_max = min(yy, y_max+context_size[1])
    x_min = max(0, x_min-context_size[2])
    x_max = min(xx, x_max+context_size[2])

    img = npImg[z_min:z_max, y_min:y_max, x_min:x_max]
    lab = npLab[z_min:z_max, y_min:y_max, x_min:x_max]
    gtv_lab = npGTVLab[z_min:z_max, y_min:y_max, x_min:x_max]

    croppedImage = sitk.GetImageFromArray(img)
    croppedLabel = sitk.GetImageFromArray(lab)
    croppedGTVLabel = sitk.GetImageFromArray(gtv_lab)


    croppedImage.SetSpacing(imImage.GetSpacing())
    croppedLabel.SetSpacing(imImage.GetSpacing())
    croppedGTVLabel.SetSpacing(imImage.GetSpacing())
    
    croppedImage.SetDirection(imImage.GetDirection())
    croppedLabel.SetDirection(imImage.GetDirection())
    croppedGTVLabel.SetDirection(imImage.GetDirection())

    return croppedImage, croppedLabel, croppedGTVLabel






