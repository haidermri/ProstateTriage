
import yaml
import pickle
import random
import numpy as np
import os
import glob
import SimpleITK as sitk
from tqdm import tqdm
import pandas as pd
import shutil
from nibabel.processing import resample_from_to
import nibabel as nib
import subprocess
import matplotlib.pyplot as plt

class Preprocess():
    def __init__(self,data_dir,coreg_dir,spatial_size,pixdim,filetype,pre_coreg_dir,**kwargs):
        self.data_dir=data_dir
        self.coreg_dir=coreg_dir
        self.spatial_size=spatial_size
        self.pixdim=pixdim
        self.filetype=filetype
        self.pre_coreg_dir = pre_coreg_dir

    def resample_volume(self, volume, mask=False, new_spacing = [0.5, 0.5, 3.0]):
        if mask:
            interpolator = sitk.sitkNearestNeighbor
        else:
            interpolator = sitk.sitkLinear
        original_spacing = volume.GetSpacing()
        original_size = volume.GetSize()
        new_size = [int(round(osz*ospc/nspc)) for osz,ospc,nspc in zip(original_size, original_spacing, new_spacing)]
        return sitk.Resample(volume, new_size, sitk.Transform(), interpolator,
                            volume.GetOrigin(), new_spacing, volume.GetDirection(), 0,
                            volume.GetPixelID())

    def get_data_with_save(self):
        priority_list=['he','ai']

        folders = glob.glob(os.sep.join([self.data_dir,'*']))

        for folder in tqdm(folders):

            axt2=glob.glob(os.sep.join([self.data_dir,folder])+os.sep+'axt2'+'*'+self.filetype)
            b1600=glob.glob(os.sep.join([self.data_dir,folder])+os.sep+'b1600'+'*'+self.filetype)
            adc=glob.glob(os.sep.join([self.data_dir,folder])+os.sep+'adc'+'*'+self.filetype)
            wglbl=glob.glob(os.sep.join([self.data_dir,folder])+os.sep+'p_'+'*'+self.filetype)
            pzlbl=glob.glob(os.sep.join([self.data_dir,folder])+os.sep+'pz_'+'*'+self.filetype)
            tzlbl=glob.glob(os.sep.join([self.data_dir,folder])+os.sep+'tz_'+'*'+self.filetype)
            t1lbl=glob.glob(os.sep.join([self.data_dir,folder])+os.sep+'t1_'+'*'+self.filetype)

            wg_t2 = None
            pz_t2 = None
            tz_t2 = None
            t1_t2 = None
            t1_b0 = None
            wg_t2_p = float('inf')
            pz_t2_p = float('inf')
            tz_t2_p = float('inf')
            t1_b0_p = float('inf')
            t1_t2_p = float('inf')

            if len(wglbl) >= 1:
                for label in wglbl:
                    name = label.split(os.sep)[-1].removesuffix(self.filetype).split('_')[2]
                    srctype = label.split(os.sep)[-1].removesuffix(self.filetype).split('_')[1]
                    if srctype == 'axt2':
                        try:
                            idx = priority_list.index(name)
                        except:
                            raise ValueError('WG name '+name+' in file '+label)
                        if idx < wg_t2_p:
                            wg_t2=label
                            wg_t2_p=idx
                    else: raise TypeError('WG type '+srctype+' in file '+label)
            if len(pzlbl) >= 1:
                for label in pzlbl:
                    name = label.split(os.sep)[-1].removesuffix(self.filetype).split('_')[2]
                    srctype = label.split(os.sep)[-1].removesuffix(self.filetype).split('_')[1]
                    if srctype == 'axt2':
                        try:
                            idx = priority_list.index(name)
                        except:
                            raise ValueError('PZ name '+name+' in file '+label)
                        if idx < pz_t2_p:
                            pz_t2=label
                            pz_t2_p=idx
                    else: raise TypeError('PZ type '+srctype+' in file '+label)
            if len(tzlbl) >= 1:
                for label in tzlbl:
                    name = label.split(os.sep)[-1].removesuffix(self.filetype).split('_')[2]
                    srctype = label.split(os.sep)[-1].removesuffix(self.filetype).split('_')[1]
                    if srctype == 'axt2':
                        try:
                            idx = priority_list.index(name)
                        except:
                            raise ValueError('TZ name '+name+' in file '+label)
                        if idx < tz_t2_p:
                            tz_t2=label
                            tz_t2_p=idx
                    else: raise TypeError('TZ type '+srctype+' in file '+label)
            if len(t1lbl) >= 1:
                for label in t1lbl:
                    try:
                        name = label.split(os.sep)[-1].removesuffix(self.filetype).split('_')[2]
                    except:
                        raise ValueError('T1 ? '+name+' in file '+label)
                    srctype = label.split(os.sep)[-1].removesuffix(self.filetype).split('_')[1]
                    if srctype == 'axt2':
                        try:
                            idx = priority_list.index(name)
                        except:
                            raise ValueError('T1 name '+name+' in file '+label)
                        if idx < t1_t2_p:
                            t1_t2=label
                            t1_t2_p=idx
                    elif srctype == 'b1600' or srctype == 'adc':
                        try:
                            idx = priority_list.index(name)
                        except:
                            raise ValueError('T1 name '+name+' in file '+label)
                        if idx < t1_b0_p:
                            t1_b0=label
                            t1_b0_p=idx
                    else: raise TypeError('T1 type '+srctype+' in file '+label)

            if not axt2 or not b1600 or not adc or ((not t1_t2 and not t1_b0) or not wg_t2 or not pz_t2 or not tz_t2):
                print('Missing files for '+folder)
                continue

            if not os.path.exists(os.sep.join([self.pre_coreg_dir,folder])):
                os.makedirs(os.sep.join([self.pre_coreg_dir,folder]))


            im_axt2 = nib.load(axt2[0])
            
            im_b1600 = nib.load(b1600[0])
            im_adc = nib.load(adc[0])
            if t1_b0 and not t1_t2:
                im_t1_t2 = nib.load(t1_b0)
                im_t1_t2 = resample_from_to(im_t1_t2, (im_axt2.shape[:3], im_axt2.affine))
            im_b1600 = resample_from_to(im_b1600, (im_axt2.shape[:3], im_axt2.affine))
            im_adc = resample_from_to(im_adc, (im_axt2.shape[:3], im_axt2.affine))
            if (not np.array_equal(im_b1600.affine, im_axt2.affine)) or (not np.array_equal(im_adc.affine, im_axt2.affine) or (t1_b0 and not t1_t2 and not np.array_equal(im_t1_t2.affine, im_axt2.affine))):
                raise ValueError('Affines not equal')
            
            pre_axt2_path = os.sep.join([self.pre_coreg_dir,folder,'axt2.nii.gz'])
            pre_b1600_path = os.sep.join([self.pre_coreg_dir,folder,'b1600.nii.gz'])
            pre_adc_path = os.sep.join([self.pre_coreg_dir,folder,'adc.nii.gz'])

            if t1_b0 and not t1_t2:
                t1_t2_pre = t1_b0.split('_')[-1]
                t1_t2 = os.sep.join([self.pre_coreg_dir,folder,'t1_axt2_'+t1_t2_pre])
                nib.save(im_t1_t2, t1_t2)
            nib.save(im_axt2, pre_axt2_path)
            nib.save(im_b1600, pre_b1600_path)
            nib.save(im_adc, pre_adc_path)

            if t1_t2:
                t1_t2_filename = t1_t2.split(os.sep)[-1]
                seg_t1_axt2 = sitk.ReadImage(t1_t2)
                seg_t1_axt2 = self.resample_volume(seg_t1_axt2, mask=True, new_spacing = [0.5, 0.5, 3.0])
                sitk.WriteImage(seg_t1_axt2, os.sep.join([self.pre_coreg_dir,folder,t1_t2_filename]))

            im_axt2 = sitk.ReadImage(pre_axt2_path)
            im_axt2 = self.resample_volume(im_axt2, mask=False, new_spacing = [0.5, 0.5, 3.0])
            im_b1600 = sitk.ReadImage(pre_b1600_path)
            im_b1600 = self.resample_volume(im_b1600, mask=False, new_spacing = [0.5, 0.5, 3.0])
            im_adc = sitk.ReadImage(pre_adc_path)
            im_adc = self.resample_volume(im_adc, mask=False, new_spacing = [0.5, 0.5, 3.0])
            sitk.WriteImage(im_axt2, pre_axt2_path)
            sitk.WriteImage(im_b1600, pre_b1600_path)
            sitk.WriteImage(im_adc, pre_adc_path)

            if wg_t2:
                wg_t2_filename = wg_t2.split(os.sep)[-1]
                seg_wg_axt2 = sitk.ReadImage(wg_t2)
                seg_wg_axt2 = self.resample_volume(seg_wg_axt2, mask=True, new_spacing = [0.5, 0.5, 3.0])
                sitk.WriteImage(seg_wg_axt2, os.sep.join([self.pre_coreg_dir,folder,wg_t2_filename]))

            if pz_t2:
                pz_t2_filename = pz_t2.split(os.sep)[-1]
                seg_pz_axt2 = sitk.ReadImage(pz_t2)
                seg_pz_axt2 = self.resample_volume(seg_pz_axt2, mask=True, new_spacing = [0.5, 0.5, 3.0])
                sitk.WriteImage(seg_pz_axt2, os.sep.join([self.pre_coreg_dir,folder,pz_t2_filename]))

            if tz_t2:
                tz_t2_filename = tz_t2.split(os.sep)[-1]
                seg_tz_axt2 = sitk.ReadImage(tz_t2)
                seg_tz_axt2 = self.resample_volume(seg_tz_axt2, mask=True, new_spacing = [0.5, 0.5, 3.0])
                sitk.WriteImage(seg_tz_axt2, os.sep.join([self.pre_coreg_dir,folder,tz_t2_filename]))

        return

    def process_final(self,imsize,pixdim):
        xsize, ysize, zsize = imsize
        folders = glob.glob(os.sep.join([self.pre_coreg_dir,'*']))
        for folder in tqdm(folders):
            axt2 = os.sep.join([self.pre_coreg_dir,folder,'axt2.nii.gz'])
            b1600 = os.sep.join([self.pre_coreg_dir,folder,'b1600.nii.gz'])
            adc = os.sep.join([self.pre_coreg_dir,folder,'adc.nii.gz'])
            wglbl=glob.glob(os.sep.join([self.pre_coreg_dir,folder])+os.sep+'p_axt2_'+'*'+self.filetype)
            pzlbl = os.sep.join([self.pre_coreg_dir,folder,'pz_axt2_ai.nii.gz'])
            if not os.path.exists(pzlbl):
                print('No PZ for',folder)
                continue
            tzlbl = os.sep.join([self.pre_coreg_dir,folder,'tz_axt2_ai.nii.gz'])
            if not os.path.exists(tzlbl):
                print('No TZ for',folder)
                continue
            t1lbl=glob.glob(os.sep.join([self.pre_coreg_dir,folder])+os.sep+'t1_axt2_'+'*'+self.filetype)
            wg_filename = wglbl[0].split(os.sep)[-1]
            if t1lbl:
                t1_filename = t1lbl[0].split(os.sep)[-1]
            else:
                t1_filename = 't1_axt2_XX.nii.gz'
            if not axt2 or not b1600 or not adc or not wglbl:
                raise ValueError('Missing files for '+folder)
                
            im_axt2 = nib.load(axt2)
            im_b1600 = nib.load(b1600)
            im_adc = nib.load(adc)
            im_wg_t2 = nib.load(wglbl[0])
            im_pz = nib.load(pzlbl)
            im_tz = nib.load(tzlbl)
            if t1lbl:
                im_t1_t2 = nib.load(t1lbl[0])
                t1_t2 = True
            else:
                t1_t2 = False
            
            imarray_wg_t2 = im_wg_t2.get_fdata()
            if not np.equal(im_axt2.header.get_zooms(),im_wg_t2.header.get_zooms()).all():
                raise ValueError('Voxel sizes not equal',folder,im_axt2.header.get_zooms(),im_wg_t2.header.get_zooms())
            if not np.equal(im_axt2.header.get_zooms(),pixdim).all():
                raise ValueError('Voxel sizes not per imsize',folder,im_axt2.header.get_zooms())
            if not np.equal(im_axt2.shape,im_wg_t2.shape).all():
                raise ValueError('Shape not equal',folder,im_axt2.shape,im_wg_t2.shape)
            indices = np.argwhere(imarray_wg_t2!=0)
            xmin = float('inf')
            xmax = float('-inf')
            ymin = float('inf')
            ymax = float('-inf')
            zmin = float('inf')
            zmax = float('-inf')
            
            for idx in indices:
                x = idx[0]
                y = idx[1]
                z = idx[2]
                if x < xmin:
                    xmin = x
                if x > xmax:
                    xmax = x
                if y < ymin:
                    ymin = y
                if y > ymax:
                    ymax = y
                if z < zmin:
                    zmin = z
                if z > zmax:
                    zmax = z


            xcenter = (xmin+xmax)/2
            ycenter = (ymin+ymax)/2
            zcenter = (zmin+zmax)/2


            
            xstart = round(xcenter - xsize//2)
            xend = round(xcenter + xsize//2)
        
            ystart = round(ycenter - ysize//2)
            yend = round(ycenter + ysize//2)
        
            zstart = round(zcenter - zsize//2)
            zend = round(zcenter + zsize//2)

            xprepad=0
            xpostpad=0
            yprepad=0
            ypostpad=0
            zprepad=0
            zpostpad=0
            if xstart < 0:
                xprepad = -xstart
                xstart += xprepad
                xend += xprepad
            if xend > imarray_wg_t2.shape[0]+xprepad:
                xpostpad = xend - (imarray_wg_t2.shape[0]+xprepad)
                xend = imarray_wg_t2.shape[0]+xprepad+xpostpad
            if ystart < 0:
                yprepad = -ystart
                ystart += yprepad
                yend += yprepad
            if yend > imarray_wg_t2.shape[1]+yprepad:
                ypostpad = yend - (imarray_wg_t2.shape[1]+yprepad)
                yend = imarray_wg_t2.shape[1]+yprepad+ypostpad
            if zstart < 0:
                zprepad = -zstart
                zstart += zprepad
                zend += zprepad
            if zend > imarray_wg_t2.shape[2]+zprepad:
                zpostpad = zend - (imarray_wg_t2.shape[2]+zprepad)
                zend = imarray_wg_t2.shape[2]+zprepad+zpostpad


            imarray_axt2 = im_axt2.get_fdata()
            if xprepad or xpostpad or yprepad or ypostpad or zprepad or zpostpad:
                imarray_axt2 = np.pad(imarray_axt2, ((xprepad,xpostpad),(yprepad,ypostpad),(zprepad,zpostpad)))
            imarray_axt2 = imarray_axt2[xstart:xend,ystart:yend,zstart:zend]
            if imarray_axt2.shape != imsize:
                print('AXT2 shape not correct: '+str(imarray_axt2.shape) + ' folder: '+folder)
                continue
            imarray_b1600 = im_b1600.get_fdata()
            if xprepad or xpostpad or yprepad or ypostpad or zprepad or zpostpad:
                imarray_b1600 = np.pad(imarray_b1600, ((xprepad,xpostpad),(yprepad,ypostpad),(zprepad,zpostpad)))
            imarray_b1600 = imarray_b1600[xstart:xend,ystart:yend,zstart:zend]
            if imarray_b1600.shape != imsize:
                print('B1600 shape not correct: '+str(imarray_b1600.shape) + ' folder: '+folder)
                continue
            imarray_adc = im_adc.get_fdata()
            if xprepad or xpostpad or yprepad or ypostpad or zprepad or zpostpad:
                imarray_adc = np.pad(imarray_adc, ((xprepad,xpostpad),(yprepad,ypostpad),(zprepad,zpostpad)))
            imarray_adc = imarray_adc[xstart:xend,ystart:yend,zstart:zend]
            if imarray_adc.shape != imsize:
                print('ADC shape not correct: '+str(imarray_adc.shape) + ' folder: '+folder)
                continue
            if xprepad or xpostpad or yprepad or ypostpad or zprepad or zpostpad:
                imarray_wg_t2 = np.pad(imarray_wg_t2, ((xprepad,xpostpad),(yprepad,ypostpad),(zprepad,zpostpad)))
            test_wg = np.copy(imarray_wg_t2)
            imarray_wg_t2 = imarray_wg_t2[xstart:xend,ystart:yend,zstart:zend]
            if imarray_wg_t2.shape != imsize:
                print('WG shape not correct: '+str(imarray_wg_t2.shape) + ' folder: '+folder)
                continue
            test_wg[xstart:xend,ystart:yend,zstart:zend] = 0
            if np.any(test_wg):
                print('WG: '+folder)
                for z in range(test_wg.shape[2]):
                    ax = plt.subplot(test_wg.shape[2]//5+1,5,z+1)
                    ax.imshow(test_wg[:,:,z])
                plt.show()
                print(xprepad,xpostpad,yprepad,ypostpad,zprepad,zpostpad)
                raise ValueError('Part of WG cropped out: '+folder)
            
            imarray_pz = im_pz.get_fdata()
            if xprepad or xpostpad or yprepad or ypostpad or zprepad or zpostpad:
                imarray_pz = np.pad(imarray_pz, ((xprepad,xpostpad),(yprepad,ypostpad),(zprepad,zpostpad)))
            imarray_pz = imarray_pz[xstart:xend,ystart:yend,zstart:zend]
            if imarray_pz.shape != imsize:
                print('PZ shape not correct: '+str(imarray_pz.shape) + ' folder: '+folder)
                continue
            imarray_tz = im_tz.get_fdata()
            if xprepad or xpostpad or yprepad or ypostpad or zprepad or zpostpad:
                imarray_tz = np.pad(imarray_tz, ((xprepad,xpostpad),(yprepad,ypostpad),(zprepad,zpostpad)))
            imarray_tz = imarray_tz[xstart:xend,ystart:yend,zstart:zend]
            if imarray_tz.shape != imsize:
                print('TZ shape not correct: '+str(imarray_tz.shape) + ' folder: '+folder)
                continue

            if t1_t2:
                imarray_t1_t2 = im_t1_t2.get_fdata()
                imarray_t1_t2 = np.where(imarray_t1_t2!=0,1,0)
                if xprepad or xpostpad or yprepad or ypostpad or zprepad or zpostpad:
                    imarray_t1_t2 = np.pad(imarray_t1_t2, ((xprepad,xpostpad),(yprepad,ypostpad),(zprepad,zpostpad)))
                imarray_t1_t2 = imarray_t1_t2[xstart:xend,ystart:yend,zstart:zend]
                if imarray_t1_t2.shape != imsize:
                    print('T1 shape not correct: '+str(imarray_t1_t2.shape) + ' folder: '+folder)
                    continue
            else:
                imarray_t1_t2 = np.zeros(imsize)
            
            im_axt2 = nib.Nifti1Image(imarray_axt2, im_axt2.affine)
            im_b1600 = nib.Nifti1Image(imarray_b1600, im_b1600.affine)
            im_adc = nib.Nifti1Image(imarray_adc, im_adc.affine)
            im_wg_t2 = nib.Nifti1Image(imarray_wg_t2, im_wg_t2.affine)
            im_pz = nib.Nifti1Image(imarray_pz, im_pz.affine)
            im_tz = nib.Nifti1Image(imarray_tz, im_tz.affine)
            if t1_t2:
                im_t1_t2 = nib.Nifti1Image(imarray_t1_t2, im_t1_t2.affine)
            else:
                im_t1_t2 = nib.Nifti1Image(imarray_t1_t2, im_axt2.affine)
            axt2_path = os.sep.join([self.coreg_dir,folder,'axt2.nii.gz'])
            b1600_path = os.sep.join([self.coreg_dir,folder,'b1600.nii.gz'])
            adc_path = os.sep.join([self.coreg_dir,folder,'adc.nii.gz'])
            wg_t2_path = os.sep.join([self.coreg_dir,folder,wg_filename])
            pz_path = os.sep.join([self.coreg_dir,folder,'pz_axt2_ai.nii.gz'])
            tz_path = os.sep.join([self.coreg_dir,folder,'tz_axt2_ai.nii.gz'])
            t1_t2_path = os.sep.join([self.coreg_dir,folder,t1_filename])
            if not os.path.exists(os.sep.join([self.coreg_dir,folder])):
                os.makedirs(os.sep.join([self.coreg_dir,folder]))
            nib.save(im_axt2, axt2_path)
            nib.save(im_b1600, b1600_path)
            nib.save(im_adc, adc_path)
            nib.save(im_wg_t2, wg_t2_path)
            nib.save(im_pz, pz_path)
            nib.save(im_tz, tz_path)
            nib.save(im_t1_t2, t1_t2_path)





def main(running_yaml):
    with open(running_yaml, 'r') as stream:
        paramdic = yaml.full_load(stream)
    
    imsize = paramdic['spatial_size']
    pixdim = paramdic['pixdim']
    processor = Preprocess(**paramdic)

    processor.get_data_with_save()

    processor.process_final(imsize,pixdim)
    print('Completed')


if __name__ == '__main__':
    running_yaml = 'preprocess_paramdic.yaml'
    main(running_yaml)
    print('Completed')