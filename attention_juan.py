### Dependencies
import argparse
import colorsys
from io import BytesIO
import os
import random
import requests
import sys
import io
import zipfile
import threading

import cv2
import h5py
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
from scipy.stats import rankdata
import skimage.io
from skimage.measure import find_contours
from tqdm import tqdm
import webdataset as wds

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import torchvision.transforms as transforms
from einops import rearrange, repeat

sys.path.append('../')
sys.path.append('../Hierarchical-Pretraining/')
import vision_transformer as vits
import vision_transformer4k as vits4k

import openslide
from tqdm import tqdm
from hipt_model_utils import get_vit256, get_vit4k, eval_transforms
import time
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.tools.tissuemask import TissueMasker
from tiatoolbox.tools.tissuemask import OtsuTissueMasker
import concurrent.futures


def get_vit256(pretrained_weights, arch='vit_small', device=torch.device('cuda:0')):
    r"""
    Builds ViT-256 Model.
    
    Args:
    - pretrained_weights (str): Path to ViT-256 Model Checkpoint.
    - arch (str): Which model architecture.
    - device (torch): Torch device to save model.
    
    Returns:
    - model256 (torch.nn): Initialized model.
    """
    
    checkpoint_key = 'teacher'
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model256 = vits.__dict__[arch](patch_size=16, num_classes=0)
    for p in model256.parameters():
        p.requires_grad = False
    model256.eval()
    model256.to(device)

    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model256.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
        
    return model256


def get_vit4k(pretrained_weights, arch='vit4k_xs', device=torch.device('cuda:1')):
    r"""
    Builds ViT-4K Model.
    
    Args:
    - pretrained_weights (str): Path to ViT-4K Model Checkpoint.
    - arch (str): Which model architecture.
    - device (torch): Torch device to save model.
    
    Returns:
    - model256 (torch.nn): Initialized model.
    """
    
    checkpoint_key = 'teacher'
    device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    model4k = vits4k.__dict__[arch](num_classes=0)
    for p in model4k.parameters():
        p.requires_grad = False
    model4k.eval()
    model4k.to(device)

    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model4k.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
        
    return model4k


def cmap_map(function, cmap):
    r""" 
    Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    
    Args:
    - function (function)
    - cmap (matplotlib.colormap)
    
    Returns:
    - matplotlib.colormap
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)


def identity(x):
    r"""
    Identity Function.
    
    Args:
    - x:
    
    Returns:
    - x
    """
    return x

def tensorbatch2im(input_image, imtype=np.uint8):
    r""""
    Converts a Tensor array into a numpy image array.
    
    Args:
        - input_image (torch.Tensor): (B, C, W, H) Torch Tensor.
        - imtype (type): the desired type of the converted numpy array
        
    Returns:
        - image_numpy (np.array): (B, W, H, C) Numpy Array.
    """
    if not isinstance(input_image, np.ndarray):
        image_numpy = input_image.cpu().float().numpy()  # convert it into a numpy array
        #if image_numpy.shape[0] == 1:  # grayscale to RGB
        #    image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def getConcatImage(imgs, how='horizontal', gap=0):
    r"""
    Function to concatenate list of images (vertical or horizontal).

    Args:
        - imgs (list of PIL.Image): List of PIL Images to concatenate.
        - how (str): How the images are concatenated (either 'horizontal' or 'vertical')
        - gap (int): Gap (in px) between images

    Return:
        - dst (PIL.Image): Concatenated image result.
    """
    gap_dist = (len(imgs)-1)*gap
    
    if how == 'vertical':
        w, h = np.max([img.width for img in imgs]), np.sum([img.height for img in imgs])
        h += gap_dist
        curr_h = 0
        dst = Image.new('RGBA', (w, h), color=(255, 255, 255, 0))
        for img in imgs:
            dst.paste(img, (0, curr_h))
            curr_h += img.height + gap

    elif how == 'horizontal':
        w, h = np.sum([img.width for img in imgs]), np.min([img.height for img in imgs])
        w += gap_dist
        curr_w = 0
        dst = Image.new('RGBA', (w, h), color=(255, 255, 255, 0))

        for idx, img in enumerate(imgs):
            dst.paste(img, (curr_w, 0))
            curr_w += img.width + gap

    return dst


def add_margin(pil_img, top, right, bottom, left, color):
    r"""
    Adds custom margin to PIL.Image.
    """
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def concat_scores256(attns, size=(256,256)):
    r"""
    """
    rank = lambda v: rankdata(v)*100/len(v)
    color_block = [rank(attn.flatten()).reshape(size) for attn in attns]
    color_hm = np.concatenate([
        np.concatenate(color_block[i:(i+16)], axis=1)
        for i in range(0,256,16)
    ])
    return color_hm


def concat_scores4k(attn, size=(4096, 4096)):
    r"""
    """
    rank = lambda v: rankdata(v)*100/len(v)
    color_hm = rank(attn.flatten()).reshape(size)
    return color_hm



def get_scores256(attns, size=(256,256)):
    r"""
    """
    rank = lambda v: rankdata(v)*100/len(v)
    color_block = [rank(attn.flatten()).reshape(size) for attn in attns][0]
    return color_block


def get_patch_attention_scores(patch, model256, scale=1, device256=torch.device('cpu')):
    #Modified to use cpu, original code has  device256=torch.device('cuda:0')
    r"""
    Forward pass in ViT-256 model with attention scores saved.
    
    Args:
    - region (PIL.Image):       4096 x 4096 Image 
    - model256 (torch.nn):      256-Level ViT 
    - scale (int):              How much to scale the output image by (e.g. - scale=4 will resize images to be 1024 x 1024.)
    
    Returns:
    - attention_256 (torch.Tensor): [1, 256/scale, 256/scale, 3] torch.Tensor of attention maps for 256-sized patches.
    """
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        )
    ])

    with torch.no_grad():   
        batch_256 = t(patch).unsqueeze(0)
        batch_256 = batch_256.to(device256, non_blocking=True)
        features_256 = model256(batch_256)

        attention_256 = model256.get_last_selfattention(batch_256)
        nh = attention_256.shape[1] # number of head
        attention_256 = attention_256[:, :, 0, 1:].reshape(256, nh, -1)
        attention_256 = attention_256.reshape(1, nh, 16, 16)
        attention_256 = nn.functional.interpolate(attention_256, scale_factor=int(16/scale), mode="nearest").cpu().numpy()

        if scale != 1:
            batch_256 = nn.functional.interpolate(batch_256, scale_factor=(1/scale), mode="nearest")
            
    return tensorbatch2im(batch_256), attention_256


def create_patch_heatmaps_indiv(patch, model256, output_dir, fname, threshold=0.5,
                             offset=16, alpha=0.5, cmap=plt.get_cmap('coolwarm'), device256=torch.device('cuda:0')):
    r"""
    Creates patch heatmaps (saved individually)
    
    Args:
    - patch (PIL.Image):        256 x 256 Image 
    - model256 (torch.nn):      256-Level ViT 
    - output_dir (str):         Save directory / subdirectory
    - fname (str):              Naming structure of files
    - offset (int):             How much to offset (from top-left corner with zero-padding) the region by for blending 
    - alpha (float):            Image blending factor for cv2.addWeighted
    - cmap (matplotlib.pyplot): Colormap for creating heatmaps
    
    Returns:
    - None
    """
    patch1 = patch.copy()
    patch2 = add_margin(patch.crop((16,16,256,256)), top=0, left=0, bottom=16, right=16, color=(255,255,255))
    b256_1, a256_1 = get_patch_attention_scores(patch1, model256, device256=device256)
    b256_1, a256_2 = get_patch_attention_scores(patch2, model256, device256=device256)
    save_region = np.array(patch.copy())
    s = 256
    offset_2 = offset

    if threshold != None:
        for i in range(6):
            score256_1 = get_scores256(a256_1[:,i,:,:], size=(s,)*2)
            score256_2 = get_scores256(a256_2[:,i,:,:], size=(s,)*2)
            new_score256_2 = np.zeros_like(score256_2)
            new_score256_2[offset_2:s, offset_2:s] = score256_2[:(s-offset_2), :(s-offset_2)]
            overlay256 = np.ones_like(score256_2)*100
            overlay256[offset_2:s, offset_2:s] += 100
            score256 = (score256_1+new_score256_2)/overlay256

            mask256 = score256.copy()
            mask256[mask256 < threshold] = 0
            mask256[mask256 > threshold] = 0.95

            color_block256 = (cmap(mask256)*255)[:,:,:3].astype(np.uint8)
            region256_hm = cv2.addWeighted(color_block256, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
            region256_hm[mask256==0] = 0
            img_inverse = save_region.copy()
            img_inverse[mask256 == 0.95] = 0
            Image.fromarray(region256_hm+img_inverse).save(os.path.join(output_dir, '%s_256th[%d].png' % (fname, i)))

    for i in range(6):
        score256_1 = get_scores256(a256_1[:,i,:,:], size=(s,)*2)
        score256_2 = get_scores256(a256_2[:,i,:,:], size=(s,)*2)
        new_score256_2 = np.zeros_like(score256_2)
        new_score256_2[offset_2:s, offset_2:s] = score256_2[:(s-offset_2), :(s-offset_2)]
        overlay256 = np.ones_like(score256_2)*100
        overlay256[offset_2:s, offset_2:s] += 100
        score256 = (score256_1+new_score256_2)/overlay256
        color_block256 = (cmap(score256)*255)[:,:,:3].astype(np.uint8)
        region256_hm = cv2.addWeighted(color_block256, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
        Image.fromarray(region256_hm).save(os.path.join(output_dir, '%s_256[%s].png' % (fname, i)))
        
        
def create_patch_heatmaps_concat(patch, model256, output_dir, fname, threshold=0.5,
                             offset=16, alpha=0.5, cmap=plt.get_cmap('coolwarm')):
    r"""
    Creates patch heatmaps (concatenated for easy comparison)
    
    Args:
    - patch (PIL.Image):        256 x 256 Image 
    - model256 (torch.nn):      256-Level ViT 
    - output_dir (str):         Save directory / subdirectory
    - fname (str):              Naming structure of files
    - offset (int):             How much to offset (from top-left corner with zero-padding) the region by for blending 
    - alpha (float):            Image blending factor for cv2.addWeighted
    - cmap (matplotlib.pyplot): Colormap for creating heatmaps
    
    Returns:
    - None
    """
    patch1 = patch.copy()
    patch2 = add_margin(patch.crop((16,16,256,256)), top=0, left=0, bottom=16, right=16, color=(255,255,255))
    b256_1, a256_1 = get_patch_attention_scores(patch1, model256)
    b256_1, a256_2 = get_patch_attention_scores(patch2, model256)
    save_region = np.array(patch.copy())
    s = 256
    offset_2 = offset

    if threshold != None:
        ths = []
        for i in range(6):
            score256_1 = get_scores256(a256_1[:,i,:,:], size=(s,)*2)
            score256_2 = get_scores256(a256_2[:,i,:,:], size=(s,)*2)
            new_score256_2 = np.zeros_like(score256_2)
            new_score256_2[offset_2:s, offset_2:s] = score256_2[:(s-offset_2), :(s-offset_2)]
            overlay256 = np.ones_like(score256_2)*100
            overlay256[offset_2:s, offset_2:s] += 100
            score256 = (score256_1+new_score256_2)/overlay256

            mask256 = score256.copy()
            mask256[mask256 < threshold] = 0
            mask256[mask256 > threshold] = 0.95

            color_block256 = (cmap(mask256)*255)[:,:,:3].astype(np.uint8)
            region256_hm = cv2.addWeighted(color_block256, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
            region256_hm[mask256==0] = 0
            img_inverse = save_region.copy()
            img_inverse[mask256 == 0.95] = 0
            ths.append(region256_hm+img_inverse)
            
        ths = [Image.fromarray(img) for img in ths]
            
        getConcatImage([getConcatImage(ths[0:3]), 
                        getConcatImage(ths[3:6])], how='vertical').save(os.path.join(output_dir, '%s_256th.png' % (fname)))
    
    
    hms = []
    for i in range(6):
        score256_1 = get_scores256(a256_1[:,i,:,:], size=(s,)*2)
        score256_2 = get_scores256(a256_2[:,i,:,:], size=(s,)*2)
        new_score256_2 = np.zeros_like(score256_2)
        new_score256_2[offset_2:s, offset_2:s] = score256_2[:(s-offset_2), :(s-offset_2)]
        overlay256 = np.ones_like(score256_2)*100
        overlay256[offset_2:s, offset_2:s] += 100
        score256 = (score256_1+new_score256_2)/overlay256
        color_block256 = (cmap(score256)*255)[:,:,:3].astype(np.uint8)
        region256_hm = cv2.addWeighted(color_block256, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
        hms.append(region256_hm)
        
    hms = [Image.fromarray(img) for img in hms]
        
    getConcatImage([getConcatImage(hms[0:3]), 
                    getConcatImage(hms[3:6])], how='vertical').save(os.path.join(output_dir, '%s_256hm.png' % (fname)))

    
def hipt_forward_pass(region, model256, model4k, scale=1,
                                device256=torch.device('cuda:0'), 
                                device4k=torch.device('cuda:1')):
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        )
    ])

    with torch.no_grad():   
        batch_256 = t(region).unsqueeze(0).unfold(2, 256, 256).unfold(3, 256, 256)
        batch_256 = rearrange(batch_256, 'b c p1 p2 w h -> (b p1 p2) c w h')
        batch_256 = batch_256.to(device256, non_blocking=True)
        features_256 = model256(batch_256)
        features_256 = features_256.unfold(0, 16, 16).transpose(0,1).unsqueeze(dim=0)
        features_4096 = model4k.forward(features_256.to(device4k))
        return features_4096


def get_region_attention_scores(region, model256, model4k, scale=1,
                                device256=torch.device('cuda:0'), 
                                device4k=torch.device('cuda:1')):
    #Modified to use cpu, original code has  device256=torch.device('cuda:0'), device4k=torch.device('cuda:1'))
    r"""
    Forward pass in hierarchical model with attention scores saved.
    
    Args:
    - region (PIL.Image):       4096 x 4096 Image 
    - model256 (torch.nn):      256-Level ViT 
    - model4k (torch.nn):       4096-Level ViT 
    - scale (int):              How much to scale the output image by (e.g. - scale=4 will resize images to be 1024 x 1024.)
    
    Returns:
    - np.array: [256, 256/scale, 256/scale, 3] np.array sequence of image patches from the 4K x 4K region.
    - attention_256 (torch.Tensor): [256, 256/scale, 256/scale, 3] torch.Tensor sequence of attention maps for 256-sized patches.
    - attention_4k (torch.Tensor): [1, 4096/scale, 4096/scale, 3] torch.Tensor sequence of attention maps for 4k-sized regions.
    """
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        )
    ])

    with torch.no_grad():   
        batch_256 = t(region).unsqueeze(0).unfold(2, 256, 256).unfold(3, 256, 256)
        batch_256 = rearrange(batch_256, 'b c p1 p2 w h -> (b p1 p2) c w h')
        batch_256 = batch_256.to(device256, non_blocking=True)
        features_256 = model256(batch_256)

        attention_256 = model256.get_last_selfattention(batch_256)
        nh = attention_256.shape[1] # number of head
        attention_256 = attention_256[:, :, 0, 1:].reshape(256, nh, -1)
        attention_256 = attention_256.reshape(256, nh, 16, 16)
        attention_256 = nn.functional.interpolate(attention_256, scale_factor=int(16/scale), mode="nearest").cpu().numpy()

        features_4096 = features_256.unfold(0, 16, 16).transpose(0,1).unsqueeze(dim=0)
        attention_4096 = model4k.get_last_selfattention(features_4096.detach().to(device4k))
        nh = attention_4096.shape[1] # number of head
        attention_4096 = attention_4096[0, :, 0, 1:].reshape(nh, -1)
        attention_4096 = attention_4096.reshape(nh, 16, 16)
        attention_4096 = nn.functional.interpolate(attention_4096.unsqueeze(0), scale_factor=int(256/scale), mode="nearest")[0].cpu().numpy()

        if scale != 1:
            batch_256 = nn.functional.interpolate(batch_256, scale_factor=(1/scale), mode="nearest")

    return tensorbatch2im(batch_256), attention_256, attention_4096

def create_heatmap_for_svs(svs_path,dir_path, model256, model4k, output_dir,head1,head2, fname='test',
                             offset=128, scale=4, alpha=0.5, cmap = plt.get_cmap('coolwarm')):
    

    
    print("######CROPPING SVS INTO IMAGES######")
    x_range,y_range = crop_svs_image(svs_path,dir_path,crop_size=(4096, 4096))
    print("######DONE CROPPING SVS INTO IMAGES######")
    
    slide = openslide.open_slide(svs_path)
    slide_width, slide_height = slide.dimensions
    x_range=slide_width//4096
    y_range=slide_height//4096
    print("######TURNING IMAGES INTO HEATMAPS######")
    for y in tqdm(range(y_range)):
        #print("                                                              Y =%s" % y)
        for x in range(x_range):
            if os.path.exists('%s/%s_%s_%s.png' %(dir_path,"crop", int(x),int(y))):
                #print("                                                              X =%s" % x)
                create_hierarchical_heatmaps_indiv_juan_fact(Image.open('%s/%s_%s_%s.png' %(dir_path,"crop", int(x),int(y))).convert('RGB'), model256, model4k, output_dir,head1,head2, fname,
                                offset=128, scale=4, alpha=0.5, cmap = plt.get_cmap('coolwarm')).save(os.path.join(dir_path, '%s_%s_%s.png' % ("hm", int(x),int(y))))
    print("######DONE TURNING IMAGES INTO HEATMAPS######")
    
    
    print("######STITCHING HEATMAPS INTO BIG HEATMAP HORIZONTALLY######")
    for y in tqdm(range(y_range)):
        image_h = Image.open('%s/%s_%s_%s.png' %(dir_path,"hm", 0,int(y)))
        for x in range(x_range):
            if os.path.exists('%s/%s_%s_%s.png' %(dir_path,"hm", int(x),int(y))):
                if x == 0:   
                    continue
                temp_image = Image.open('%s/%s_%s_%s.png' %(dir_path,"hm", int(x),int(y)))
                c_image = Image.new('RGB', (image_h.width + temp_image.width, image_h.height))
                c_image.paste(image_h, (0, 0))
                c_image.paste(temp_image, (image_h.width, 0))
                image_h = c_image
        image_h.save(os.path.join(dir_path, '%s_%s.png' % ("h", int(y))))
    print("######DONE STITCHING HEATMAPS INTO BIG HEATMAP HORIZONTALLY######")
    

    print("######STITCHING HEATMAPS INTO BIG HEATMAP VERTICALLY######")
    array_v = np.asarray(Image.open('%s/%s_%s.png' %(dir_path,"h",0)))
    for y in tqdm(range(1,y_range)):
        print(array_v.shape)
        if os.path.exists('%s/%s_%s.png' %(dir_path,"h", int(y))):
            print("concatenating for y=",y)
            array_v = np.concatenate((array_v,np.asarray(Image.open('%s/%s_%s.png' %(dir_path,"h",int(y))))), axis=0)
    Image.fromarray(array_v).save(os.path.join(dir_path, 'full_heatmap.png'))

            #Image.open("image.png")






    '''
    regions_hm=[]

    for i in tqdm(images):

        region_hm = create_hierarchical_heatmaps_indiv_juan_fact(i, model256, model4k, output_dir,head1,head2, fname,
                             offset=128, scale=4, alpha=0.5, cmap = plt.get_cmap('coolwarm'))
        regions_hm.append(region_hm)

    region_h=regions_hm[0]
    region_hw=regions_hm[0]
    for y in range(y_range):
        for x in range(x_range):
            if x == 0:
                region_h = regions_hm[x*(y+1)]
                continue
            region_h = np.concatenate(region_h,regions_hm[x*(y+1)],axis=1)
        if y == 0:
            region_hw=region_h
            continue
        region_hw = np.concatenate(region_hw,region_h,axis=0)
    

    Image.fromarray(region_hw).save(os.path.join(output_dir, '%s_h1-%s_h2-%s.png' % (fname,head1+1,head2+2)))
    '''

    return





def crop_svs_image(file_path,dir_path,crop_size=(4096, 4096)):
    """
    Crop a whole slide image into non-overlapping images of specified size.
    Args:
        file_path (str): the path to the .svs file
        crop_size (tuple): a tuple of the form (width, height) specifying the size of the cropped images
    Returns:
        list: a list of PIL image objects
    """

    if not os.path.exists(dir_path):
        os.system("mkdir %s" %(dir_path))

    # Open the slide
    slide = openslide.open_slide(file_path)
    
    # Get the dimensions of the slide
    slide_width, slide_height = slide.dimensions

    # Initialize an empty list to store the cropped images
    images = []

    # Iterate over the slide, crop it into non-overlapping images of specified size
    x_range=slide_width//crop_size[0]
    y_range=slide_height//crop_size[1]
    for y in tqdm(range(0, slide_height, crop_size[1])):
        for x in range(0, slide_width, crop_size[0]):
            if x+crop_size[0] <= slide_width and y+crop_size[1] <= slide_height:
                image = slide.read_region((x,y), 0, crop_size)

                xx=0
                yy=0
                if x==0:
                    xx=0
                else:
                    xx=x/crop_size[0]
                if y==0:
                    yy=0
                else:
                    yy=y/crop_size[1]

                image.save(os.path.join(dir_path, '%s_%s_%s.png' % ("crop", int(xx),int(yy))))
    return x_range,y_range

import os
import time
import numpy as np
from PIL import Image
from tqdm import tqdm
import openslide
#from histolab.masks import OtsuTissueMasker

def crop_and_save_svs(file_path, output_path, crop_size=(256, 256), tissue_threshold=0.33):
    t1 = time.time()
    print("Processing the slide from", file_path)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Open the slide
    slide = openslide.open_slide(file_path)
    
    # Get the dimensions of the slide
    slide_width, slide_height = slide.dimensions
    
    target_resolution = 20.0  # x20 magnification
    downsample_factor = float(slide.properties['openslide.mpp-x']) / target_resolution
    level = slide.get_best_level_for_downsample(downsample_factor)
    
    # Iterate over the slide, crop it into non-overlapping images of specified size
    for y in tqdm(range(0, slide_height, crop_size[1])):
        for x in range(0, slide_width, crop_size[0]):
            if x + crop_size[0] <= slide_width and y + crop_size[1] <= slide_height:
                image = slide.read_region((x, y), level, crop_size).convert('RGB')
                image_np = np.array(image)  # Convert PIL image to numpy array
                masker = OtsuTissueMasker()
                # Fit the masker to the image
                masker.fit(np.array([image_np]))
                # Generate a tissue mask using the fitted masker
                tissue_mask = masker.transform(np.array([image_np]))[0]
                # Calculate the percentage of tissue in the region
                tissue_percent = np.count_nonzero(tissue_mask) / tissue_mask.size * 100
                # Check if the tissue percentage is greater than the threshold
                if tissue_percent >= tissue_threshold:
                    # Save the image to the output directory
                    image.save(os.path.join(output_path, f"crop_x{x}_y{y}.png"))
    
    t2 = time.time()
    print(f"Processing completed in {t2 - t1} seconds.")


def crop_and_save_svs_multithread(file_path, zip_file_path, crop_size=(256, 256), tissue_threshold=0.33):
    t1 = time.time()
    print("Processing the slide from", file_path)

    #if not os.path.exists(zip_file_path):
    #    os.makedirs(zip_file_path)

    # Create a new ZIP file if it doesn't exist
    if not os.path.exists(zip_file_path):
        with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED):
            pass  # Just creating the zip file, no need to add any files yet
    try:
        # Attempt to open the slide
        slide = openslide.open_slide(file_path)
        
        # Proceed with your existing code to process the slide...
        slide_width, slide_height = slide.dimensions
        # Your existing processing code continues here...

    except openslide.lowlevel.OpenSlideUnsupportedFormatError as e:
        print(f"Error opening slide: {e}")
        # Handle the error, for example, by logging it, skipping this slide, etc.
        with open('svs_error_files.txt', 'a') as error_file:
            error_file.write(f"{file_path}\n")
        return  # Skip processing this slide
    except Exception as e:
        print(f"An unexpected error occurred with the slide: {e}")
        # Handle unexpected errors
        with open('svs_that_did_not_open.txt', 'a') as error_file:
            error_file.write(f"{file_path}\n")
        return  # Skip processing this slide
    
    target_resolution = 20.0  # x20 magnification
    downsample_factor = float(slide.properties['openslide.mpp-x']) / target_resolution
    level = slide.get_best_level_for_downsample(downsample_factor)
    print("We pushed thru errors")
    #zip_lock = threading.Lock()
    # Define a function to process each crop and save the image
    # Create a temporary folder for storing PNG files
    print("yikes")
    temp_folder = f"/temp_{int(time.time())}"
    os.chdir("scratch")
    os.makedirs(temp_folder)
    print("it worked")
    assert 1==0
    def process_crop(x, y):
        image = slide.read_region((x, y), level, crop_size).convert('RGB')
        image_np = np.array(image)  # Convert PIL image to numpy array
        masker = OtsuTissueMasker()
        # Fit the masker to the image
        masker.fit(np.array([image_np]))
        # Generate a tissue mask using the fitted masker
        tissue_mask = masker.transform(np.array([image_np]))[0]
        # Calculate the percentage of tissue in the region
        tissue_percent = np.count_nonzero(tissue_mask) / tissue_mask.size * 100
        # Check if the tissue percentage is greater than the threshold
        if tissue_percent >= tissue_threshold:
            # Save the image to the output directory
            # Convert the PIL image to a byte buffer
            #img_byte_arr = io.BytesIO()
            #image.save(img_byte_arr, format='PNG')
            #img_byte_arr = img_byte_arr.getvalue()     
            timestamp = int(time.time() * 1000)       
            image.save(os.path.join(temp_folder, f"crop_x{x}_y{y}_{timestamp}.png"))
            #with zipfile.ZipFile(zip_file_path, 'a', zipfile.ZIP_DEFLATED) as zipf:
                #zipf.writestr(f"crop_x{x}_y{y}_{timestamp}.png", img_byte_arr)

    # Create a list of arguments for processing each crop
    args_list = [(x, y) for y in range(0, slide_height, crop_size[1])
                         for x in range(0, slide_width, crop_size[0])
                         if x + crop_size[0] <= slide_width and y + crop_size[1] <= slide_height]

    # Use multithreading to process crops in parallel
    print("This is right before concurrent.futures")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda args: process_crop(*args), args_list), total=len(args_list)))

    print("Files to temp folder done, now we put them in the zip")
    total_files = sum(len(files) for _, _, files in os.walk(temp_folder))
    # Add PNG files from the temporary folder to the ZIP file
    with tqdm(total=total_files, dec="Adding files to ZIP", unit="files") as pbar:
        with zipfile.ZipFile(zip_file_path, 'a', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(temp_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, temp_folder))

    # Clean up: Delete the temporary folder
    if os.path.exists(temp_folder):
        for root, _, files in os.walk(temp_folder):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
        os.rmdir(temp_folder)



    t2 = time.time()
    print(f"Processing completed in {t2 - t1} seconds.")

# Example usage:
# crop_and_save_svs(file_path='path_to_svs_file', output_path='path_to_save_cropped_images')



def create_embedding_for_svs(svs_path,dir_path, model):
    

    
    print("######CROPPING SVS INTO IMAGES######")
    x_range,y_range = crop_svs_image(svs_path,dir_path,crop_size=(4096, 4096))
    print("######DONE CROPPING SVS INTO IMAGES######")
    
    slide = openslide.open_slide(svs_path)
    slide_width, slide_height = slide.dimensions
    x_range=slide_width//4096
    y_range=slide_height//4096
    print("######TURNING IMAGES INTO EMBEDDINGS######")
    tensor_list=[]
    for y in tqdm(range(y_range)):
        #print("                                                              Y =%s" % y)
        for x in range(x_range):
            if os.path.exists('%s/%s_%s_%s.png' %(dir_path,"crop", int(x),int(y))):
                crop = Image.open('%s/%s_%s_%s.png' %(dir_path,"crop", int(x),int(y))).convert('RGB') 
                reg = eval_transforms()(crop).unsqueeze(dim=0) 
                tensor_list.append(model(reg))   
    print("######DONE TURNING IMAGES INTO EMBEDDINGS######")
    embedding = torch.cat(tensor_list,dim=0)
    torch.save(embedding,dir_path+'embedding.pt')
    return embedding

def embed_svs_v2(file_path,embed_name,model,crop_size=(4096, 4096)):
    t1 = time.time()
    print("we are inside embed_svs_v2 in attention_juan.py")
    """
    Crop a whole slide image into non-overlapping images of specified size.
    Args:
        file_path (str): the path to the .svs file
        crop_size (tuple): a tuple of the form (width, height) specifying the size of the cropped images
    Returns:
        list: a list of PIL image objects
    """

    #print("error here?")
    # Open the slide
    print(file_path)
    print("@@@@@@@@")
    slide = openslide.open_slide(file_path)
    print("able_to_open_file_here")
    tissue_threshold = 0.25
    
    # Get the dimensions of the slide
    slide_width, slide_height = slide.dimensions



    # Determine the downsample factor
    downsamples = slide.level_downsamples
    target_resolution = 20.0  # x20 magnification

    if 'openslide.mpp-x' in slide.properties:
        downsample_factor = float(slide.properties['openslide.mpp-x']) / target_resolution
    else:
        print("Warning: 'openslide.mpp-x' not found in slide properties. Using level[0] downsample factor.")
        downsample_factor = float(slide.properties['openslide.level[0].downsample'])

    level = slide.get_best_level_for_downsample(downsample_factor)



    # Initialize an empty list to store tensors
    tensor_list=[]

    # Iterate over the slide, crop it into non-overlapping images of specified size
    x_range=slide_width//crop_size[0]
    y_range=slide_height//crop_size[1]
    with torch.no_grad():
        for y in tqdm(range(0, slide_height, crop_size[1])):
            for x in range(0, slide_width, crop_size[0]):
                if x+crop_size[0] <= slide_width and y+crop_size[1] <= slide_height:
                    image = slide.read_region((x,y), level, crop_size).convert('RGB')
                    image_np = np.array(image)  # Convert PIL image to numpy array
                    masker = OtsuTissueMasker()
                    # Fit the masker to the image
                    masker.fit(np.array([image_np]))
                    # Generate a tissue mask using the fitted masker
                    tissue_mask = masker.transform(np.array([image_np]))[0]
                    # Calculate the percentage of tissue in the region
                    tissue_percent = np.count_nonzero(tissue_mask) / tissue_mask.size * 100
                    # Check if the tissue percentage is greater than 35%
                    if tissue_percent >= tissue_threshold:
                        processed_image = eval_transforms()(image).unsqueeze(dim=0)
                        tensor_list.append(model(processed_image))
                        print("tensor_list@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")


    if len(tensor_list)==0:
        print("exclusing_slide_because_has_no_tissue_regions")
        return 0 
    embedding = torch.cat(tensor_list,dim=0)

    print("The embeddings is",embedding)
    print("the type is", type(embedding))

    #embed_array = embedding.cpu().numpy()
    #np.save(embed_name, embed_array)

    torch.save(embedding,embed_name)  
    t2 = time.time()
    print("it took ",t2-t1," seconds.")
    return embedding

def save_svs_masked_patches(file_path,dir_path,embed_name,model,crop_size=(4096, 4096)):
    """
    Crop a whole slide image into non-overlapping images of specified size.
    Args:
        file_path (str): the path to the .svs file
        crop_size (tuple): a tuple of the form (width, height) specifying the size of the cropped images
    Returns:
        list: a list of PIL image objects
    """


    # Open the slide
    slide = openslide.open_slide(file_path)
    tissue_threshold = 0.25
    
    # Get the dimensions of the slide
    slide_width, slide_height = slide.dimensions

    downsamples = slide.level_downsamples

    target_resolution = 20.0  # x20 magnification
    downsample_factor = float(slide.properties['openslide.mpp-x']) / target_resolution

    level = slide.get_best_level_for_downsample(downsample_factor)


    # Initialize an empty list to store tensors
    tensor_list=[]


    #image.save(os.path.join(dir_path, '%s_%s_%s.png' % ("crop", int(xx),int(yy))))
    # Iterate over the slide, crop it into non-overlapping images of specified size
    x_range=slide_width//crop_size[0]
    y_range=slide_height//crop_size[1]
    for y in tqdm(range(0, slide_height, crop_size[1])):
        for x in range(0, slide_width, crop_size[0]):
            if x+crop_size[0] <= slide_width and y+crop_size[1] <= slide_height:
                image = slide.read_region((x,y), level, crop_size).convert('RGB')
                image_np = np.array(image)  # Convert PIL image to numpy array
                masker = OtsuTissueMasker()
                # Fit the masker to the image
                masker.fit(np.array([image_np]))
                # Generate a tissue mask using the fitted maskersource ~/HIPT_Embedding_Env/bin/activate
                tissue_mask = masker.transform(np.array([image_np]))[0]
                # Calculate the percentage of tissue in the region
                tissue_percent = np.count_nonzero(tissue_mask) / tissue_mask.size * 100
                # Check if the tissue percentage is greater than 25%
                if tissue_percent >= tissue_threshold:
                    image.save(os.path.join(dir_path, '%s_%s_%s.png' % ("crop", int(y),int(x))))

def compute_tissue_percent(file_path):
    """
    Computes the tissue percentage of a given .svs file.
    Args:
        file_path (str): the path to the .svs file
    Returns:
        float: the tissue percentage of the slide
    """
    slide = openslide.open_slide(file_path)
    slide_width, slide_height = slide.dimensions
    downsamples = slide.level_downsamples
    target_resolution = 20.0  # x20 magnification
    downsample_factor = float(slide.properties['openslide.mpp-x']) / target_resolution
    level = slide.get_best_level_for_downsample(downsample_factor)
    tile_size = 4096
    tissue_mask = np.zeros((slide_height, slide_width), dtype=bool)
    for y in range(0, slide_height, tile_size):
        y = min(y, slide_height - tile_size)  # ensure that the tile doesn't extend beyond the slide
        for x in range(0, slide_width, tile_size):
            x = min(x, slide_width - tile_size)  # ensure that the tile doesn't extend beyond the slide
            tile_width = min(tile_size, slide_width - x)  # adjust tile width if at the right edge of the slide
            tile_height = min(tile_size, slide_height - y)  # adjust tile height if at the bottom edge of the slide
            image = slide.read_region((x, y), level, (tile_width, tile_height)).convert('RGB')
            image_np = np.array(image)
            masker = OtsuTissueMasker()
            masker.fit(np.array([image_np]))
            tile_tissue_mask = masker.transform(np.array([image_np]))[0]
            tissue_mask[y:y + tile_height, x:x + tile_width] = tile_tissue_mask
    tissue_percent = np.count_nonzero(tissue_mask) / tissue_mask.size * 100
    return tissue_percent

def select_svs(folder_path, output_folder, crop_size=(4096, 4096)):
    """
    Selects the .svs file in the given folder_path that has the most tissue coverage and saves the
    cropped patches of the selected .svs file with tissue coverage to the specified output folder.
    Args:
        folder_path (str): the path to the folder containing the .svs files
        output_folder (str): the path to the output folder where the selected cropped patches will be saved
        crop_size (tuple): a tuple of the form (width, height) specifying the size of the cropped images
    Returns:
        str: the path to the .svs file with the most tissue coverage
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    max_tissue_percent = 0.0
    max_tissue_svs = ''
    for filename in os.listdir(folder_path):
        if filename.endswith('.svs'):
            file_path = os.path.join(folder_path, filename)
            tissue_percent = compute_tissue_percent(file_path)
            if tissue_percent > max_tissue_percent:
                max_tissue_percent = tissue_percent
                max_tissue_svs = file_path
    if max_tissue_svs:
        save_svs_masked_patches(max_tissue_svs, output_folder, crop_size=crop_size)
    return max_tissue_svs

def create_hierarchical_heatmaps_indiv_juan_fact(region, model256, model4k, output_dir,head1,head2, fname,
                             offset=128, scale=4, alpha=0.5, cmap = plt.get_cmap('coolwarm')):
    r"""
    Creates hierarchical heatmaps (Raw H&E + ViT-256 + ViT-4K + Blended Heatmaps saved individually).  
    
    Args:
    - region (PIL.Image):       4096 x 4096 Image 
    - model256 (torch.nn):      256-Level ViT 
    - model4k (torch.nn):       4096-Level ViT 
    - output_dir (str):         Save directory / subdirectory
    - fname (str):              Naming structure of files
    - offset (int):             How much to offset (from top-left corner with zero-padding) the region by for blending 
    - scale (int):              How much to scale the output image by 
    - alpha (float):            Image blending factor for cv2.addWeighted
    - cmap (matplotlib.pyplot): Colormap for creating heatmaps
    
    Returns:
    - None
    """
    
    region2 = add_margin(region.crop((128,128,4096,4096)), 
                     top=0, left=0, bottom=128, right=128, color=(255,255,255))
    region3 = add_margin(region.crop((128*2,128*2,4096,4096)), 
                     top=0, left=0, bottom=128*2, right=128*2, color=(255,255,255))
    region4 = add_margin(region.crop((128*3,128*3,4096,4096)), 
                     top=0, left=0, bottom=128*4, right=128*4, color=(255,255,255))
    
    b256_1, a256_1, a4k_1 = get_region_attention_scores(region, model256, model4k, scale)
    
    b256_2, a256_2, a4k_2 = get_region_attention_scores(region2, model256, model4k, scale)
    b256_3, a256_3, a4k_3 = get_region_attention_scores(region3, model256, model4k, scale)
    b256_4, a256_4, a4k_4 = get_region_attention_scores(region4, model256, model4k, scale)
    offset_2 = (offset*1)//scale
    offset_3 = (offset*2)//scale
    offset_4 = (offset*3)//scale
    s = 4096//scale
    save_region = np.array(region.resize((s, s)))
    if False:
        for j in range(6):
            score4k_1 = concat_scores4k(a4k_1[j], size=(s,)*2)
            score4k = score4k_1 / 100
            color_block4k = (cmap(score4k)*255)[:,:,:3].astype(np.uint8)
            region4k_hm = cv2.addWeighted(color_block4k, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
            Image.fromarray(region4k_hm).save(os.path.join(output_dir, '%s_4k[%s].png' % (fname, j)))
        
    j=head2
        
    i=head1
  
    score4k_1 = concat_scores4k(a4k_1[j], size=(s,)*2)
    score4k_2 = concat_scores4k(a4k_2[j], size=(s,)*2)
    score4k_3 = concat_scores4k(a4k_3[j], size=(s,)*2)
    score4k_4 = concat_scores4k(a4k_4[j], size=(s,)*2)

    new_score4k_2 = np.zeros_like(score4k_2)
    new_score4k_2[offset_2:s, offset_2:s] = score4k_2[:(s-offset_2), :(s-offset_2)]
    new_score4k_3 = np.zeros_like(score4k_3)
    new_score4k_3[offset_3:s, offset_3:s] = score4k_3[:(s-offset_3), :(s-offset_3)]
    new_score4k_4 = np.zeros_like(score4k_4)
    new_score4k_4[offset_4:s, offset_4:s] = score4k_4[:(s-offset_4), :(s-offset_4)]

    overlay4k = np.ones_like(score4k_2)*100
    overlay4k[offset_2:s, offset_2:s] += 100
    overlay4k[offset_3:s, offset_3:s] += 100
    overlay4k[offset_4:s, offset_4:s] += 100
    score4k = (score4k_1+new_score4k_2+new_score4k_3+new_score4k_4)/overlay4k

    score256_1 = concat_scores256(a256_1[:,i,:,:], size=(s//16,)*2)
    score256_2 = concat_scores256(a256_2[:,i,:,:], size=(s//16,)*2)
    new_score256_2 = np.zeros_like(score256_2)
    new_score256_2[offset_2:s, offset_2:s] = score256_2[:(s-offset_2), :(s-offset_2)]
    overlay256 = np.ones_like(score256_2)*100*2
    overlay256[offset_2:s, offset_2:s] += 100*2
    score256 = (score256_1+new_score256_2)*2/overlay256

    factorize = lambda data: (data - np.min(data)) / (np.max(data) - np.min(data))
    score = (score4k*overlay4k+score256*overlay256)/(overlay4k+overlay256) #factorize(score256*score4k)
    color_block = (cmap(score)*255)[:,:,:3].astype(np.uint8)
    region_hm = cv2.addWeighted(color_block, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
    #Image.fromarray(region_hm).save(os.path.join(output_dir, '%s_factorized_4k[%s]_256[%s].png' % (fname, j, i)))           
    
    #return Image.fromarray(region_hm)
    return Image.fromarray(region_hm)

def create_hierarchical_heatmaps_indiv_juan(region, model256, model4k, output_dir,head1,head2, fname,
                             offset=128, scale=4, alpha=0.5, cmap = plt.get_cmap('coolwarm')):
    r"""
    Creates hierarchical heatmaps (Raw H&E + ViT-256 + ViT-4K + Blended Heatmaps saved individually).  
    
    Args:
    - region (PIL.Image):       4096 x 4096 Image 
    - model256 (torch.nn):      256-Level ViT 
    - model4k (torch.nn):       4096-Level ViT 
    - output_dir (str):         Save directory / subdirectory
    - fname (str):              Naming structure of files
    - offset (int):             How much to offset (from top-left corner with zero-padding) the region by for blending 
    - scale (int):              How much to scale the output image by 
    - alpha (float):            Image blending factor for cv2.addWeighted
    - cmap (matplotlib.pyplot): Colormap for creating heatmaps
    
    Returns:
    - None
    """
    
    region2 = add_margin(region.crop((128,128,4096,4096)), 
                     top=0, left=0, bottom=128, right=128, color=(255,255,255))
    region3 = add_margin(region.crop((128*2,128*2,4096,4096)), 
                     top=0, left=0, bottom=128*2, right=128*2, color=(255,255,255))
    region4 = add_margin(region.crop((128*3,128*3,4096,4096)), 
                     top=0, left=0, bottom=128*4, right=128*4, color=(255,255,255))
    
    b256_1, a256_1, a4k_1 = get_region_attention_scores(region, model256, model4k, scale)
    
    b256_2, a256_2, a4k_2 = get_region_attention_scores(region2, model256, model4k, scale)
    b256_3, a256_3, a4k_3 = get_region_attention_scores(region3, model256, model4k, scale)
    b256_4, a256_4, a4k_4 = get_region_attention_scores(region4, model256, model4k, scale)
    offset_2 = (offset*1)//scale
    offset_3 = (offset*2)//scale
    offset_4 = (offset*3)//scale
    s = 4096//scale
    save_region = np.array(region.resize((s, s)))
    if False:
        for j in range(6):
            score4k_1 = concat_scores4k(a4k_1[j], size=(s,)*2)
            score4k = score4k_1 / 100
            color_block4k = (cmap(score4k)*255)[:,:,:3].astype(np.uint8)
            region4k_hm = cv2.addWeighted(color_block4k, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
            Image.fromarray(region4k_hm).save(os.path.join(output_dir, '%s_4k[%s].png' % (fname, j)))
        
    j=head2
    score4k_1 = concat_scores4k(a4k_1[j], size=(s,)*2)
    score4k_2 = concat_scores4k(a4k_2[j], size=(s,)*2)
    score4k_3 = concat_scores4k(a4k_3[j], size=(s,)*2)
    score4k_4 = concat_scores4k(a4k_4[j], size=(s,)*2)

    new_score4k_2 = np.zeros_like(score4k_2)
    new_score4k_2[offset_2:s, offset_2:s] = score4k_2[:(s-offset_2), :(s-offset_2)]
    new_score4k_3 = np.zeros_like(score4k_3)
    new_score4k_3[offset_3:s, offset_3:s] = score4k_3[:(s-offset_3), :(s-offset_3)]
    new_score4k_4 = np.zeros_like(score4k_4)
    new_score4k_4[offset_4:s, offset_4:s] = score4k_4[:(s-offset_4), :(s-offset_4)]

    overlay4k = np.ones_like(score4k_2)*100
    overlay4k[offset_2:s, offset_2:s] += 100
    overlay4k[offset_3:s, offset_3:s] += 100
    overlay4k[offset_4:s, offset_4:s] += 100
    score4k = (score4k_1+new_score4k_2+new_score4k_3+new_score4k_4)/overlay4k

    color_block4k = (cmap(score4k)*255)[:,:,:3].astype(np.uint8)
    region4k_hm = cv2.addWeighted(color_block4k, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
    Image.fromarray(region4k_hm).save(os.path.join(output_dir, '%s_1024[%s].png' % (fname, j)))
        
    i=head1
    score256_1 = concat_scores256(a256_1[:,i,:,:], size=(s//16,)*2)
    score256_2 = concat_scores256(a256_2[:,i,:,:], size=(s//16,)*2)
    new_score256_2 = np.zeros_like(score256_2)
    new_score256_2[offset_2:s, offset_2:s] = score256_2[:(s-offset_2), :(s-offset_2)]
    overlay256 = np.ones_like(score256_2)*100
    overlay256[offset_2:s, offset_2:s] += 100
    score256 = (score256_1+new_score256_2)/overlay256
    color_block256 = (cmap(score256)*255)[:,:,:3].astype(np.uint8)
    region256_hm = cv2.addWeighted(color_block256, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
    Image.fromarray(region256_hm).save(os.path.join(output_dir, '%s_256[%s].png' % (fname, i)))
    
    score4k_1 = concat_scores4k(a4k_1[j], size=(s,)*2)
    score4k_2 = concat_scores4k(a4k_2[j], size=(s,)*2)
    score4k_3 = concat_scores4k(a4k_3[j], size=(s,)*2)
    score4k_4 = concat_scores4k(a4k_4[j], size=(s,)*2)

    new_score4k_2 = np.zeros_like(score4k_2)
    new_score4k_2[offset_2:s, offset_2:s] = score4k_2[:(s-offset_2), :(s-offset_2)]
    new_score4k_3 = np.zeros_like(score4k_3)
    new_score4k_3[offset_3:s, offset_3:s] = score4k_3[:(s-offset_3), :(s-offset_3)]
    new_score4k_4 = np.zeros_like(score4k_4)
    new_score4k_4[offset_4:s, offset_4:s] = score4k_4[:(s-offset_4), :(s-offset_4)]

    overlay4k = np.ones_like(score4k_2)*100
    overlay4k[offset_2:s, offset_2:s] += 100
    overlay4k[offset_3:s, offset_3:s] += 100
    overlay4k[offset_4:s, offset_4:s] += 100
    score4k = (score4k_1+new_score4k_2+new_score4k_3+new_score4k_4)/overlay4k

    score256_1 = concat_scores256(a256_1[:,i,:,:], size=(s//16,)*2)
    score256_2 = concat_scores256(a256_2[:,i,:,:], size=(s//16,)*2)
    new_score256_2 = np.zeros_like(score256_2)
    new_score256_2[offset_2:s, offset_2:s] = score256_2[:(s-offset_2), :(s-offset_2)]
    overlay256 = np.ones_like(score256_2)*100*2
    overlay256[offset_2:s, offset_2:s] += 100*2
    score256 = (score256_1+new_score256_2)*2/overlay256

    factorize = lambda data: (data - np.min(data)) / (np.max(data) - np.min(data))
    score = (score4k*overlay4k+score256*overlay256)/(overlay4k+overlay256) #factorize(score256*score4k)
    color_block = (cmap(score)*255)[:,:,:3].astype(np.uint8)
    region_hm = cv2.addWeighted(color_block, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
    Image.fromarray(region_hm).save(os.path.join(output_dir, '%s_factorized_4k[%s]_256[%s].png' % (fname, j, i)))
            
    return

def create_hierarchical_heatmaps_indiv(region, model256, model4k, output_dir, fname,
                             offset=128, scale=4, alpha=0.5, cmap = plt.get_cmap('coolwarm'), threshold=None):
    r"""
    Creates hierarchical heatmaps (Raw H&E + ViT-256 + ViT-4K + Blended Heatmaps saved individually).  
    
    Args:
    - region (PIL.Image):       4096 x 4096 Image 
    - model256 (torch.nn):      256-Level ViT 
    - model4k (torch.nn):       4096-Level ViT 
    - output_dir (str):         Save directory / subdirectory
    - fname (str):              Naming structure of files
    - offset (int):             How much to offset (from top-left corner with zero-padding) the region by for blending 
    - scale (int):              How much to scale the output image by 
    - alpha (float):            Image blending factor for cv2.addWeighted
    - cmap (matplotlib.pyplot): Colormap for creating heatmaps
    
    Returns:
    - None
    """
    
    region2 = add_margin(region.crop((128,128,4096,4096)), 
                     top=0, left=0, bottom=128, right=128, color=(255,255,255))
    region3 = add_margin(region.crop((128*2,128*2,4096,4096)), 
                     top=0, left=0, bottom=128*2, right=128*2, color=(255,255,255))
    region4 = add_margin(region.crop((128*3,128*3,4096,4096)), 
                     top=0, left=0, bottom=128*4, right=128*4, color=(255,255,255))
    
    b256_1, a256_1, a4k_1 = get_region_attention_scores(region, model256, model4k, scale)
    
    b256_2, a256_2, a4k_2 = get_region_attention_scores(region2, model256, model4k, scale)
    b256_3, a256_3, a4k_3 = get_region_attention_scores(region3, model256, model4k, scale)
    b256_4, a256_4, a4k_4 = get_region_attention_scores(region4, model256, model4k, scale)
    offset_2 = (offset*1)//scale
    offset_3 = (offset*2)//scale
    offset_4 = (offset*3)//scale
    s = 4096//scale
    save_region = np.array(region.resize((s, s)))
    
    if threshold != None:
        for i in range(6):
            score256_1 = concat_scores256(a256_1[:,i,:,:], size=(s//16,)*2)
            score256_2 = concat_scores256(a256_2[:,i,:,:], size=(s//16,)*2)
            new_score256_2 = np.zeros_like(score256_2)
            new_score256_2[offset_2:s, offset_2:s] = score256_2[:(s-offset_2), :(s-offset_2)]
            overlay256 = np.ones_like(score256_2)*100
            overlay256[offset_2:s, offset_2:s] += 100
            score256 = (score256_1+new_score256_2)/overlay256
            
            mask256 = score256.copy()
            mask256[mask256 < threshold] = 0
            mask256[mask256 > threshold] = 0.95
            
            color_block256 = (cmap(mask256)*255)[:,:,:3].astype(np.uint8)
            region256_hm = cv2.addWeighted(color_block256, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
            region256_hm[mask256==0] = 0
            img_inverse = save_region.copy()
            img_inverse[mask256 == 0.95] = 0
            Image.fromarray(region256_hm+img_inverse).save(os.path.join(output_dir, '%s_256th[%d].png' % (fname, i)))
    
    if False:
        for j in range(6):
            score4k_1 = concat_scores4k(a4k_1[j], size=(s,)*2)
            score4k = score4k_1 / 100
            color_block4k = (cmap(score4k)*255)[:,:,:3].astype(np.uint8)
            region4k_hm = cv2.addWeighted(color_block4k, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
            Image.fromarray(region4k_hm).save(os.path.join(output_dir, '%s_4k[%s].png' % (fname, j)))
        
    for j in range(6):
        score4k_1 = concat_scores4k(a4k_1[j], size=(s,)*2)
        score4k_2 = concat_scores4k(a4k_2[j], size=(s,)*2)
        score4k_3 = concat_scores4k(a4k_3[j], size=(s,)*2)
        score4k_4 = concat_scores4k(a4k_4[j], size=(s,)*2)

        new_score4k_2 = np.zeros_like(score4k_2)
        new_score4k_2[offset_2:s, offset_2:s] = score4k_2[:(s-offset_2), :(s-offset_2)]
        new_score4k_3 = np.zeros_like(score4k_3)
        new_score4k_3[offset_3:s, offset_3:s] = score4k_3[:(s-offset_3), :(s-offset_3)]
        new_score4k_4 = np.zeros_like(score4k_4)
        new_score4k_4[offset_4:s, offset_4:s] = score4k_4[:(s-offset_4), :(s-offset_4)]

        overlay4k = np.ones_like(score4k_2)*100
        overlay4k[offset_2:s, offset_2:s] += 100
        overlay4k[offset_3:s, offset_3:s] += 100
        overlay4k[offset_4:s, offset_4:s] += 100
        score4k = (score4k_1+new_score4k_2+new_score4k_3+new_score4k_4)/overlay4k

        color_block4k = (cmap(score4k)*255)[:,:,:3].astype(np.uint8)
        region4k_hm = cv2.addWeighted(color_block4k, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
        
        Image.fromarray(region4k_hm).save(os.path.join(output_dir, '%s_1024[%s].png' % (fname, j)))
        
    for i in range(6):
        score256_1 = concat_scores256(a256_1[:,i,:,:], size=(s//16,)*2)
        score256_2 = concat_scores256(a256_2[:,i,:,:], size=(s//16,)*2)
        new_score256_2 = np.zeros_like(score256_2)
        new_score256_2[offset_2:s, offset_2:s] = score256_2[:(s-offset_2), :(s-offset_2)]
        overlay256 = np.ones_like(score256_2)*100
        overlay256[offset_2:s, offset_2:s] += 100
        score256 = (score256_1+new_score256_2)/overlay256
        color_block256 = (cmap(score256)*255)[:,:,:3].astype(np.uint8)
        region256_hm = cv2.addWeighted(color_block256, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
        Image.fromarray(region256_hm).save(os.path.join(output_dir, '%s_256[%s].png' % (fname, i)))
    
    for j in range(6):
        score4k_1 = concat_scores4k(a4k_1[j], size=(s,)*2)
        score4k_2 = concat_scores4k(a4k_2[j], size=(s,)*2)
        score4k_3 = concat_scores4k(a4k_3[j], size=(s,)*2)
        score4k_4 = concat_scores4k(a4k_4[j], size=(s,)*2)

        new_score4k_2 = np.zeros_like(score4k_2)
        new_score4k_2[offset_2:s, offset_2:s] = score4k_2[:(s-offset_2), :(s-offset_2)]
        new_score4k_3 = np.zeros_like(score4k_3)
        new_score4k_3[offset_3:s, offset_3:s] = score4k_3[:(s-offset_3), :(s-offset_3)]
        new_score4k_4 = np.zeros_like(score4k_4)
        new_score4k_4[offset_4:s, offset_4:s] = score4k_4[:(s-offset_4), :(s-offset_4)]

        overlay4k = np.ones_like(score4k_2)*100
        overlay4k[offset_2:s, offset_2:s] += 100
        overlay4k[offset_3:s, offset_3:s] += 100
        overlay4k[offset_4:s, offset_4:s] += 100
        score4k = (score4k_1+new_score4k_2+new_score4k_3+new_score4k_4)/overlay4k

        for i in range(6):
            score256_1 = concat_scores256(a256_1[:,i,:,:], size=(s//16,)*2)
            score256_2 = concat_scores256(a256_2[:,i,:,:], size=(s//16,)*2)
            new_score256_2 = np.zeros_like(score256_2)
            new_score256_2[offset_2:s, offset_2:s] = score256_2[:(s-offset_2), :(s-offset_2)]
            overlay256 = np.ones_like(score256_2)*100*2
            overlay256[offset_2:s, offset_2:s] += 100*2
            score256 = (score256_1+new_score256_2)*2/overlay256

            factorize = lambda data: (data - np.min(data)) / (np.max(data) - np.min(data))
            score = (score4k*overlay4k+score256*overlay256)/(overlay4k+overlay256) #factorize(score256*score4k)
            color_block = (cmap(score)*255)[:,:,:3].astype(np.uint8)
            region_hm = cv2.addWeighted(color_block, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
            
            Image.fromarray(region_hm).save(os.path.join(output_dir, '%s_factorized_4k[%s]_256[%s].png' % (fname, j, i)))
            
    return


def create_hierarchical_heatmaps_concat(region, model256, model4k, output_dir, fname,
                             offset=128, scale=4, alpha=0.5, cmap = plt.get_cmap('coolwarm')):
    r"""
    Creates hierarchical heatmaps (With Raw H&E + ViT-256 + ViT-4K + Blended Heatmaps concatenated for easy comparison)
    
    Args:
    - region (PIL.Image):       4096 x 4096 Image 
    - model256 (torch.nn):      256-Level ViT 
    - model4k (torch.nn):       4096-Level ViT 
    - output_dir (str):         Save directory / subdirectory
    - fname (str):              Naming structure of files
    - offset (int):             How much to offset (from top-left corner with zero-padding) the region by for blending 
    - scale (int):              How much to scale the output image by 
    - alpha (float):            Image blending factor for cv2.addWeighted
    - cmap (matplotlib.pyplot): Colormap for creating heatmaps
    
    Returns:
    - None
    """
    
    region2 = add_margin(region.crop((128,128,4096,4096)), 
                     top=0, left=0, bottom=128, right=128, color=(255,255,255))
    region3 = add_margin(region.crop((128*2,128*2,4096,4096)), 
                     top=0, left=0, bottom=128*2, right=128*2, color=(255,255,255))
    region4 = add_margin(region.crop((128*3,128*3,4096,4096)), 
                     top=0, left=0, bottom=128*4, right=128*4, color=(255,255,255))
    
    b256_1, a256_1, a4k_1 = get_region_attention_scores(region, model256, model4k, scale)
    
    b256_2, a256_2, a4k_2 = get_region_attention_scores(region2, model256, model4k, scale)
    b256_3, a256_3, a4k_3 = get_region_attention_scores(region3, model256, model4k, scale)
    b256_4, a256_4, a4k_4 = get_region_attention_scores(region4, model256, model4k, scale)
    offset_2 = (offset*1)//scale
    offset_3 = (offset*2)//scale
    offset_4 = (offset*3)//scale
    s = 4096//scale
    save_region = np.array(region.resize((s, s)))

    for j in range(6):
        score4k_1 = concat_scores4k(a4k_1[j], size=(s,)*2)
        score4k_2 = concat_scores4k(a4k_2[j], size=(s,)*2)
        score4k_3 = concat_scores4k(a4k_3[j], size=(s,)*2)
        score4k_4 = concat_scores4k(a4k_4[j], size=(s,)*2)

        new_score4k_2 = np.zeros_like(score4k_2)
        new_score4k_2[offset_2:s, offset_2:s] = score4k_2[:(s-offset_2), :(s-offset_2)]
        new_score4k_3 = np.zeros_like(score4k_3)
        new_score4k_3[offset_3:s, offset_3:s] = score4k_3[:(s-offset_3), :(s-offset_3)]
        new_score4k_4 = np.zeros_like(score4k_4)
        new_score4k_4[offset_4:s, offset_4:s] = score4k_4[:(s-offset_4), :(s-offset_4)]

        overlay4k = np.ones_like(score4k_2)*100
        overlay4k[offset_2:s, offset_2:s] += 100
        overlay4k[offset_3:s, offset_3:s] += 100
        overlay4k[offset_4:s, offset_4:s] += 100
        score4k = (score4k_1+new_score4k_2+new_score4k_3+new_score4k_4)/overlay4k
        
        color_block4k = (cmap(score4k_1/100)*255)[:,:,:3].astype(np.uint8)
        region4k_hm = cv2.addWeighted(color_block4k, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
        
        for i in range(6):
            score256_1 = concat_scores256(a256_1[:,i,:,:], size=(s//16,)*2)
            score256_2 = concat_scores256(a256_2[:,i,:,:], size=(s//16,)*2)
            new_score256_2 = np.zeros_like(score256_2)
            new_score256_2[offset_2:s, offset_2:s] = score256_2[:(s-offset_2), :(s-offset_2)]
            overlay256 = np.ones_like(score256_2)*100*2
            overlay256[offset_2:s, offset_2:s] += 100*2
            score256 = (score256_1+new_score256_2)*2/overlay256
            
            color_block256 = (cmap(score256)*255)[:,:,:3].astype(np.uint8)
            region256_hm = cv2.addWeighted(color_block256, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
        
            factorize = lambda data: (data - np.min(data)) / (np.max(data) - np.min(data))
            score = (score4k*overlay4k+score256*overlay256)/(overlay4k+overlay256) #factorize(score256*score4k)
            color_block = (cmap(score)*255)[:,:,:3].astype(np.uint8)
            region_hm = cv2.addWeighted(color_block, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
            
            pad = 100
            canvas = Image.new('RGB', (s*2+pad,)*2, (255,)*3)
            draw = ImageDraw.Draw(canvas)
            font = ImageFont.truetype("arial.ttf", 50)
            draw.text((1024*0.5-pad*2, pad//4), "ViT-256 (Head: %d)" % i, (0, 0, 0), font=font)
            canvas = canvas.rotate(90)
            draw = ImageDraw.Draw(canvas)
            draw.text((1024*1.5-pad, pad//4), "ViT-4K (Head: %d)" % j, (0, 0, 0), font=font)
            canvas.paste(Image.fromarray(save_region), (pad,pad))
            canvas.paste(Image.fromarray(region4k_hm), (1024+pad,pad))
            canvas.paste(Image.fromarray(region256_hm), (pad,1024+pad))
            canvas.paste(Image.fromarray(region_hm), (s+pad,s+pad))
            canvas.save(os.path.join(output_dir, '%s_4k[%s]_256[%s].png' % (fname, j, i)))

    return


def create_hierarchical_heatmaps_concat_select(region, model256, model4k, output_dir, fname,
                             offset=128, scale=4, alpha=0.5, cmap = plt.get_cmap('coolwarm')):
    r"""
    Creates hierarchical heatmaps (With Raw H&E + ViT-256 + ViT-4K + Blended Heatmaps concatenated for easy comparison)

    Note that only select attention heads are used.
    
    Args:
    - region (PIL.Image):       4096 x 4096 Image 
    - model256 (torch.nn):      256-Level ViT 
    - model4k (torch.nn):       4096-Level ViT 
    - output_dir (str):         Save directory / subdirectory
    - fname (str):              Naming structure of files
    - offset (int):             How much to offset (from top-left corner with zero-padding) the region by for blending 
    - scale (int):              How much to scale the output image by 
    - alpha (float):            Image blending factor for cv2.addWeighted
    - cmap (matplotlib.pyplot): Colormap for creating heatmaps
    
    Returns:
    - None
    """
    
    region2 = add_margin(region.crop((128,128,4096,4096)), 
                     top=0, left=0, bottom=128, right=128, color=(255,255,255))
    region3 = add_margin(region.crop((128*2,128*2,4096,4096)), 
                     top=0, left=0, bottom=128*2, right=128*2, color=(255,255,255))
    region4 = add_margin(region.crop((128*3,128*3,4096,4096)), 
                     top=0, left=0, bottom=128*4, right=128*4, color=(255,255,255))
    
    b256_1, a256_1, a4k_1 = get_region_attention_scores(region, model256, model4k, scale)
    
    b256_2, a256_2, a4k_2 = get_region_attention_scores(region2, model256, model4k, scale)
    b256_3, a256_3, a4k_3 = get_region_attention_scores(region3, model256, model4k, scale)
    b256_4, a256_4, a4k_4 = get_region_attention_scores(region4, model256, model4k, scale)
    offset_2 = (offset*1)//scale
    offset_3 = (offset*2)//scale
    offset_4 = (offset*3)//scale
    s = 4096//scale
    save_region = np.array(region.resize((s, s)))
    
    canvas = [[Image.fromarray(save_region), None, None], [None, None, None]]
    for idx_4k, j in enumerate([0,5]):
        score4k_1 = concat_scores4k(a4k_1[j], size=(s,)*2)
        score4k_2 = concat_scores4k(a4k_2[j], size=(s,)*2)
        score4k_3 = concat_scores4k(a4k_3[j], size=(s,)*2)
        score4k_4 = concat_scores4k(a4k_4[j], size=(s,)*2)

        new_score4k_2 = np.zeros_like(score4k_2)
        new_score4k_2[offset_2:s, offset_2:s] = score4k_2[:(s-offset_2), :(s-offset_2)]
        new_score4k_3 = np.zeros_like(score4k_3)
        new_score4k_3[offset_3:s, offset_3:s] = score4k_3[:(s-offset_3), :(s-offset_3)]
        new_score4k_4 = np.zeros_like(score4k_4)
        new_score4k_4[offset_4:s, offset_4:s] = score4k_4[:(s-offset_4), :(s-offset_4)]

        overlay4k = np.ones_like(score4k_2)*100
        overlay4k[offset_2:s, offset_2:s] += 100
        overlay4k[offset_3:s, offset_3:s] += 100
        overlay4k[offset_4:s, offset_4:s] += 100
        score4k = (score4k_1+new_score4k_2+new_score4k_3+new_score4k_4)/overlay4k
        
        color_block4k = (cmap(score4k_1/100)*255)[:,:,:3].astype(np.uint8)
        region4k_hm = cv2.addWeighted(color_block4k, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
        canvas[0][idx_4k+1] = Image.fromarray(region4k_hm)
        
        for idx_256, i in enumerate([2]):
            score256_1 = concat_scores256(a256_1[:,i,:,:], size=(s//16,)*2)
            score256_2 = concat_scores256(a256_2[:,i,:,:], size=(s//16,)*2)
            new_score256_2 = np.zeros_like(score256_2)
            new_score256_2[offset_2:s, offset_2:s] = score256_2[:(s-offset_2), :(s-offset_2)]
            overlay256 = np.ones_like(score256_2)*100*2
            overlay256[offset_2:s, offset_2:s] += 100*2
            score256 = (score256_1+new_score256_2)*2/overlay256
            
            color_block256 = (cmap(score256)*255)[:,:,:3].astype(np.uint8)
            region256_hm = cv2.addWeighted(color_block256, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
            canvas[idx_256+1][0] = Image.fromarray(region256_hm)
            
            factorize = lambda data: (data - np.min(data)) / (np.max(data) - np.min(data))
            score = (score4k*overlay4k+score256*overlay256)/(overlay4k+overlay256) #factorize(score256*score4k)
            color_block = (cmap(score)*255)[:,:,:3].astype(np.uint8)
            region_hm = cv2.addWeighted(color_block, alpha, save_region.copy(), 1-alpha, 0, save_region.copy())
            canvas[idx_256+1][idx_4k+1] = Image.fromarray(region_hm)
            
    canvas = getConcatImage([getConcatImage(row) for row in canvas], how='vertical')
    canvas.save(os.path.join(output_dir, '%s_heatmap.png' % (fname)))
    return