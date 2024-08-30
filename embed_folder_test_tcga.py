### Dependencies
import argparse
import colorsys
from io import BytesIO
import os
import random
import requests
import sys
# import Threading 
import concurrent.futures

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
# sys.path.append('projects/def-senger/multimodality/1-Hierarchical-Pretraining')
# projects/def-senger/multimodality/1-Hierarchical-Pretraining
import vision_transformer as vits
import vision_transformer4k as vits4k
import openslide
from tqdm import tqdm
from hipt_model_utils import get_vit256, get_vit4k, eval_transforms
import time
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.tools.tissuemask import TissueMasker
from tiatoolbox.tools.tissuemask import OtsuTissueMasker
from os.path import join
from pathlib import Path
import argparse
from typing import List
from hipt_4k import HIPT_4K
from hipt_model_utils import get_vit256, get_vit4k, eval_transforms
from svsReader import readSVS
#from hipt_heatmap_utils import *
#from attention_visualization_utils import *
from attention_juan import *




def embed_folder(main_folder: str, output_folder: str,model) -> None:
    """
    Embed all .svs files in subfolders of main_folder and save the embeddings as .pt files in corresponding subfolders
    of output_folder.
    """
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Walk through main_folder and get all subfolders
    for root, dirs, files in os.walk(main_folder):
        for dir_name in dirs:
            # Create subfolder in output folder with same name
            output_dir = join(output_folder, dir_name)
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Get all .svs files in subfolder
            svs_files = [f for f in os.listdir(join(root, dir_name)) if f.endswith('.svs')]

            # Embed each .svs file and save as .pt file in output folder
            for svs_file in svs_files:
                svs_file_path = join(root, dir_name, svs_file)
                np_file_path = join(output_dir, os.path.splitext(svs_file)[0] + '.npy')
                if not os.path.isfile(np_file_path):
                    # Embed svs_file if corresponding pt_file doesn't exist yet
                    embedding = embed_svs_v2(svs_file_path, np_file_path, model)
                    embed_array = embedding.cpu().numpy()
                    np.save(np_file_path, embed_array)
                    #torch.save(embedding, pt_file_path)

def embed_single_ss_folder(main_folder: str, output_folder: str, model, multithread: bool) -> None:

    """
    Embed all .svs files in a single ss folder of a box parent folder
    such that each subfolder can be submitted as a single job 

    """
    svs_files = [os.path.join(main_folder, file) for file in os.listdir(main_folder) if os.path.isfile(os.path.join(main_folder, file)) if file.endswith('.svs')]
    print(svs_files)
    # obtain all svs files in the subfolder into a list ADD IF IT ENDS WITH .SVS
    for root, dirs, files in tqdm(os.walk(main_folder)):
        for file in files:
            if file.endswith('.svs'):
                svs_filepath = os.path.join(root,file) # full svs file path
                print(svs_filepath)
                svs_basename = os.path.splitext(os.path.basename(svs_filepath))[0] # svs file basename like 1012096 from 1012096.svs
                ss_folder_basename = os.path.basename(root) # ss folder basename like SS-16-03048, used for output_dir 
                output_dir = join(output_folder,ss_folder_basename)
                Path(output_dir).mkdir(parents=True, exist_ok=True) # create output folder of current SS folder if it doesn't already exist 
                # If no multithreading of this files for-loop is enabled in the argument, 
                if multithread:
                    start_time = time.time()
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future_to_svs_file = {executor.submit(embed_svs_v2, os.path.join(root,file), 
                                                            join(output_dir, svs_basename + '.npy'), model): svs_file 
                                            for svs_file in svs_files}
                        for future in concurrent.futures.as_completed(future_to_svs_file):
                            svs_file = future_to_svs_file[future]
                            try:
                                embedding = future.result()
                                embed_array = embedding.cpu().numpy()
                                np.save(join(output_dir, svs_basename + '.npy'), embed_array)
                            except Exception as exc:
                                print(f'Error processing {svs_file}: {exc}')
                    end_time = time.time()
                    print(f'Time taken: {end_time - start_time} seconds')

                # IF multithreading processing svs files within the ss folder 
                else:
                    np_filepath = join(output_dir, svs_basename + '.npy')
                    print(np_filepath)
                    if not os.path.isfile(np_filepath):
                        # Embed svs_file if corresponding pt_file doesn't exist yet
                        embedding = embed_svs_v2(svs_filepath, np_filepath, model)
                        embed_array = embedding.cpu().numpy()
                        np.save(np_filepath, embed_array)


# NEED TO FIX THE OUTPUT FOLDER PATH, NOW WITHIN THE MULTITHREADING, IT DID NOT GO INTO THE SS-FOLDER INSIDE THE EMBEDDINGS FOLDER OUTPUTS 
# BUT OUTSIDE 
# THEN SUBMIT MULTIPLE JOBS ON DIFFERENT NODES GNU PARALLEL 

# PROBLEM: DID NOT CREATE OUTPUT FOLDER output_dir in embed_snigle_ss_folder_v2 such that the .npy output files are stored in just current dir 

def embed_single_ss_folder_v2(main_folder: str, output_folder: str, model, multithread: bool) -> None:
    svs_files = [os.path.join(main_folder, file) for file in os.listdir(main_folder) if os.path.isfile(os.path.join(main_folder, file)) if file.endswith('.svs')]

    if multithread:
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_svs_file = {executor.submit(embed_svs_v2, svs_filepath, join(output_folder, os.path.splitext(os.path.basename(svs_filepath))[0] + '.npy'), model): svs_filepath
                            for svs_filepath in svs_files}
            for future in concurrent.futures.as_completed(future_to_svs_file):
                svs_filepath = future_to_svs_file[future]
                try:
                    embedding = future.result()
                    embed_array = embedding.cpu().numpy()
                    np.save(join(output_folder, os.path.splitext(os.path.basename(svs_filepath))[0] + '.npy'), embed_array)
                except Exception as exc:
                    print(f'Error processing {svs_filepath}: {exc}')
        end_time = time.time()
        print(f'Time taken: {end_time - start_time} seconds')
    else:
        # iterate over all the files in the main_folder directory
        for root, dirs, files in tqdm(os.walk(main_folder)):
            for file in files:
                if file.endswith('.svs'):
                    svs_filepath = os.path.join(root,file) # full svs file path
                    svs_basename = os.path.splitext(os.path.basename(svs_filepath))[0] # svs file basename like 1012096 from 1012096.svs
                    ss_folder_basename = os.path.basename(root) # ss folder basename like SS-16-03048, used for output_dir 
                    output_dir = join(output_folder,ss_folder_basename)
                    Path(output_dir).mkdir(parents=True, exist_ok=True) # create output folder of current SS folder if it doesn't already exist 
                    np_filepath = join(output_dir, svs_basename + '.npy')
                    if not os.path.isfile(np_filepath):
                        # Embed svs_file if corresponding pt_file doesn't exist yet
                        embedding = embed_svs_v2(svs_filepath, np_filepath, model)
                        embed_array = embedding.cpu().numpy()
                        np.save(np_filepath, embed_array)

def embed_single_ss_folder_v3(main_folder: str, output_folder: str, model, multithread: bool) -> None:
    svs_files = [os.path.join(main_folder, file) for file in os.listdir(main_folder) if os.path.isfile(os.path.join(main_folder, file)) if file.endswith('.svs')]
    
    if multithread:
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_svs_file = {}
            for svs_filepath in svs_files:
                svs_basename = os.path.splitext(os.path.basename(svs_filepath))[0]
                ss_folder_basename = os.path.basename(os.path.dirname(svs_filepath))
                output_dir = join(output_folder, ss_folder_basename)
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                future = executor.submit(embed_svs_v2, svs_filepath, model)
                future_to_svs_file[future] = (svs_filepath, svs_basename, output_dir)
                # A dictionary called future_to_svs_file is created to map each future (i.e., each submitted task) to the corresponding svs file path, svs file name, and output directory.


            for future in concurrent.futures.as_completed(future_to_svs_file):
                svs_filepath, svs_basename, output_dir = future_to_svs_file[future]
                try:
                    embedding = future.result()
                    embed_array = embedding.cpu().numpy()
                    np_filepath = join(output_dir, svs_basename + '.npy')
                    np.save(np_filepath, embed_array)
                except Exception as exc:
                    print(f'Error processing {svs_filepath}: {exc}')
        end_time = time.time()
        print(f'Time taken: {end_time - start_time} seconds')

    else:
        # iterate over all the files in the main_folder directory
        for root, dirs, files in tqdm(os.walk(main_folder)):
            for file in files:
                if file.endswith('.svs'):
                    svs_filepath = os.path.join(root,file) # full svs file path
                    svs_basename = os.path.splitext(os.path.basename(svs_filepath))[0] # svs file basename like 1012096 from 1012096.svs
                    ss_folder_basename = os.path.basename(root) # ss folder basename like SS-16-03048, used for output_dir 
                    output_dir = join(output_folder,ss_folder_basename)
                    Path(output_dir).mkdir(parents=True, exist_ok=True) # create output folder of current SS folder if it doesn't already exist 
                    np_filepath = join(output_dir, svs_basename + '.npy')
                    if not os.path.isfile(np_filepath):
                        # Embed svs_file if corresponding pt_file doesn't exist yet
                        embedding = embed_svs_v2(svs_filepath, np_filepath, model)
                        embed_array = embedding.cpu().numpy()
                        np.save(np_filepath, embed_array)

def embed_single_ss_folder_v4(main_folder: str, output_folder: str, model, multithread: bool) -> None:
    svs_files = [os.path.join(main_folder, file) for file in os.listdir(main_folder) if os.path.isfile(os.path.join(main_folder, file)) if file.endswith('.svs')]

    for svs_filepath in svs_files:
        svs_basename = os.path.splitext(os.path.basename(svs_filepath))[0]
        ss_folder_basename = os.path.basename(os.path.dirname(svs_filepath))
        output_dir = join(output_folder, ss_folder_basename)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    if multithread:
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_svs_file = {}
            for svs_filepath in svs_files:
                svs_basename = os.path.splitext(os.path.basename(svs_filepath))[0]
                ss_folder_basename = os.path.basename(os.path.dirname(svs_filepath))
                output_dir = join(output_folder, ss_folder_basename)
                np_filepath = join(output_dir, svs_basename + '.npy')
                future = executor.submit(embed_svs_v2, svs_filepath, np_filepath, model)
                future_to_svs_file[future] = svs_filepath

            for future in concurrent.futures.as_completed(future_to_svs_file):
                svs_filepath = future_to_svs_file[future]
                try:
                    embedding = future.result()
                    embed_array = embedding.cpu().numpy()
                    np_filepath = join(output_folder, os.path.splitext(os.path.basename(svs_filepath))[0] + '.npy')
                    np.save(np_filepath, embed_array)
                except Exception as exc:
                    print(f'Error processing {svs_filepath}: {exc}')
        end_time = time.time()
        print(f'Time taken: {end_time - start_time} seconds')
    else:
        for svs_filepath in svs_files:
            svs_basename = os.path.splitext(os.path.basename(svs_filepath))[0]
            ss_folder_basename = os.path.basename(os.path.dirname(svs_filepath))
            output_dir = join(output_folder, ss_folder_basename)
            np_filepath = join(output_dir, svs_basename + '.npy')
            embedding = embed_svs_v2(svs_filepath, np_filepath, model)
            embed_array = embedding.cpu().numpy()
            np.save(np_filepath, embed_array)


def embed_single_ss_folder_v5(main_folder: str, output_folder: str, model, multithread: bool) -> None:
    svs_files = [os.path.join(main_folder, file) for file in os.listdir(main_folder) if os.path.isfile(os.path.join(main_folder, file)) if file.endswith('.svs')]

    if multithread:
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_svs_file = {}
            for svs_filepath in svs_files:
                svs_basename = os.path.splitext(os.path.basename(svs_filepath))[0]
                ss_folder_basename = os.path.basename(os.path.dirname(svs_filepath))
                output_dir = join(output_folder, ss_folder_basename)
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                future = executor.submit(embed_svs_v2, svs_filepath, join(output_dir, os.path.splitext(os.path.basename(svs_filepath))[0] + '.npy'), model)
                future_to_svs_file[future] = svs_filepath
            for future in concurrent.futures.as_completed(future_to_svs_file):
                svs_filepath = future_to_svs_file[future]
                try:
                    embedding = future.result()
                    embed_array = embedding.cpu().numpy()
                    np.save(join(output_dir, os.path.splitext(os.path.basename(svs_filepath))[0] + '.npy'), embed_array)
                except Exception as exc:
                    print(f'Error processing {svs_filepath}: {exc}')
        end_time = time.time()
        print(f'Time taken: {end_time - start_time} seconds')
    else:
        for svs_filepath in svs_files:
            print(svs_filepath)
            svs_basename = os.path.splitext(os.path.basename(svs_filepath))[0]
            ss_folder_basename = os.path.basename(os.path.dirname(svs_filepath))
            output_dir = join(output_folder, ss_folder_basename)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            embed_svs_v2(svs_filepath, join(output_dir, os.path.splitext(os.path.basename(svs_filepath))[0] + '.npy'), model)


def embed_tcga_list(svs_list_file: str, model, multithread: bool) -> None:
    num_workers=32
    
    base_path = "/home/sorkwos/scratch/TCGA-HNSC"
    with open(svs_list_file, 'r') as file:
        svs_files = [os.path.join(base_path, line.strip()) for line in file.readlines()]
        print(svs_files)


    if multithread:
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_svs_file = {}
            for svs_filepath in svs_files:
                output_dir = os.path.dirname(svs_filepath)
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                future = executor.submit(embed_svs_v2, svs_filepath, join(output_dir, os.path.splitext(os.path.basename(svs_filepath))[0] + '.pt'), model)
                future_to_svs_file[future] = svs_filepath
            for future in concurrent.futures.as_completed(future_to_svs_file):
                svs_filepath = future_to_svs_file[future]
                try:
                    embedding = future.result()
                    if  isinstance(embedding, int) and embedding == 0:
                        print(f'Skipping {svs_filepath} due to missing properties.')
                        continue
                    # No need to convert to numpy and save again, as it's already saved in embed_svs_v2
                    #embed_array = embedding.cpu().numpy()
                    #np.save(join(output_dir, os.path.splitext(os.path.basename(svs_filepath))[0] + '.npy'), embed_array)
                except Exception as exc:
                    print(f'Error processing {svs_filepath}: {exc}')
        end_time = time.time()
        print(f'Time taken: {end_time - start_time} seconds')
    else:
        for svs_filepath in svs_files:
            print(svs_filepath)
            output_dir = os.path.dirname(svs_filepath)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            embedding = embed_svs_v2(svs_filepath, join(output_dir, os.path.splitext(os.path.basename(svs_filepath))[0] + '.pt'), model)
            if  isinstance(embedding, int) and embedding == 0:
                print(f'Skipping {svs_filepath} due to missing properties.')
                continue
def embed_folder_multithread(main_folder: str, output_folder: str, model) -> None:
    """
    Embed all .svs files in subfolders of main_folder and save the embeddings as .npy files in corresponding subfolders
    of output_folder.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Walk through main_folder and get all subfolders
    for root, dirs, files in os.walk(main_folder):
        for dir_name in dirs:
            # Create subfolder in output folder with same name
            output_dir = os.path.join(output_folder, dir_name)
            os.makedirs(output_dir, exist_ok=True)

            # Get all .svs files in subfolder
            svs_files = [f for f in os.listdir(os.path.join(root, dir_name)) if f.endswith('.svs')]

            # Embed each .svs file and save as .npy file in output folder
            with concurrent.futures.ThreadPoolExecutor() as executor:
            # with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # if I wanted to create 4 threads 
                futures = []
                for svs_file in svs_files:
                    svs_file_path = os.path.join(root, dir_name, svs_file)
                    np_file_path = os.path.join(output_dir, os.path.splitext(svs_file)[0] + '.npy')
                    if not os.path.isfile(np_file_path):
                        # Embed svs_file if corresponding np_file doesn't exist yet
                        futures.append(executor.submit(embed_svs_v2, svs_file_path, np_file_path, model))
                for future in futures:
                    # Wait for each embedding process to finish
                    future.result()

def main():
    parser = argparse.ArgumentParser(description='Embed all .svs files in a folder and save embeddings as .pt files.')
    parser.add_argument('main_folder', type=str, help='path to main folder')
    #parser.add_argument('output_folder', type=str, help='path to output folder')
    parser.add_argument('--multithread', dest='multithread', action='store_true', help='enable multithreading')
    args = parser.parse_args()

    # Define your model here
    light_jet = cmap_map(lambda x: x/2 + 0.5, matplotlib.cm.jet)
    #pretrained_weights256 = '../Checkpoints/vit256_small_dino.pth'
    #pretrained_weights4k = '../Checkpoints/vit4k_xs_dino.pth'
    pretrained_weights256 = '/Checkpoints/vit256_small_dino.pth'
    pretrained_weights4k = '/Checkpoints/vit4k_xs_dino.pth'

        # Check for available GPUs and assign them
    if torch.cuda.device_count() < 2:
        raise RuntimeError("This script requires at least 2 GPUs.")
    

    device256 = torch.device('cuda:0')
    device4k = torch.device('cuda:1')

    ### ViT_256 + ViT_4K loaded independently (used for Attention Heatmaps)
    model256 = get_vit256(pretrained_weights=pretrained_weights256, device=device256)
    model4k = get_vit4k(pretrained_weights=pretrained_weights4k, device=device4k)

    ### ViT_256 + ViT_4K loaded into HIPT_4K API
    model = HIPT_4K(pretrained_weights256, pretrained_weights4k, device256, device4k)
    model.eval()
    print(args.main_folder)
    embed_tcga_list(args.main_folder, model, args.multithread)

    # embed_single_ss_folder_v3(args.main_folder, args.output_folder, model, args.multithread)

if __name__ == '__main__':
    main()