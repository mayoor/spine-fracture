from pydicom import dcmread
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from functools import lru_cache
import os
import pandas as pd

def show_dcm_image(image, scale=False):
    ds = dcmread(image)
    return ds.pixel_array

def show_slices_for_patient(segmentation_info, plot_dim=(3,10), root_folder="train_images", **kwargs):
    fig, axs = plt.subplots(plot_dim[0], plot_dim[1], **kwargs)
    for r in range(plot_dim[0]):
        for c in range(plot_dim[1]):
            index = (r*plot_dim[1])+c
            if index == len(segmentation_info):
                return fig, axs
            record = segmentation_info.iloc[index]
            axs[r,c].imshow(show_dcm_image(f'{root_folder}/{record.StudyInstanceUID}/{record.slice_number}.dcm'))
            axs[r,c].add_patch(Rectangle((record.x,record.y),record.width,record.height,linewidth=1,edgecolor='r',facecolor='none'))
    return fig, axs

def show_dcm_image_with_bounding_box(image, x, y, width, height):
    arr = show_dcm_image(image)
    plt.imshow(arr, cmap="gray")
    plt.gca().add_patch(Rectangle((x,y),width,height,linewidth=1,edgecolor='r',facecolor='none'))
    return plt

def get_boundingbox_records_sample_by_fracture_count(df_train, numsample = 1, rand_state=42):
    np.random.seed(rand_state)
    num_fractures_with_bb = df_train[[f"C{i}" for i in range(1,8)]][(df_train["patient_overall"]==1) & (df_train["has_boundingbox"]==True)].sum(axis=1)
    fracture_count_patient_with_bb = {k:num_fractures_with_bb[num_fractures_with_bb == k].index.tolist() for k in range(1,7)}
    fracture_count_patient_sample_with_bb = {k: np.random.choice(fracture_count_patient_with_bb[k], numsample,) for k in fracture_count_patient_with_bb}
    return fracture_count_patient_sample_with_bb

@lru_cache(1)
def get_segmentation_info_ids(segmentation_dir="segmentations"):
    """
    Fetch StudyInstanceUID with segmentation_info
    """
    ids = []
    for file in os.listdir(segmentation_dir):
        ids.append(file[:-4]) # remove .nii from the file name to get the studyinstnaceuid
    return pd.DataFrame(data = ids, columns=["StudyInstanceUID"])

def get_sample_by_fracture_count(fracture_count_record, df_boundingbox, df_train, count=1):
    return df_boundingbox[df_boundingbox.StudyInstanceUID == df_train.iloc[fracture_count_record[count][0]].StudyInstanceUID]



def get_segmentation_samples(df_train, numsample = 1, rand_state=42):
    np.random.seed(rand_state)
    num_fractures_with_segmentation = df_train[[f"C{i}" for i in range(1,8)]][(df_train["patient_overall"]==1) & (df_train["has_boundingbox"]==True) & (df_train["has_segmentation"]==True)].sum(axis=1)
    fracture_count_patient_with_segmentation = {k:num_fractures_with_segmentation[num_fractures_with_segmentation == k].index.tolist() for k in num_fractures_with_segmentation.unique()}
    fracture_count_patient_sample_with_segmentation = {k: np.random.choice(fracture_count_patient_with_segmentation[k], numsample,) for k in fracture_count_patient_with_segmentation}
    return fracture_count_patient_sample_with_segmentation
