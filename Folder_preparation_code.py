# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 23:41:09 2020

@author: jeswanth.gutti
"""

import pandas as pd
import os
from PIL import Image
from shutil import copyfile

#file expects Faces folder should exist
# Folder name 'Age_folder' should exist with empty subfolders '0', '1'
# Folder name 'Gender_folder' should exist with empty subfolders '0', '1', '2', '3', '4', '5', '6', '7'

folder_path = 'faces/'

def encode_age_tuple(df):
    df = df.reset_index(drop=True)
    age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
    
    df['Age_Label_Prsence'] = df['age'].apply(lambda x: x in age_list)
    df = df[df['Age_Label_Prsence']]
    
    df['Age_Index'] = df['age'].apply(lambda x: age_list.index(x))
    
    df = df.reset_index(drop=True)
    
    return df


def get_total_df(no_of_folds):
    All_dfs=[]
    total_files_list = ['fold_'+str(idX_)+'_data.txt'  for idX_ in range(no_of_folds)]
    
    for file_name in total_files_list:
        fold = total_files_list.index(file_name)+1
        temp_df = pd.read_csv(file_name, sep="\t")
        temp_df['Fold_Index'] = fold
        All_dfs.append(temp_df)
    Total_images_details_df = pd.concat(All_dfs)
    
    return Total_images_details_df

def encode_gender_tuple(df):
    df = df.reset_index(drop=True)
    gender = ['f', 'm']
    
    df['gender_Label_Prsence'] = df['gender'].apply(lambda x: x in gender)
    df = df[df['gender_Label_Prsence']]
    
    df['Gender_Index'] = df['gender'].apply(lambda x: gender.index(x))
    
    df = df.reset_index(drop=True)
    
    return df


def move_files_age(df):
    for x in range(df.shape[0]):
        row = df.loc[x]
        
        copyfile(folder_path+row['user_id']+'/'+'coarse_tilt_aligned_face.'+str(row['face_id'])+'.'+row['original_image'], 'Age_folder/'+str(row['Gender_Index'])+'/coarse_tilt_aligned_face.'+str(row['face_id'])+'.'+row['original_image'])
        
def move_files_gender(df):
    for x in range(df.shape[0]):
        row = df.loc[x]
        
        copyfile(folder_path+row['user_id']+'/'+'coarse_tilt_aligned_face.'+str(row['face_id'])+'.'+row['original_image'], 'Gender_folder/'+str(row['Gender_Index'])+'/coarse_tilt_aligned_face.'+str(row['face_id'])+'.'+row['original_image'])
        

if __name__="__main__":
    no_of_folds = 5
    Total_images_details_df = get_total_df(no_of_folds)
    Total_images_details_df = Total_images_details_df.reset_index(drop=True)
    Total_images_details_df = encode_age_tuple(Total_images_details_df)
    Total_images_details_df = Total_images_details_df.reset_index(drop=True)
    Total_images_details_df = encode_gender_tuple(Total_images_details_df)
    move_files_age(Total_images_details_df)
    move_files_gender(Total_images_details_df)
