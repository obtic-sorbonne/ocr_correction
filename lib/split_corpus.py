#####################################################################################################
# Script pour diviser le corpus et et actualiser les métadonnées en ajoutant les types de split (train, test, dev)
# On se trouve à /Dataset/ repertoire
# On a déjà un fichier de métadonnées de ICDAR2017 qui a été filtré sur les dates (1800-1900)
#####################################################################################################

import os
import shutil # pour les opérations sur les fichiers
import pandas as pd
import random

directory = './ICDAR2017'
output_csv = './full_metadata.csv'

train_dir = './train'  
dev_dir = './dev'       
test_dir = './test'     

# If not created
os.makedirs(train_dir, exist_ok=True)
os.makedirs(dev_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

metadata_df = pd.read_csv(output_csv, sep=';')

# Getting names of all files in corpus
files = [f for f in os.listdir(directory) if f.endswith('.txt')]
random.shuffle(files)  # randomize the selection

# On divise avec les pour-cents
train_ratio = 0.8
dev_ratio = 0.1
test_ratio = 0.1

#Calcules
num_files = len(files)
num_train = int(num_files * train_ratio)
num_dev = int(num_files * dev_ratio)
num_test = num_files - num_train - num_dev

# Splitting tout
train_files = files[:num_train]
dev_files = files[num_train:num_train + num_dev]
test_files = files[num_train + num_dev:]

# Update the "Split" column in pd
def update_metadata_split(file_list, split_type):
    for file in file_list:
        metadata_df.loc[metadata_df['File'] == file, 'Split'] = split_type
        if split_type == 'train':
            shutil.copy(os.path.join(directory, file), os.path.join(train_dir, file))
        elif split_type == 'dev':
            shutil.copy(os.path.join(directory, file), os.path.join(dev_dir, file))
        elif split_type == 'test':
            shutil.copy(os.path.join(directory, file), os.path.join(test_dir, file))

update_metadata_split(train_files, 'train')
update_metadata_split(dev_files, 'dev')
update_metadata_split(test_files, 'test')

metadata_df.to_csv(output_csv, sep=';', index=False)

print(f"Metadata CSV updated: {output_csv}")
print(f"Files copied to:\nTrain: {train_dir}\nDev: {dev_dir}\nTest: {test_dir}")
