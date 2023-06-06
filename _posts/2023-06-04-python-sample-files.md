---
layout: post
title: "Python sample files from folder"
date: 2023-06-04 20:00:00 +0800
tags: [python]
---

## for a CNN model doing binary classification task. I have one class in folder 1 and another class in folder 2. How to randomly sample equal number of data from both classes and mix them and save sampled files from folder1 and folder2 to directory sampled_folder1 and sampled_folder2

To copy the selected files into new directories, you can use the `shutil` module in Python. First, create the new directories if they don't exist already. Then, copy the selected files into these new directories.

Here's how you could modify the previous code to perform these steps:

```python
import os
import random
import shutil

# List files in each folder
folder1_files = os.listdir('folder1')
folder2_files = os.listdir('folder2')

# Make sure each filepath includes the folder name
folder1_files = ['folder1/' + file for file in folder1_files]
folder2_files = ['folder2/' + file for file in folder2_files]

# Sample from each class
min_samples = min(len(folder1_files), len(folder2_files))  # get the number of samples in the smaller class
folder1_samples = random.sample(folder1_files, min_samples)
folder2_samples = random.sample(folder2_files, min_samples)

# Create new directories
os.makedirs('sampled_folder1', exist_ok=True)
os.makedirs('sampled_folder2', exist_ok=True)

# Copy files to new directories
for file in folder1_samples:
    shutil.copy(file, 'sampled_folder1')

for file in folder2_samples:
    shutil.copy(file, 'sampled_folder2')
```

This script will copy the selected files from `folder1` and `folder2` to `sampled_folder1` and `sampled_folder2`, respectively. Then, you can proceed with train/test split as before, but using the new sampled folders. Note that `os.makedirs` with the `exist_ok=True` parameter will create the directory if it doesn't exist, but won't throw an error if the directory does already exist.

Be careful with file copying: this script will overwrite any files in the destination folders that have the same name as a file being copied. If this could be a problem, you may want to add some checks to this script to prevent overwriting.