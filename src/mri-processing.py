import nibabel as nib
import numpy as np
from PIL import Image

# Load the NIfTI file

#get all the files in the directory
import os
import glob

# Get all the .nii files in the directory
# t1_nii_files = glob.glob("/data/spine/gtu-lumbar/dataset/train/16bit")
# t1ce_nii_files = glob.glob("/data/brain/BraTS2020/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/*/*t1ce.nii")

t1_png_files = glob.glob("/data/spine/gtu-lumbar/dataset/train/16bit/*__T1_WEI_SAG.png")
t2_png_files = glob.glob("/data/spine/gtu-lumbar/dataset/train/16bit/*__T2_WEI_SAG.png")

# sort the files in alphabetical order
t1_png_files.sort()
t2_png_files.sort()

# # create the directory
# os.makedirs("/data/brain_big/BraTS2024/train/BraSyn/16bit/t1", exist_ok=True)
# os.makedirs("/data/brain_big/BraTS2024/train/BraSyn/16bit/t2", exist_ok=True)

#move the files

#split the files

ratio = 0.8

t1_png_files_train = t1_png_files[:int(ratio * len(t1_png_files))]
t2_png_files_train = t2_png_files[:int(ratio * len(t2_png_files))]
t1_png_files_val = t1_png_files[int(ratio * len(t1_png_files)):]
t2_png_files_val = t2_png_files[int(ratio * len(t2_png_files)):]

for i, (t1_png_file, t2_png_file) in enumerate(zip(t1_png_files_train, t2_png_files_train)):

    print(t1_png_file, t2_png_file)

    #copy the files
    os.system(f"cp {t1_png_file} /data/datasets/spine/gtu/train/t1/{i}.png")
    os.system(f"cp {t2_png_file} /data/datasets/spine/gtu/train/t2/{i}.png")

for i, (t1_png_file, t2_png_file) in enumerate(zip(t1_png_files_val, t2_png_files_val)):
    #copy the files
    os.system(f"cp {t1_png_file} /data/datasets/spine/gtu/test/t1/{i}.png")
    os.system(f"cp {t2_png_file} /data/datasets/spine/gtu/test/t2/{i}.png")
    

# save_path = "/data/brain/BraTS2020/test"

# # save_path = "/home/jaewan/spine-diff/"

# for i, (t1_nii_file, t1ce_nii_file) in enumerate(zip(t1_nii_files, t1ce_nii_files)):
#     # Load the NIfTI file
#     img = nib.load(t1_nii_file)
#     img2 = nib.load(t1ce_nii_file)

#     # Convert to a NumPy array
#     data = img.get_fdata().astype(np.uint16)
#     data2 = img2.get_fdata().astype(np.uint16)

#     img1 = data[:, :, data.shape[2] // 2]
#     img2 = data2[:, :, data2.shape[2] // 2]

#     print(img1.max(), img1.min())
#     print(img2.max(), img2.min())

#     # img1 = (img1 - img1.min()) / (img1.max() - img1.min())
#     # img2 = (img2 - img2.min()) / (img2.max() - img2.min())

#     # print(os.path.join(save_path, "t1", f"{i}.png"))

#     # exit()

#     Image.fromarray(img1).save(os.path.join(save_path, "t1", f"{i}.png"))
#     Image.fromarray(img2).save(os.path.join(save_path, "t1ce", f"{i}.png"))

# img = nib.load(nii_file)

# # Convert to a NumPy array
# data = img.get_fdata()

# # Display the shape of the data
# print(data.shape)
# print(data[:, :, data.shape[2] // 2].shape)

# # # Display a single slice (assuming axial slices)
# # plt.imshow(data[:, :, data.shape[2] // 2], cmap="gray")

# # #save the image
# # plt.savefig('output2.png')