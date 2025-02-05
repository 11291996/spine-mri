import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Load the NIfTI file
nii_file = "/data/brain/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t1.nii"  # or .nii
img = nib.load(nii_file)

# Convert to a NumPy array
data = img.get_fdata()

# Display the shape of the data
print(data.shape)
print(data[:, :, data.shape[2] // 2].shape)

# # Display a single slice (assuming axial slices)
# plt.imshow(data[:, :, data.shape[2] // 2], cmap="gray")

# #save the image
# plt.savefig('output2.png')