# pip install pydicom
#%%
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ImplicitVRLittleEndian

# Create a new DICOM dataset
dataset = Dataset()

# Set DICOM data elements
dataset.PatientName = "John Doe"
dataset.PatientID = "12345"
dataset.Modality = "CT"
dataset.SeriesDescription = "CT Scan"
dataset.Rows = 512
dataset.Columns = 512
dataset.PixelSpacing = [0.976562, 0.976562]
dataset.BitsAllocated = 16
dataset.BitsStored = 16
dataset.HighBit = 15
dataset.PixelRepresentation = 0

# Create a simple 2D image using NumPy (512x512 with a circle)
image_size = (512, 512)
x, y = np.ogrid[:image_size[0], :image_size[1]]
circle_center = (image_size[0] // 2, image_size[1] // 2)
radius = 100
mask = (x - circle_center[0]) ** 2 + (y - circle_center[1]) ** 2 < radius ** 2
image_data = np.zeros(image_size, dtype=np.uint16)
image_data[mask] = 1000  # Set pixel values inside the circle to 1000

# Convert image data to bytes and set to DICOM dataset
dataset.PixelData = image_data.tobytes()

# Set DICOM file meta information
file_meta = Dataset()
file_meta.MediaStorageSOPClassUID = pydicom.uid.ImplicitVRLittleEndian
file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

# Set the file meta information in the dataset
dataset.file_meta = file_meta

# Set output file path
output_file = "example.dcm"

# Save the dataset as a DICOM file
dataset.save_as(output_file)

print(f"DICOM file has been created: {output_file}")