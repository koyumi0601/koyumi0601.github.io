# pip install pynrrd
import numpy as np
import nrrd

# Data creation (this example creates a simple 3D array)
data = np.zeros((100, 100, 100), dtype=np.float32)
data[30:70, 30:70, 30:70] = 1.0  # Set some arbitrary data values

# Metadata creation for the data
header = {'spacings': [1.0, 1.0, 1.0],  # Spacing for each axis
          'units': ['mm', 'mm', 'mm'],  # Units for each axis
          'type': 'float',             # Data type
          'encoding': 'gzip',          # Compression method
          'space directions': np.eye(3)  # Adding the space directions for each axis
          }

# Save as NRRD file
output_path = 'example.nrrd'
nrrd.write(output_path, data, header)

print(f"NRRD file has been saved to {output_path}.")