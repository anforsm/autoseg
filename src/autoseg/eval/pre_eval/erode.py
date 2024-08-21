from tqdm import tqdm
import numpy as np
import zarr
import numpy as np
from scipy.ndimage import binary_erosion, grey_erosion
import sys

fn = sys.argv[1]
ds = sys.argv[2]

f = zarr.open(fn, "a")

arr = f[ds][:].astype(np.uint64)


eroded = np.zeros_like(arr)


for z in tqdm(range(arr.shape[0])):
    # eroded[z] = binary_erosion(arr[z], iterations=7)
    grey_erosion(arr[z], size=(7, 7), output=eroded[z])


# eroded = binary_erosion(arr)
# assert eroded.dtype == arr.dtype
# print("before", len(np.unique(arr)))
# print("after", len(np.unique(filtered)))

f[f"eroded/{ds}"] = eroded
f[f"eroded/{ds}"].attrs["offset"] = f[ds].attrs["offset"]
f[f"eroded/{ds}"].attrs["resolution"] = f[ds].attrs["resolution"]
