import numpy as np
import zarr
import numpy as np
import sys

fn = sys.argv[1]
ds = sys.argv[2]

f = zarr.open(fn, "a")

resolution = f[ds].attrs["resolution"]
print("before")
print(resolution)


attrs = f[ds].attrs.asdict()
f[ds] = f[ds][:].astype(np.uint64)
f[ds].attrs.put(attrs)

resolution = f[ds].attrs["resolution"]
print("after")
print(resolution)
# f[ds].attrs["resolution"] = "resolution"
# f[ds].attrs["offset"] = "offset"
