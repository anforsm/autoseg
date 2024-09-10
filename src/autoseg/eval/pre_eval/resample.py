import zarr
import gunpowder as gp
import numpy as np
from funlib.persistence import prepare_ds

block_shape = gp.Coordinate((20, 200, 200))
voxel_size = gp.Coordinate([50, 2, 2])
target_voxel_size = gp.Coordinate([4, 4, 4])
block_size = block_shape * voxel_size


def resample(raw_file, in_dataset, out_file, out_dataset):
    raw = gp.ArrayKey("RAW")
    resampled = gp.ArrayKey("RESAMPLED")

    scan_request = gp.BatchRequest()
    scan_request.add(raw, block_size)
    scan_request.add(resampled, block_size)

    z = zarr.open(raw_file)
    arr = z[in_dataset]
    dtype = arr.dtype
    num_channels = 1

    source = gp.ZarrSource(
        raw_file,
        {
            raw: in_dataset,
        },
        {
            raw: gp.ArraySpec(interpolatable=True),
        },
    )

    with gp.build(source):
        total_roi = source.spec[raw].roi

    total_roi = total_roi.snap_to_grid(target_voxel_size)

    prepare_ds(
        out_file,
        out_dataset,
        total_roi,
        target_voxel_size,
        dtype,
        write_size=block_size,
        compressor={"id": "blosc", "clevel": 3},
        delete=True,
        num_channels=None,
    )

    scan = gp.Scan(scan_request, num_workers=50)

    write = gp.ZarrWrite(
        dataset_names={
            resampled: out_dataset,
        },
        store=out_file,
    )

    pipeline = (
        source
        + gp.Resample(
            raw,
            target_voxel_size,
            resampled,
        )
        + write
        + scan
    )

    request = gp.BatchRequest()

    # request[raw] = total_roi
    # request[resampled] = total_roi

    with gp.build(pipeline):
        pipeline.request_batch(request)

    return total_roi


if __name__ == "__main__":
    zarr_file = "/home/anton/.cache/autoseg/datasets/SynapseWeb/kh2015/data/spine.zarr"
    for ds in ["raw", "labels", "labels_mask"]:
        print(ds)
        resample(zarr_file, ds + "/s0", zarr_file, ds + "/resampled")
