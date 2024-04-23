import sys
import numpy as np
from tqdm import tqdm

import zarr
from funlib.persistence import open_ds, prepare_ds
import daisy
from scipy.ndimage import binary_erosion
from logging import getLogger

logger = getLogger(__name__)


def erode(labels, steps):
    # get all foreground voxels by erosion of each component
    foreground = np.zeros(shape=labels.shape, dtype=bool)

    for section in range(labels.shape[0]):
        for label in tqdm(np.unique(labels[section])):
            # Assume that masked out values are the same as the label we are
            # eroding in this iteration. This ensures that at the boundary to
            # a masked region the value blob is not shrinking.
            if label == 0:
                continue
            label_mask = labels[section] == label

            eroded_label_mask = binary_erosion(
                label_mask, iterations=steps, border_value=1
            )

            foreground[section] = np.logical_or(eroded_label_mask, foreground[section])

    # label new background
    background = np.logical_not(foreground)
    labels[background] = 0

    return labels


if __name__ == "__main__":
    f = sys.argv[1]  # path to zarr
    # ds = sys.argv[2] #name of label dataset to erode
    # ds = "labels_filtered_relabeled"
    ds = "labels/s0"
    out_ds = "labels_f_r_eroded"
    iters = 7

    dtype = np.uint32

    seg_ds = open_ds(f, ds, mode="r")
    roi = seg_ds.roi
    voxel_size = seg_ds.voxel_size

    if len(seg_ds.shape) == 3 and seg_ds.shape[0] == 1:
        squeeze = True
        chunk_shape = seg_ds.chunk_shape[1:]
    else:
        squeeze = False
        chunk_shape = seg_ds.chunk_shape

    read_roi = write_roi = daisy.Roi([0] * len(voxel_size), chunk_shape) * voxel_size

    out_ds = prepare_ds(
        f,
        f"{out_ds}",
        total_roi=roi,
        voxel_size=voxel_size,
        write_size=write_roi.shape,
        dtype=dtype,
        delete=True,
    )

    def worker(block: daisy.Block):
        seg_array = seg_ds.to_ndarray(block.read_roi)

        if squeeze:
            seg_array = seg_array[0]

        if iters > 0:
            seg_array = erode(seg_array, iters)

        out_ds[block.write_roi] = seg_array

        return True

    # create task
    task = daisy.Task(
        "ErodeSegTask",
        total_roi=roi,
        read_roi=read_roi,
        write_roi=write_roi,
        process_function=worker,
        num_workers=32,
        max_retries=3,
        fit="shrink",
    )

    # run task
    ret = daisy.run_blockwise([task])
