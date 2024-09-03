zarr_dir="/home/anton/.cache/autoseg/datasets/SynapseWeb/kh2015/data"

# Loop through all zarr files in the directory
for zarr in "$zarr_dir"/*.zarr.zip; do
    # Extract just the filename without the path
    zarr_name=$(basename "$zarr")

    echo "Processing $zarr_name ($zarr)"

    python pre_eval/erode.py "$zarr" labels/s1
    python skeletonize.py "$zarr" eroded/labels/s1

    echo "Finished processing $zarr_name ($zarr)"
    echo "-------------------------"
done
