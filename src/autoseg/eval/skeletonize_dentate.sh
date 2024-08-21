zarr_dir="/home/anton/.cache/autoseg/datasets/SynapseWeb/team_dentate/data"

# Loop through all zarr files in the directory
for zarr in "$zarr_dir"/*.zarr; do
    # Extract just the filename without the path
    zarr_name=$(basename "$zarr")

    echo "Processing $zarr_name ($zarr)"

    python pre_eval/fix_dtype.py "$zarr" volumes/neuron_ids/s0
    python pre_eval/fix_dtype.py "$zarr" volumes/neuron_ids/s1
    python pre_eval/fix_dtype.py "$zarr" volumes/neuron_ids/s2
    python pre_eval/erode.py "$zarr" volumes/neuron_ids/s1
    python pre_eval/filter_relabel.py "$zarr" eroded/volumes/neuron_ids/s1
    python skeletonize.py "$zarr" relabelled_eroded/volumes/neuron_ids/s1

    echo "Finished processing $zarr_name ($zarr)"
    echo "-------------------------"
done
