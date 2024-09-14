local voxel_size_func = import "autoseg/user_configs/anton/resolution_experiments/voxel_size_func";

voxel_size_func.get_pipeline_for_voxel_size([50, 16, 16], name="v2_UNet_s3")
