local voxel_size_func = import "autoseg/user_configs/anton/resolution_experiments/voxel_size_func";

voxel_size_func.get_pipeline_for_voxel_size([50, 2, 2], name="v2_UNet_s0") +
{
  predict+: {
    mask: [
      {
        path: "SynapseWeb/kh2015/oblique",
        dataset: "labels_mask/s0",
      }
    ],
    source: [
      {
        path: "SynapseWeb/kh2015/oblique",
        dataset: "raw/s0"
      }
    ]
  },
  evaluation: {
    method: "hagglom",
    results_dir: "results",
    ground_truth_skeletons: "eval/refined_skels.graphml",
    ground_truth_labels: [
      {
        path: "SynapseWeb/kh2015/oblique",
        dataset: "labels/s0"
      }
    ]
  }
}
