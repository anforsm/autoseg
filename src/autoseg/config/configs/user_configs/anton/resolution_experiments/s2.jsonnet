local voxel_size_func = import "autoseg/user_configs/anton/resolution_experiments/voxel_size_func";

voxel_size_func.get_pipeline_for_voxel_size([50, 8, 8], name="UNet_s2")
 +
{
  predict+: {
    mask: [
      {
        path: "SynapseWeb/kh2015/oblique",
        dataset: "labels_mask/s2",
      }
    ],
    source: [
      {
        path: "SynapseWeb/kh2015/oblique",
        dataset: "raw/s2"
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
        dataset: "labels/s2"
      }
    ]
  }
}
