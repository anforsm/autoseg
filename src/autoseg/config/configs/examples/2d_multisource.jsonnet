local zarrsource_2d = import "autoseg/defaults/zarrsource_2d";
local pad_2d = import "autoseg/defaults/pad_2d";

local train_config = import "autoseg/defaults/train";
local predict = import "autoseg/defaults/predict";
local augment = import "autoseg/defaults/augment_2d";


local input_image_shape = [196, 196];
local output_image_shape = [104, 104];

{
  pipeline: {
    _order: ["source", "normalize", "augment", "target"],

    source: [
      [[zarrsource_2d.zarr_source_2d(
        "/home/anton/.cache/autoseg/datasets/SynapseWeb/kh2015/data/2d_oblique.zarr/",
        i
      )] + pad_2d for i in std.range(10, 78)],
      {random_provider: {}}
    ],

    normalize: [
      {normalize: {
        array: "RAW"
      }},
    ],

    target: [
      {add_affinities: {
        labels: "LABELS",
        labels_mask: "LABELS_MASK",
        affinities: "GT_AFFS",
        affinities_mask: "GT_AFFS_MASK",
        affinity_neighborhood: [
          [-1,0],
          [0,-1],
        ]
      }},
      {balance_labels: {
        labels: "GT_AFFS",
        scales: "AFFS_WEIGHTS",
        mask: "GT_AFFS_MASK",
      }},
      {intensity_scale_shift: {
        array: "RAW",
        scale: 2,
        shift: -1
      }},
      {unsqueeze: {
        arrays: ["RAW"],
        axis: 0,
      }}
    ]

  } + augment,

  training: {
    multi_gpu: false,
    train_dataloader: {
      batch_size: 2,
      parallel: false,
      num_workers: 4,
      precache_per_worker: 4,
      input_image_shape: input_image_shape,
      output_image_shape: output_image_shape,
    },
    logging: {
      wandb: false,
    }
  },

  predict: {
    source: [
      {
        path: "/home/anton/.cache/autoseg/datasets/SynapseWeb/kh2015/data/2d_oblique.zarr",
        dataset: "raw/s0/" + i
      }
    for i in std.range(10, 30)],

    model: {
      path: "out/latest_model3.pt"
    },

    output: [
      // Multiple datasets such as AFFs and LSDs
      {
        path: "out_zarr_2d.zarr",
        dataset: "pred_affs",
        stacked: true,
      },

      // Return the result as a numpy array in memory
      // {
      //   return_np_array: true
      // }

    ],



  }

}
