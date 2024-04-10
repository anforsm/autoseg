{
  predict: {
    multi_gpu: false,
    num_workers: 6,
    shape_increase: [0, 800, 800],
    source: [
      {
        #path: "/home/anton/.cache/autoseg/datasets/SynapseWeb/kh2015/data/spine.zarr",
        #path: "SynapseWeb/kh2015/spine",
        #path: "/scratch/09699/anforsm/autoseg_cache/datasets/SynapseWeb/kh2015/data/spine.zarr",
        #path: "/tmp/spine.zarr.zip",
        path: "/tmp/oblique.zarr.zip",
        dataset: "raw/s0",
      }
    ],

    output: [
      {
        path: "/tmp/oblique_prediction.zarr",
        dataset: "preds/affs",
        num_channels: 3,
        #stacked: true,
      },
    ],
  }
}
