{
  predict: {
    multi_gpu: false,
    num_workers: 10,
    source: [
      {
        path: "SynapseWeb/kh2015/oblique",
        dataset: "raw/s0"
      }
    ],

    model: {
      path: "out/latest_model3.pt"
    },

    output: [
      // Multiple datasets such as AFFs and LSDs
      {
        path: "multiout.zarr",
        dataset: "preds/lsds",
        num_channels: 10,
        #stacked: true,
      },
      {
        path: "multiout.zarr",
        dataset: "preds/affs",
        num_channels: 3,
        #stacked: true,
      },

      // Return the result as a numpy array in memory
      // {
      //   return_np_array: true
      // }

    ],



  }
}
