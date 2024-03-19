{
  predict: {
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
        path: "out_zarr.zarr",
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
