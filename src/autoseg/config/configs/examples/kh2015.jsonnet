
local train_config = import "autoseg/defaults/train";
local predict = import "autoseg/defaults/predict";
local augment = import "autoseg/defaults/augment";
local affs_target = import "autoseg/defaults/affs_target";
local lsds_target = import "autoseg/defaults/lsd_target";

local model = import "autoseg/defaults/3d_model";

{
  pipeline: {
    local pad = import "autoseg/defaults/pad",
    local zarrsource = import "autoseg/defaults/zarrsource",

    _order: ["source", "normalize", "augment", "lsd_target", "target"],
    #_outputs: ["RAW", "LABELS", "GT_AFFS", "AFFS_WEIGHTS", "GT_LSDS", "LSDS_WEIGHTS", "GT_AFFS_MASK", "LABELS_MASK"],
    _outputs: ["RAW", "LABELS", "GT_AFFS", "AFFS_WEIGHTS", "GT_AFFS_MASK", "LABELS_MASK"],

    source: [
      [zarrsource.zarr_source("SynapseWeb/kh2015/oblique")] + pad,
    ],

    normalize: [
      {normalize: {
        array: "RAW"
      }},
    ],

  } + augment + affs_target# + lsds_target

} + train_config + predict + model
