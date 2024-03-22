
local train_config = import "autoseg/defaults/train";
local predict = import "autoseg/defaults/predict";
local augment = import "autoseg/defaults/augment";
local affs_target = import "autoseg/defaults/affs_target";

local model = import "autoseg/defaults/3d_model";

{
  pipeline: {
    local pad = import "autoseg/defaults/pad",
    local zarrsource = import "autoseg/defaults/zarrsource",

    _order: ["source", "normalize", "augment", "target"],

    source: [
      [zarrsource.zarr_source("SynapseWeb/kh2015/oblique")] + pad,
    ],

    normalize: [
      {normalize: {
        array: "RAW"
      }},
    ],

  } + augment + affs_target

} + train_config + predict + model
