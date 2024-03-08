
local train_config = import "autoseg/defaults/train";
local augment = import "autoseg/defaults/augment";
local affs_target = import "autoseg/defaults/affs_target";

{
  pipeline: {
    local pad = import "autoseg/defaults/pad",
    local zarrsource = import "autoseg/defaults/zarrsource",

    _order: ["source", "normalize", "augment", "target"],

    source: [
      [
        [zarrsource.zarr_source("SynapseWeb/kh2015/oblique")] + pad,
        [zarrsource.zarr_source("SynapseWeb/kh2015/spine")] + pad,
      ],
      {random_provider: {}}
    ],

    normalize: [
      {normalize: {
        array: "RAW"
      }},
    ],

  } + augment + affs_target

} + train_config
