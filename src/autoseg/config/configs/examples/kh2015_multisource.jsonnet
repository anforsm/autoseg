local kh2015 = import "autoseg/examples/kh2015";

local zarrsource = import "autoseg/defaults/zarrsource";
local pad = import "autoseg/defaults/pad";

kh2015 + {pipeline+: {source: [
  [
    [zarrsource.zarr_source("SynapseWeb/kh2015/oblique")] + pad,
    [zarrsource.zarr_source("SynapseWeb/kh2015/spine")] + pad,
  ],
  {random_provider: {}}
]}}
