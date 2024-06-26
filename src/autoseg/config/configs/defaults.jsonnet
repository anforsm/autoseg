local pipeline = import "autoseg/defaults/pipeline";
local train = import "autoseg/defaults/train";
local model = import "autoseg/defaults/model";
local predict = import "autoseg/defaults/predict";
local evaluation = import "autoseg/defaults/evaluation";

{}
 + pipeline
 + train
 + model.get_model_config()
 + predict
 + evaluation
