# set config to be first positional arg
config=$1
# run train script
conda activate segmentation
python train.py $config
# run prediction script
python predict.py $config
conda activate autoseg2
# run post processing for fragments etc.
python post_processign/main.py $config
# run evaluation script
python eval/evaluate.py $config
