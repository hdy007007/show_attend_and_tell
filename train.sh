DATA=$(date +%y%m%d%H%M) 

python train.py 2>&1|tee ./train_batch/log/train$DATA.log