## Get started

### Prepare env
local host developing
```shell
conda create -n cond_graft python=3.8
conda activate cond_graft

conda install -c pytorch pytorch torchvision
pip install openmim
mim install mmdet
```
or an existing env on server
```shell
conda activate lym
```

Then make a soft link for data and add it to git-ignore
```shell
ln -s ~/Desktop/datasets ./data
```

### Conduct training and evaluation
```shell
# test training with single gpu on local host
sh scripts/host_train.sh
# train with gpu slurm
sh scripts/slurm_train.sh your_partition_name
```