#!/bin/bash

# ## full + aug
# hs=full
# tbl=std
# d_type="_augment"

# ## - aug
hs=full
tbl=std
d_type=""

## - aug - table
# hs=full
# tbl=no
# d_type=""

# ## - aug - table -history
# hs=no
# tbl=no
# d_type=""


# toy="--toy"
toy=""
# epoch=1 # 600 for spider, 200 for +aug

#DATE=`date '+%Y-%m-%d-%H:%M:%S'`

data_root=generated_datasets/generated_data${d_type}
save_dir="logdirs/20190225/pregen_noaug_encbaseline"
log_dir=${save_dir}/train_log
mkdir -p ${save_dir}
mkdir -p ${log_dir}


module=col
epoch=600
echo python -u train.py \
  --data_root    ${data_root} \
  --save_dir     ${save_dir} \
  --history_type ${hs} \
  --table_type   ${tbl} \
  --train_component ${module} \
  --epoch        ${epoch} \
  ${toy} \
  "| tee" "${log_dir}/train_${d_type}_hs=${hs}_tbl=${tbl}_${module}.txt" 

epoch=300
for module in multi_sql keyword op agg root_tem des_asc having andor
do
  echo python -u train.py \
    --data_root    ${data_root} \
    --save_dir     ${save_dir} \
    --history_type ${hs} \
    --table_type   ${tbl} \
    --train_component ${module} \
    --epoch        ${epoch} \
    ${toy} \
    "| tee" "${log_dir}/train_${d_type}_hs=${hs}_tbl=${tbl}_${module}.txt" 
done
