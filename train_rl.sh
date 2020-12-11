export CUDA_VISIBLE_DEVICES=0,1

work_space=/data1/jhliu/BertFastRLWorkSpace/bert_rl/model
abs_dir=/data1/jhliu/BertFastRLWorkSpace/bert_abstractor/model
ext_dir=/data1/jhliu/BertFastRLWorkSpace/bert_extractor/model

python train_full_rl.py --bert_max_sent_len 500 --path ${work_space} --abs_dir ${abs_dir} --ext_dir ${ext_dir} --fix_bert