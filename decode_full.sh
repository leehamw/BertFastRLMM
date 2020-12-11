export CUDA_VISIBLE_DEVICES=0,1

work_space=/data1/jhliu/BertFastRLWorkSpace/bert_rl/model/
save_space=/data1/jhliu/BertFastRLWorkSpace/bert_rl_test_result/file

python decode_full_model.py --bert_max_sent_len 500 --path ${save_space} --model_dir ${work_space} --test --fix_bert