export CUDA_VISIBLE_DEVICES=0,1,2

work_space=/data1/jhliu/BertFastRLWorkSpace/bert_extractor_finetune/model

python train_extractor_ml.py --use_bert --aux_cuda 2 --path ${work_space} --max_word 200 --max_sent 50 --cumul_batch 8 --batch 4
