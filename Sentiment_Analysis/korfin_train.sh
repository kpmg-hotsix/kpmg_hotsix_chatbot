deepspeed --num_gpus=2 korfin_train.py --model_name lighthouse/mdeberta-v3-base-kor-further --batch_size 32
# deepspeed --num_gpus=2 korfin_train.py --model_name jinmang2/kpfbert
deepspeed --num_gpus=2 korfin_train.py --model_name klue/roberta-large --batch_size 32
# deepspeed --num_gpus=2 korfin_train.py --model_name tunib/electra-ko-base
# deepspeed --num_gpus=2 korfin_train.py --model_name lassl/bert-ko-base
deepspeed --num_gpus=2 korfin_train.py --model_name snunlp/KR-FinBert-SC --batch_size 32
deepspeed --num_gpus=2 korfin_train.py --model_name KB/bert-base-swedish-cased --batch_size 32
deepspeed --num_gpus=2 korfin_train.py --model_name beomi/kcbert-base --batch_size 32
deepspeed --num_gpus=2 korfin_train.py --model_name beomi/KcELECTRA-base-v2022 --batch_size 32
deepspeed --num_gpus=2 korfin_train.py --model_name snunlp/KR-Medium --batch_size 32