
export CUDA_VISIBLE_DEVICES=0
#export CUDA_VISIBLE_DEVICES=0,1,2
data_dir="./data/binary/wmt16_en_de_bpe32k"
save_path="./myLiteTransExp"
my_model_path="./my_lite_plugins/"

python train.py ${data_dir}  --save-dir ${save_path} \
    --arch transformer_multibranch_v2_wmt_en_de --user-dir ${my_model_path} \
    --no-progress-bar --share-all-embeddings --log-interval 200 \
    --optimizer adam  --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --weight-decay 0.0 --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 --update-freq 16 --keep-last-epochs 10 \
    --ddp-backend no_c10d  --max-tokens 7200  --lr-scheduler cosine \
    --warmup-init-lr 1e-7 --warmup-updates 10000 --lr-shrink 1 \
    --max-epoch 10 --max-update 5000000  --lr 4e-4  \
    --dropout 0.1 --attention-dropout 0.08 --t-mult 1 \
    --encoder-layers 6 --decoder-layers 6 --weight-dropout 0.08 \
    --encoder-glu 1 --decoder-glu 1 --conv-linear  --lr-period-updates 40000 \
    --encoder-branch-type attn:1:256:4 dynamic:default:256:4 \
    --decoder-branch-type attn:1:256:4 dynamic:default:256:4 \
    --encoder-embed-dim 512 --encoder-ffn-embed-dim 512 \
    --decoder-embed-dim 512 --decoder-ffn-embed-dim 512 \
    --tensorboard-logdir ${save_path}/tensorboard --num-workers 1
#    --tensorboard-logdir ${save_path}/tensorboard --num-workers 1 --no-scale-emb --fp16 --fp16-scale-tolerance 0.25

