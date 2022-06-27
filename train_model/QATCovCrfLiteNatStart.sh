export CUDA_VISIBLE_DEVICES=0
data_dir="./data/binary/wmt16_en_de_bpe32k"
save_path="./myQATLiteConvOPExp"
my_model_path="./my_lite_plugins/"

python train.py ${data_dir}  --save-dir ${save_path} \
    --arch conv_crf_transformer --user-dir ${my_model_path}  \
    --noise full_mask --no-progress-bar --quant-noise-scalar 1.0 \
    --log-interval 1000 --optimizer adam --adam-betas '(0.9, 0.999)' \
    --adam-eps 1e-6 --task translation_lev --apply-bert-init \
    --clip-norm 5 --weight-decay 0.01 --criterion nat_loss  --src-embedding-copy \
    --label-smoothing 0.1 --keep-last-epochs 5 \
    --ddp-backend no_c10d  --max-tokens 12000  --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-7 --stop-min-lr 1e-9 --warmup-updates 4000  \
    --max-epoch 45 --max-update 2000000  --conv-linear  \
    --lr 5e-4  --dropout 0.1 --attention-dropout 0.08 --activation-fn gelu \
    --encoder-layers 6 --decoder-layers 6 --weight-dropout 0.08 \
    --encoder-glu 1 --decoder-glu 1 --decoder-layerdrop 0.02 \
    --max-source-positions 254 --max-target-positions 254 \
    --length-loss-factor 0.2 --eval-bleu \
    --encoder-embed-dim 512 --encoder-ffn-embed-dim 2048 \
    --decoder-embed-dim 512 --decoder-ffn-embed-dim 2048 \
    --encoder-kernel-size-list 3 5 5 7 15 15 \
    --decoder-kernel-size-list 3 5 5 7 15 15 \
    --word-ins-loss-factor 0.5 --crf-lowrank-approx 32 --crf-beam-approx 32 \
    --eval-bleu-args '{"iter_decode_max_iter": 0, "iter_decode_with_beam": 1}' \
    --maximize-best-checkpoint-metric --decoder-learned-pos --encoder-learned-pos \
    --num-workers 1 --reset-optimizer --skip-invalid-size-inputs-valid-test
