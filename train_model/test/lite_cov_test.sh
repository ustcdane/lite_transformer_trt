current_path=$(cd "$(dirname "$0")";pwd)
echo "Current path is $current_path"
project="$current_path/../"

checkpoints_path="$project/myLiteConvOPExp"
gpu=${2:-0}
subset=${3:-"test"}

mkdir -p $checkpoints_path/exp
base_coder="../train_model"
exe_wd="$project/fairseq_cli"
testBin="$project/test/wmt16test"

# change BS to stat model gpu latent of var bs size
BS=16
MAX_SEQ=32

CUDA_VISIBLE_DEVICES=$gpu python $exe_wd/conv_onnx_generate.py $testBin  \
        --path "$checkpoints_path/checkpoint_last.pt" \
	--user-dir "$base_coder/my_lite_onnx_plugins" \
        --task translation_lev  --max-sentences $BS \
        --beam 1  --max-seq-len $MAX_SEQ \
        --source-lang ${src} --target-lang ${tgt} --beam 1 \
        --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 \
        --iter-decode-with-beam 1 --gen-subset $subset > $checkpoints_path/exp/${subset}_gen.out 

GEN=$checkpoints_path/exp/${subset}_gen.out

SYS=$GEN.sys
REF=$GEN.ref

grep ^H $GEN | cut -f3- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $SYS
grep ^T $GEN | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $REF
python $exe_wd/score.py --sys $SYS --ref $REF | tee $checkpoints_path/exp/checkpoint_best.result
