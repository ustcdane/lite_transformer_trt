checkpoints_path="/root/workspace/lite_transformer_trt/myLiteTransExp" #$1
gpu=${2:-0}
subset=${3:-"test"}

mkdir -p $checkpoints_path/exp

base_coder="/root/workspace/lite_transformer_trt/"
exe_wd="/root/workspace/lite_transformer_trt/fairseq_cli"
testBin="/root/workspace/lite_transformer_trt/test/wmt16test"
CUDA_VISIBLE_DEVICES=$gpu python $exe_wd/onnx_generate.py $testBin  \
	--user-dir "$base_coder/my_lite_plugins" \
        --path "$checkpoints_path/checkpoint_best.pt" --gen-subset $subset \
        --beam 1 --batch-size 128 --remove-bpe  --lenpen 0.6 > $checkpoints_path/exp/${subset}_gen.out 

GEN=$checkpoints_path/exp/${subset}_gen.out

SYS=$GEN.sys
REF=$GEN.ref

#grep ^H $GEN | cut -f3- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $SYS
#grep ^T $GEN | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $REF
#python $exe_wd/score.py --sys $SYS --ref $REF | tee $checkpoints_path/exp/checkpoint_best.result
