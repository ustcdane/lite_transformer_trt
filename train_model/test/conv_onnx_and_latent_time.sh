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

# fairse plugins my_lite_onnx_plugins 和训练 my_lite_plugins不同之处在于 my_lite_plugins/models/liteConv_NAT.py forward不同

CUDA_VISIBLE_DEVICES=$gpu python $exe_wd/conv_onnx_and_latent_time.py $testBin  \
	--path "$checkpoints_path/checkpoint_last.pt" \
	--user-dir "$base_coder/my_lite_onnx_plugins" \
        --task translation_lev  --max-sentences $BS \
        --beam 1  --max-seq-len $MAX_SEQ \
        --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 \
        --iter-decode-with-beam 1 --gen-subset $subset > $checkpoints_path/exp/${subset}_gen.out 

avg_time=`grep  Total_time  $checkpoints_path/exp/${subset}_gen.out`
AVG=`echo $avg_time| awk '{print $NF}'`
echo "Avg time is  $AVG BS is $BS max seq len is $MAX_SEQ"
latent=$(echo "scale=4; $AVG * $BS" | bc)
echo  "BS latent  $latent"
