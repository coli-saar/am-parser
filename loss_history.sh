# First argument: model directory
# Second argument, base path for data, e.g. data/AMR/2015/
model=$1
data=$2
cuda=0

# Annotate loss of gold trees:
python topdown_parser/annotate_loss.py  $model $data/gold-dev/gold-dev.amconll $model/gold_annotated.amconll --cuda-device $cuda

# Parse gold dev data:
python -m allenpipeline predict $model $data/gold-dev/gold-dev.amconll $model/best_parsed_gold.amconll --include-package topdown_parser --cuda-device 0

# Annotate loss of prediction:
python topdown_parser/annotate_loss.py  $model $model/best_parsed_gold.amconll $model/best_parsed_annotated.amconll --cuda-device $cuda

# Show results:
#200 bins
python topdown_parser/tools/logl_hist.py $model/best_parsed_annotated.amconll $model/gold_annotated.amconll 200 --save $model/loss_plott.svg


