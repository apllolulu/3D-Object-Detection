# crooped gt dir
#GT_DIR=/media/hdc/KITTI/for_voxelnet/cropped_dataset/validation/label_2
GT_DIR=/media/ora/362b0807-5852-4af1-8f79-03a42964afb0/xy/3DObjectDetection/voxelnet/data/dataset/validation/label_2
# pred dir
PRED_DIR=$1

# output log
OUTPUT=$2
# start test
nohup `pwd`/kitti_eval/evaluate_object_3d_offline $GT_DIR $PRED_DIR > $OUTPUT 2>&1 &
