NAME=$1
python train.py --model Unet_light \
            --dataset cifar \
            --epochs 100 \
            --batch_size 16 \
            --lr 0.0001 \
            --save_dir ./checkpoints/ \
            --name $NAME \
            --save_interval 10 \
            --image_dir dataset/cifar/train \
            --image_list_train namelist/paris_training.txt \
            --image_list_test namelist/paris_testing.txt \
            --mask_dir dataset/mask/testing_mask_dataset/ \
            --mask_list_train namelist/nv_mask_training.txt \
            --mask_list_test namelist/nv_mask_testing.txt
