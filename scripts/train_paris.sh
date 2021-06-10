NAME=$1
python train.py --model Unet \
            --dataset paris \
            --epochs 200 \
            --batch_size 8 \
            --save_dir ./checkpoints/ \
            --name $NAME \
            --lr 0.0001 \
            --save_interval 10 \
            --image_dir dataset/paris/ \
            --image_list_train namelist/paris_training.txt \
            --image_list_test namelist/paris_testing.txt \
            --mask_dir dataset/mask/testing_mask_dataset/ \
            --mask_list_train namelist/nv_mask_training.txt \
            --mask_list_test namelist/nv_mask_testing.txt
            
python train.py --model Unet \
            --dataset paris \
            --epochs 50 \
            --batch_size 8 \
            --save_dir ./checkpoints/ \
            --mode ft \
            --init_weight ./checkpoints/20210609_paris_no_decay_1e4_conv/200.pt \
            --name 20210609_paris_no_decay_1e4_conv_200_ft \
            --lr 0.00001 \
            --save_interval 10 \
            --image_dir dataset/paris/ \
            --image_list_train namelist/paris_training.txt \
            --image_list_test namelist/paris_testing.txt \
            --mask_dir dataset/mask/testing_mask_dataset/ \
            --mask_list_train namelist/nv_mask_training.txt \
            --mask_list_test namelist/nv_mask_testing.txt