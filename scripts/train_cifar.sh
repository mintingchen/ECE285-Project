python train.py --model Unet_light \
            --dataset cifar \
            --epochs 1 \
            --batch_size 16 \
            --image_dir dataset/cifar/test \
            --image_list_train namelist/paris_training.txt \
            --image_list_test namelist/paris_training.txt \
            --mask_dir dataset/mask/testing_mask_dataset/ \
            --mask_list_train namelist/nv_mask_training.txt \
            --mask_list_test namelist/nv_mask_training.txt
