python test.py --model Unet_light \
            --dataset cifar \
            --checkpoint ./checkpoints/test/100.pt \
            --name "model_name" \
            --seq_path ./output/ \
            --image_dir dataset/cifar/test \
            --image_list_train namelist/paris_training.txt \
            --image_list_test namelist/paris_training.txt \
            --mask_dir dataset/mask/testing_mask_dataset/ \
            --mask_list_train namelist/nv_mask_training.txt \
            --mask_list_test namelist/nv_mask_training.txt \
            --show_ratio 100