python test.py --model Unet \
            --dataset paris \
            --checkpoint ./checkpoints/20210529_paris_100_ft/50.pt \
            --seq_path ./output/ \
            --image_dir dataset/paris/ \
            --image_list_train namelist/paris_training.txt \
            --image_list_test namelist/paris_testing.txt \
            --mask_dir dataset/qd_imd/test/ \
            --mask_list_test namelist/qd_mask_testing.txt \
            --mask_reverse No \
            --show_ratio 10
