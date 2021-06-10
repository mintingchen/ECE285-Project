python test.py --model Unet_light \
            --dataset cifar \
<<<<<<< HEAD
            --checkpoint ./checkpoints/20210606_paris_no_decay_1e4_200_ft/50.pt \
            --seq_path ./output/ \
=======
            --checkpoint ./checkpoints/cifar_model/50.pt \
            --name "cifar_model" \
            --seq_path ./output_cifarmodelft/ \
>>>>>>> bde5aaf610a80654650dff9f51902396088a626b
            --image_dir dataset/cifar/test \
            --image_list_train namelist/paris_training.txt \
            --image_list_test namelist/paris_training.txt \
            --mask_dir dataset/mask/testing_mask_dataset/ \
            --mask_list_train namelist/nv_mask_training.txt \
            --mask_list_test namelist/nv_mask_training.txt \
            --show_ratio 500