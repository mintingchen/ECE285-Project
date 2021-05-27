# ECE285-Project

## Quick start
### Train on Paris dataset
```
bash scripts/train_paris.sh MODEL_NAME
```

### Train on Cifar dataset

```
bash scripts/train_cifar.sh MODEL_NAME
```
Note:
1. Change the "MODEL_NAME" to whatever you like.
2. The checkpoints and config log will be saved in checkponts/MODEL_NAME.
3. In datahub, to visualize the training loss, choose "checkpoints" folder and click "Tensorboard".