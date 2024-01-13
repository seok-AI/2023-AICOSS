# 2023-AICOSS
[![2023 AICOSS hackathon competition](/pngs/banner.png)](https://dacon.io/competitions/official/236201/overview/description)


## Content
1. [train val split](#train-val-split)
2. [Training](#Training)
3. [preprocessing](#Preprocessing)
4. [Loss](#Loss)
5. [Model](#Model)
6. [Ablation Study](#Ablation-Study)
7. [Result](#Result)
8. [Leaderboard](#Leaderboard)
9. [Reference](#Reference)


## train val split

```python
result = []

for i in range(100):
    train_, val_ = train_test_split(train, test_size=0.1, random_state=i, shuffle=True)
    re_val = np.array(calcul_pro(val_))
    re_train = np.array(calcul_pro(train_))
    difference = re_train - re_val
    std_value = difference.std()
    if np.abs(std_value) < 0.25:
        mean_value = difference.mean()
        result.append((i, mean_value, std_value))

for state, mean, std in result:
    print(f'random_state: {state}  {mean:.2f} {std:.2f}')
```

We executed a methodical assessment of the class-wise ratio disparities between the training and validation sets across 100 different random states. Given that these measurements are expressed in percentages, the standard deviation (std) holds greater significance than the mean in our analysis. Hence, we proceeded with the selection of the random state that not only exhibits a std less than 0.25 but also possesses the minimal mean value within that subset.

<table>
  <tr>
    <td><img src="/pngs/77_vs_78_train_map.png" width="450" height="350" /></td>
    <td><img src="/pngs/77_vs_78_val_map.png" width="450" height="350" /></td>
  </tr>
</table>

Through the above method, we compared the training mAP of worst seed (77) and best seed (78) and the val mAP. While the training map is almost the same, the validation mAP of worst seed is performing much worse

## Training

Virtual Environment Settings
```bash
git clone https://github.com/seok-AI/2023-AICOSS
cd 2023-AICOSS/
conda env create -f aicoss.yaml
```

```bash
conda activate aicoss
```

training example
```bash
python main.py \
--model_name tresnet_xl_mldecoder \
--loss_name PartialSelectiveLoss \
--epochs 10 \
--lr 3e-4 \
--min_lr 1e-6 \
--weight_decay 1e-5 \
--augment weak \
--gpu 0
```

[Training Codes](/MODEL_WEIGHT.md)

For multi-GPU training, you can use [this repository](https://github.com/junpark-ai/AICOSS).

## preprocessing
**Cutout**

<!-- <figure>
    <img src="/pngs/cutout.png" width="800" height="600" />
</figure> -->

<figure>
    <img src="/pngs/cutout.png" width="600" height="230" />
</figure>

**RandAugment**

<figure>
    <img src="/pngs/RandAugment.png" width="647" height="547" />
</figure>

[RandAugment paper](https://arxiv.org/pdf/1909.13719.pdf)

**AutoAugment**

<figure>
    <img src="/pngs/AutoAugment.png" width="629" height="420" />
</figure>
[AutoAugment paper](https://arxiv.org/pdf/1805.09501.pdf)

## Loss
**PartialSelectiveLoss**
<figure>
    <img src="/pngs/PASL.png" width="900" height="300" />
</figure>
[PartialSelectiveLoss paper](https://arxiv.org/pdf/2110.10955.pdf)

## Model
**TResNet v2 L + ML decoder**

<figure>
    <img src="/pngs/MLDecoder.png" width="581" height="418" />
</figure>

[TresNet paper](https://arxiv.org/pdf/2003.13630.pdf)

[ML-Decoder paper](https://arxiv.org/pdf/2111.12933.pdf)

## Ablation Study

<table>
  <tr>
    <td><img src="/pngs/Ablation_Study_Train_Loss.png" width="450" height="350" /></td>
    <td><img src="/pngs/Ablation_Study_Val_Loss.png" width="450" height="350" /></td>
  </tr>
  <tr>
    <td><img src="/pngs/Ablation_Study_Train_mAP.png" width="450" height="350" /></td>
    <td><img src="/pngs/Ablation_Study_Val_mAP.png" width="450" height="350" /></td>
  </tr>
</table>


[ Ablation Study with TResNet_XL + ASL ]
When augmentation was not performed, validation loss increased and overfitting occurred. When only one augmentation was added, the performance was always better than when not added, and when all augmentation was applied, the best performance was achieved without overfitting.

|Augment|Train mAP|Validation mAP|
|---|---|---|
|None|1.0|0.9707|
|AutoAugment|0.9989|0.9720|
|Cutout|0.9971|0.9727|
|All|0.9872|0.9751|

## Result

<figure>
    <img src="/pngs/multi_gpu_.png" width="900" height="400" />
</figure>


<table>
  <tr>
    <td><img src="/pngs/Single Model SOTA_val.png" width="450" height="350" /></td>
    <td><img src="/pngs/Single Model SOTA_train.png" width="450" height="350" /></td>
  </tr>
</table>

|Model Name|Train mAP|Validation mAP|Test mAP|
|---|---|---|---|
|Single Model SOTA|0.9772|0.9765|0.9690|


## Leaderboard
<figure>
    <img src="/pngs/uosota.png" width="900" height="400" />
</figure>



## Reference
* https://github.com/Alibaba-MIIL/ML_Decoder
* https://github.com/Alibaba-MIIL/TResNet
* https://github.com/huggingface/pytorch-image-models/tree/main/timm
* https://arxiv.org/pdf/1909.13719.pdf
* https://arxiv.org/pdf/2110.10955.pdf
* https://arxiv.org/pdf/2003.13630.pdf
* https://arxiv.org/pdf/2111.12933.pdf
