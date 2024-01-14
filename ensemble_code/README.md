# 2023-AICOSS
2023 AICOSS hackathon competition


## Weights

All 7 weights must be located at '/2023-AICOSS/sensible_code/weights/'

Also, the names of all weights should be: 
                                    
                                    ['cvt_q2l-pasl.pt', 'tresnet_xl_learnable_mldecoder-min_lr-1e-5.pt', 'tresnetv2_l_mldecoder-zlpr.pt',
                                    'tresnet_xl_learnable_mldecoder-AutoAugment-H-V.pt','tresnet_xl_mldecoder-pasl.pt',
                                    'tresnet_xl_mldecoder-two-loss.pt','cvt384_q2l-GradAccum8.pt']

<br/>

## Ensemble code

All ensembles can be done in single RTX 3090

3-ensemble
```bash
python ensemble.py \
--num_ensemble 3 \
--batch_size 1024 \
--gpu 0 \
--fast
```

6-ensemble
```bash
python ensemble.py \
--num_ensemble 6 \
--batch_size 512 \
--gpu 1 \
--fast
```

7-ensemble
```bash
python ensemble.py \
--num_ensemble 7 \
--batch_size 32 \
--gpu 2 \
--fast
```


## 6-Model Ensemble fp32 vs. fp16

| |FP32|FP16|diff.|
|:---:|:---:|:---:|:---:|
|Inference time [sec]|493|289|-41.38%|
|Inference speed [img/sec]|101.85|173.75|+70.59%|
|Public mAP|0.97618|0.97614|-0.00410%|
|Private mAP|0.97707|0.97702|-0.00512%|

