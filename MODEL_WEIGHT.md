# 2023-AICOSS
2023 AICOSS hackathon competition


## Weights

|Model Name|Loss|Prior|LR|MIN_LR|Augment|Batch Size|DDP|
|---|---|---|---|---|---|---|---|
|[TResNet_XL_ML-Decoder](https://drive.google.com/file/d/1-3vIOifnBwPmFoANO2-Z0fQ-EgKIOdS6/view?usp=sharing)|P-ASL|X|3e-4|1e-6|Weak|128|X|
|[TResNet_XL_ML-Decoder](https://drive.google.com/file/d/1-4IsvCrmixW7dFYqe3Oo6a9D5tHURNQa/view?usp=sharing)|Two-Way|-|3e-4|1e-6|Weak|128|X|
|[TResNet_v2_L_ML-Decoder](https://drive.google.com/file/d/1-IbW3bcqPh00QD5LYFqqPdtL-c-VD1Oo/view?usp=sharing)|ZLPR|-|3e-4|1e-6|Weak|128|X|
|[CvT_21_Q2L](https://drive.google.com/file/d/193JMT1IdexNFFJpiNfG-gxNDylTWA1BX/view?usp=sharing)|P-ASL|X|3e-4|1e-6|Weak|400|O|
|[TResNet_XL_Learnable_ML-Decoder](https://drive.google.com/file/d/1-1-FHMt8EJYM_8eGgCqVR_jZV_oVCRQe/view?usp=sharing)|P-ASL|O|3e-4|1e-6|Strong|128|X|
|[TResNet_XL_Learnable_ML-Decoder](https://drive.google.com/file/d/1-3oIn3zzd6hHrPembaP6_g-ahQdWjiaf/view?usp=sharing)|P-ASL|O|3e-4|1e-5|Strong|128|X|


The above six models can be Ensemble with single RTX 3090 GPU. (Batch Size 512)


|Ensemble|Public mAP|Private mAP|Single GPU Train Time|Multi GPU Train Time|
|---|---|---|---|---|
|Single Model|0.96904|---|1h|<20m|
|3-Model|0.97421|---|3h|<1h|
|6-Model|0.97618|---|6h|<2h|
|6-Model + CvT384|0.97705|0.97764|-|<9h|
|6-Model + CvT384 + SwinV2|0.97848|---|-|??|
