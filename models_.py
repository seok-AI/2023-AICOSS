import timm
import torch.nn as nn
import torch
from ml_decoder import MLDecoder, learnable_MLDecoder
import random
import os
import numpy as np


## Define Model
class swinv2(nn.Module):
    def __init__(self):
        super(swinv2, self).__init__()

        self.backbone = timm.create_model("swinv2_base_window8_256", pretrained=True)
        self.backbone.head.fc = nn.Identity()
        
        # Image Classifier
        # self.cls = nn.Linear(1536, 60) # large
        self.cls = nn.Linear(1024, 60)

    def forward(self, images):
        # Swin v2
        features_vit = self.backbone(images)
        
        # multi label classification
        out = self.cls(features_vit)
        return out
    
class resnet50(nn.Module):
    def __init__(self):
        super(resnet50, self).__init__()

        self.backbone = timm.create_model("resnet50", pretrained=True)
        self.backbone.fc = nn.Identity()
        
        # Image Classifier
        self.cls = nn.Linear(2048, 60)

    def forward(self, images):
        # Swin v2
        features_vit = self.backbone(images)
        
        # multi label classification
        out = self.cls(features_vit)
        return out
    
    
class resnet101(nn.Module):
    def __init__(self):
        super(resnet101, self).__init__()

        self.backbone = timm.create_model("resnet101", pretrained=True)
        self.backbone.fc = nn.Identity()
        
        # Image Classifier
        self.cls = nn.Linear(2048, 60)

    def forward(self, images):
        # Swin v2
        features_vit = self.backbone(images)
        
        # multi label classification
        out = self.cls(features_vit)
        return out
    

class resnet50_mldecoder(nn.Module):
    def __init__(self):
        super(resnet50_mldecoder, self).__init__()

        self.backbone = timm.create_model("resnet50", pretrained=True)
        self.backbone.fc = nn.Identity()
        self.backbone.global_pool = nn.Identity()
        # Image Classifier
        self.mldecoder = MLDecoder(num_classes=60)

    def forward(self, images):
        # Swin v2
        features_vit = self.backbone(images)
        
        # multi label classification
        out = self.mldecoder(features_vit)
        return out
    
    
class tresnet_xl_mldecoder(nn.Module):
    def __init__(self):
        super(tresnet_xl_mldecoder, self).__init__()

        self.backbone = timm.create_model("tresnet_xl", pretrained=True)
        self.backbone.head = nn.Identity()
        # Image Classifier
        self.mldecoder = MLDecoder(num_classes=60, initial_num_features=2656)

    def forward(self, images):
        # Swin v2
        features_vit = self.backbone(images)
        
        # multi label classification
        out = self.mldecoder(features_vit)
        return out

class tresnet_xl_learnable_mldecoder(nn.Module):
    def __init__(self):
        super(tresnet_xl_learnable_mldecoder, self).__init__()

        self.backbone = timm.create_model("tresnet_xl", pretrained=True)
        self.backbone.head = nn.Identity()
        # Image Classifier
        self.mldecoder = learnable_MLDecoder(num_classes=60, initial_num_features=2656)

    def forward(self, images):
        # Swin v2
        features_vit = self.backbone(images)
        
        # multi label classification
        out = self.mldecoder(features_vit)
        return out

class tresnet_l_learnable_mldecoder(nn.Module):
    def __init__(self):
        super(tresnet_l_learnable_mldecoder, self).__init__()

        self.backbone = timm.create_model("tresnet_l", pretrained=True)
        self.backbone.head = nn.Identity()
        # Image Classifier
        self.mldecoder = learnable_MLDecoder(num_classes=60, initial_num_features=2432)

    def forward(self, images):
        # Swin v2
        features_vit = self.backbone(images)
        
        # multi label classification
        out = self.mldecoder(features_vit)
        return out

class swinv2_mldecoder(nn.Module):
    def __init__(self):
        super(swinv2_mldecoder, self).__init__()

        self.backbone = timm.create_model("swinv2_base_window12_192", pretrained=True)
        self.backbone.head = nn.Identity()
        
        # Image Classifier
        # self.cls = nn.Linear(1536, 60) # large
        self.cls = MLDecoder(num_classes=60, initial_num_features=1024)

    def forward(self, images):
        # Swin v2
        features_vit = self.backbone(images).permute(0,3,1,2)
        # multi label classification
        out = self.cls(features_vit)
        return out
    

# [bs features 7 7]

class tresnetv2_l_mldecoder(nn.Module):
    def __init__(self):
        super(tresnetv2_l_mldecoder, self).__init__()

        self.backbone = timm.create_model("tresnet_v2_l", pretrained=True)
        self.backbone.head = nn.Identity()
        # Image Classifier
        self.mldecoder = MLDecoder(num_classes=60, initial_num_features=2048)

    def forward(self, images):
        # Swin v2
        features_vit = self.backbone(images)
        
        # multi label classification
        out = self.mldecoder(features_vit)
        return out

class tresnetv2_l_learnable_mldecoder(nn.Module):
    def __init__(self):
        super(tresnetv2_l_learnable_mldecoder, self).__init__()

        self.backbone = timm.create_model("tresnet_v2_l", pretrained=True)
        self.backbone.head = nn.Identity()
        # Image Classifier
        self.mldecoder = learnable_MLDecoder(num_classes=60, initial_num_features=2048)

    def forward(self, images):
        # Swin v2
        features_vit = self.backbone(images)
        
        # multi label classification
        out = self.mldecoder(features_vit)
        return out

# 모델 변경할 때, conv_out 부분만 바꿔주면 됨
class tresnet_xl_q2l(nn.Module):
    def __init__(
        self, conv_out=2656, num_classes=60, hidden_dim=256, nheads=8, 
        encoder_layers=1, decoder_layers=2):
        """
        Args:
            conv_out (int): Backbone output channels.
            num_classes (int): Number of possible label classes
            hidden_dim (int, optional): Hidden channels from linear projection of
            backbone output. Defaults to 256.
        """        
        
        super().__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        self.backbone = timm.create_model("tresnet_xl", pretrained=True)
        self.backbone.head = nn.Identity()
        self.conv = nn.Conv2d(conv_out, hidden_dim, 1)
        self.transformer = nn.Transformer(
            hidden_dim, nheads, encoder_layers, decoder_layers)


        # prediction head
        self.classifier = nn.Linear(num_classes * hidden_dim, num_classes)

        # learnable label embedding
        self.label_emb = nn.Parameter(torch.rand(1, num_classes, hidden_dim))

    def forward(self, x):
        
        # produces output of shape [N x C x H x W]
        out = self.backbone(x)
        
        # reduce number of feature planes for the transformer
        h = self.conv(out)
        B, C, H, W = h.shape

        # convert h from [N x C x H x W] to [H*W x N x C] (N=batch size)
        # this corresponds to the [SIZE x BATCH_SIZE x EMBED_DIM] dimensions 
        # that the transformer expects
        h = h.flatten(2).permute(2, 0, 1)
        
        # image feature vector "h" is sent in after transformation above; we 
        # also convert label_emb from [1 x TARGET x (hidden)EMBED_SIZE] to 
        # [TARGET x BATCH_SIZE x (hidden)EMBED_SIZE]
        label_emb = self.label_emb.repeat(B, 1, 1)
        label_emb = label_emb.transpose(0, 1)
        h = self.transformer(h, label_emb).transpose(0, 1)
        
        # output from transformer was of dim [TARGET x BATCH_SIZE x EMBED_SIZE];
        # however, we transposed it to [BATCH_SIZE x TARGET x EMBED_SIZE] above.
        # below we reshape to [BATCH_SIZE x TARGET*EMBED_SIZE].
        #
        # next, we project transformer outputs to class labels
        h = torch.reshape(h,(B, self.num_classes * self.hidden_dim))

        return self.classifier(h)


from transformers import CvtForImageClassification

class cvt_q2l(nn.Module):
    def __init__(
        self, conv_out=384, num_classes=60, hidden_dim=256, nheads=8, 
        encoder_layers=1, decoder_layers=2):
        """
        Args:
            conv_out (int): Backbone output channels.
            num_classes (int): Number of possible label classes
            hidden_dim (int, optional): Hidden channels from linear projection of
            backbone output. Defaults to 256.
        """        
        
        super().__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        self.backbone = CvtForImageClassification.from_pretrained('microsoft/cvt-21')
        self.backbone.classifier = nn.Identity()
        self.backbone.layernorm = nn.Identity()
        self.conv = nn.Conv2d(conv_out, hidden_dim, 1)
        self.transformer = nn.Transformer(
            hidden_dim, nheads, encoder_layers, decoder_layers)


        # prediction head
        self.classifier = nn.Linear(num_classes * hidden_dim, num_classes)

        # learnable label embedding
        self.label_emb = nn.Parameter(torch.rand(1, num_classes, hidden_dim))

    def forward(self, x):
        
        # produces output of shape [N x C x H x W]
        out = self.backbone(x, output_hidden_states=True).hidden_states[2]
        
        # reduce number of feature planes for the transformer
        h = self.conv(out)
        B, C, H, W = h.shape

        # convert h from [N x C x H x W] to [H*W x N x C] (N=batch size)
        # this corresponds to the [SIZE x BATCH_SIZE x EMBED_DIM] dimensions 
        # that the transformer expects
        h = h.flatten(2).permute(2, 0, 1)
        
        # image feature vector "h" is sent in after transformation above; we 
        # also convert label_emb from [1 x TARGET x (hidden)EMBED_SIZE] to 
        # [TARGET x BATCH_SIZE x (hidden)EMBED_SIZE]
        label_emb = self.label_emb.repeat(B, 1, 1)
        label_emb = label_emb.transpose(0, 1)
        h = self.transformer(h, label_emb).transpose(0, 1)
        
        # output from transformer was of dim [TARGET x BATCH_SIZE x EMBED_SIZE];
        # however, we transposed it to [BATCH_SIZE x TARGET x EMBED_SIZE] above.
        # below we reshape to [BATCH_SIZE x TARGET*EMBED_SIZE].
        #
        # next, we project transformer outputs to class labels
        h = torch.reshape(h,(B, self.num_classes * self.hidden_dim))

        return self.classifier(h)
