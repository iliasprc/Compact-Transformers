# Models

## Models from paper [Escaping the Big Data Paradigm with Compact Transformers](https://arxiv.org/abs/2104.05704)
**[vit]** : models based on the classical vision transformer with class token
**[Lightweight ViT]** : Different from [ViT](https://arxiv.org/abs/2010.11929) we show that <i>an image 
is <b>not always</b> worth 16x16 words</i> and the image patch size matters.
Transformers are not in fact ''data-hungry,'' as the authors proposed, and
smaller patching can be used to train efficiently on smaller datasets.

**[cvt]** : Compact Vision Transformers better utilize information with Sequence Pooling post 
encoder, eliminating the need for the class token while achieving better
accuracy.

## Our research models
#### early_fusion_image_grassmanian_vit 

Early fusion of image patches and observability matrices as input to a vanilla transformer

### grassmanian_cct 


### grassmanian_vit

Grassmanian  Transformer with attention that adopts SVD and QR decompositions to extract observability and orthogonal matrices,
while the attention is caluclated from distance on grasmmanian manifold


### img_gm_vit   

Late fusion of output of the two transformers, vanilla-Transformer and Grassmanian Transformer

### img_riem_vit 

Late fusion of output of the two transformers, vanilla-Transformer and Riemmanian Transformer


###  riemmanian_vit 

Riemmanian Transformer with attention that adopts covariance matrices and distance on riemmanian manifold

### img_gm_riem_vit  

### gm_riem_vit  





### manifold_cct  

### manifold_vit  

