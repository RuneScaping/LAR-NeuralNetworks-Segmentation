# CNN For HGG/LGG Brain tumor Segmentation

Is a convolutional neural network inspired on the paper of *[ (S. Pereira et al.)]( http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7426413&isnumber=7463083)*  for the model implementation and the code of [Nikki Aldeborgh
(naldeborgh7575)](https://github.com/naldeborgh7575/brain_segmentation) for the patch extraction and image preprocessing.


The model for HGG tumor modality is the following:

| Layers      | Input           | Output |
| --- |---|---|
| Convolution | 4x33x33 | 64x33x33 |
|Leaky Relu| 64x33x33 | 64x33x33 |
| Convolution  | 64x33x33| 64x33x33|
|Leaky Relu| 64x33x33 | 64x33x33 |
| Convolution  | 64x33x33| 64x33x33|
|Leaky Relu| 64x33x33 | 64x33x33 |
|Max Pooling|  64x33x33| 64x16x16 |
| Convolution  | 64x16x16| 128x16x16|
|Leaky Relu | 128x16x16| 128x16x16|
| Convolution  | 128x16x16| 128x16x16|
| Convolution  | 128x16x16| 128x16x16|
|Leaky Relu | 128x16x16| 128x16x16|
|Max Pooling| 128x16x16| 128x7x7|
|Fully Connected|6272|256|
|Fully Connected|256|5|




 For LGG tumor modality instead the following

| Layers      | Input           | Output |
| --- |---|---|
| Convolution | 4x33x33 | 64x33x33 |
|Leaky Relu| 64x33x33 | 64x33x33 |
| Convolution  | 64x33x33| 64x33x33|
|Leaky Relu| 64x33x33 | 64x33x33 |
|Max Pooling|  64x33x33| 64x16x16 |
| Convolution  | 64x16x16| 128x16x16|
|Leaky Relu| 128x16x16| 128x16x16|
| Convolution  | 128x16x16| 128x16x16|
|Max Pooling| 128x16x16| 128x7x7|
|Fully Connected|6272|256|
|Fully Connected|256|5|


For both models in the and is used the SoftMax activation function.

1.  In 
	* brain_pipeline
	* patch_extractor
	* patch_library
 the conversion of all '.mha'  files into '.png'  to all brain images is performed. To each brain image from every patient, all different modalities ( (FLAIR), T1, T1-contrasted, and T2 )   are put together into one  single stripe .  The output for an image is the following:
[image](https://github.com/cvdlab/nn-segmentation-for-lar/blob/mast