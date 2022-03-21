# CNN For HGG/LGG Brain tumor Segmentation

Is a convolutional neural network inspired on the paper of *[ (S. Pereira et al.)]( http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7426413&isnumber=7463083)*  for the model implementation and the code of [Nikki Aldeborgh
(naldeborgh7575)](https://github.com/naldeborgh7575/brain_segmentation) for the patch extraction and image preprocessing.


The model for HGG tumor modality is the following:

| Layers      | Input           | Output |
| --- |---|---|
| Convolution | 4x33x33 | 64x33x33 |
|Leaky Relu|