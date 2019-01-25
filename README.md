## Semantic Segmentation: a practice on Fully Convolutional Network (FCN) and Deconvelutional Network (DeconvNet)
### Project for course _Deep Learning Do It Yourself 2018/2019_ at ENS Paris


#### Pierre Jobic, Corentin Barloy and Kexin Ren


##### Data:
PASCAL VOC 2012 segmentation dataset (train + val)
- training images = 1444
- val images = 1464



##### Data Preprocessing:
color map (3*224*224) --> one-hot class map (21 * 224 * 224)



##### FCN training:
- Device: Google Colab w/ GPU
- #epochs = 50; 100
- Batch size = 64
- Learning rate = 0.0001 
- Scheduler_lr = (factor = 0.1, each 15 steps)
- Loss func =  CrossEntropyLoss
- Pre-trained VGG (ILSVRC dataset)



##### FCN results:
in _results_ folder, there are 3 versions of FCN -
(1) lr = 0.0001, 50 epochs
(2) lr_scheduler(0.1, 15) (initial lr = 0.0001), 50 epochs
(3) lr_scheduler(0.1, 20) (initial lr = 0.0001), 100 epochs



##### DeconvNet training:
- Device: Google Colab w/ GPU
- #epochs = 50
- Batch size = 64
- Learning rate = 0.001
- Loss func =  MSELoss



##### Reference:
FCN: Long, J., Shelhamer, E., & Darrell, T. (2015). Fully convolutional networks for semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3431-3440).


DeconvNet: Noh, H., Hong, S., & Han, B. (2015). Learning deconvolution network for semantic segmentation. In Proceedings of the IEEE international conference on computer vision (pp. 1520-1528).

