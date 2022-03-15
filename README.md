# Modern Convolutional Neural Network Architectures
Forked from https://github.com/Nyandwi/ModernConvNets

**重要性**：

卷积神经网络(卷积神经网络或cnn)是一类神经网络算法，主要用于图像分类、目标检测和图像分割等视觉识别任务。在视觉识别中使用卷积神经网络无疑是深度学习领域2010年代最重大的发明之一。

**背景信息:**：

一个标准的ConvNet架构通常由3个主要层组成，分别是卷积层、最大池化层和全连接层。卷积层是卷积网络的主要组成部分。它们用于使用过滤器提取图像中的特征。

池化层用于对卷积层生成的激活或特征图进行下采样。下采样也可以通过在正常的卷积层中使用跨距(大于1)来实现，但最大池化层没有任何可学参数，而且它们引入了平移不变性，从而提高了空间诱导偏差代价下的模型泛化。全连接层用于分类目的(将学习到的特征与其各自的标签匹配)。在分类设置中，最后一个全连接层通常使用softmax激活功能激活!

遵循上述结构的ConvNets架构的例子是AlexNet和VGG。但大多数现代的卷积网络架构已经超越了单纯的卷积堆栈、最大池化和全连接层。例如，像ResNet这样的架构和其他类似的网络都涉及到残差连接。

**问题:** 
如何选择合适的网络结构？

第一条经验法则是，您不应该试图从头开始设计自己的架构。如果你正在处理一般问题，从ResNet-50开始不会有什么坏处。如果您正在构建一个计算资源有限的基于移动的可视化应用，请尝试MobileNets(或其他移动友好的架构，如ShuffleNetv2或ESPNetv2)。

为了更好地平衡准确性和计算效率，请尝试EfficientNet或最新的ConvNeXt!

也就是说，选择架构(或学习算法)并不是免费的午餐。没有通用的架构。没有一个单一的体系结构可以保证对所有的数据集和问题都有效。都是实验。这是所有努力!

如果你是一个梦想家，或者喜欢呆在这个领域的前沿，看看视觉Transformers吧! 我们还不知道，但它们可能是ConvNets的继承者!

## ConvNet Architectures

* AlexNet - Deep Convolutional Neural Networks: [implementation](convnets/01-alexnet.ipynb), [paper](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
  
* VGG - Very Deep Convolutional Networks for Large Scale Image Recognition: [implementation](convnets/02-vgg.ipynb), [paper](https://arxiv.org/pdf/1409.1556.pdf)
  
* GoogLeNet(Inceptionv1) - Going Deeper with Convolutions: [implementation](convnets/03-googlenet.ipynb), [paper](https://arxiv.org/abs/1409.4842)

* ResNet - Deep Residual Learning for Image Recognition: [implementation](convnets/04-resnet.ipynb), [annotated paper](annotated_papers/resnet.pdf) [paper](https://arxiv.org/abs/1512.03385)

* ResNeXt - Aggregated Residual Transformations for Deep Neural Networks: [implementation](convnets/06-resnext.ipynb), [annotated paper](annotated_papers/resnext.pdf), [paper](https://arxiv.org/abs/1611.05431v2)

* Xception - Deep Learning with Depthwise Separable Convolutions: [implementation](convnets/07-xception.ipynb), [annotated paper](annotated_papers/xception.pdf), [paper](https://arxiv.org/abs/1610.02357)

* DenseNet - Densely Connected Convolutional Neural Networks: [implementation](convnets/05-densenet.ipynb), [annotated paper](annotated_papers/densenet.pdf), [paper](https://arxiv.org/abs/1608.06993v5)

* MobileNetV1 - Efficient Convolutional Neural Networks for Mobile Vision Applications: [implementation](convnets/08-mobilenet.ipynb), [annotated paper](annotated_papers/mobilenet.pdf), [paper](https://arxiv.org/abs/1704.04861v1)

* MobileNetV2 - Inverted Residuals and Linear Bottlenecks: [implementation](convnets/09-mobilenetv2.ipynb) [annotated paper](annotated_papers/mobilenetv2.pdf), [paper](https://arxiv.org/abs/1801.04381)

* EfficientNet - Rethinking Model Scaling for Convolutional Neural Networks: [implementation](convnets/10-efficientnet.ipynb), [annotated paper](annotated_papers/efficientnetv1.pdf), [paper](https://arxiv.org/abs/1905.11946v5). See also [EfficientNetV2](https://arxiv.org/abs/2104.00298v3)

* RegNet - Designing Network Design Spaces: [implementation](convnets/11-regnet.ipynb), [annotated paper](annotated_papers/regnet.pdf), [paper](hhttps://arxiv.org/abs/2003.13678). See also [this](https://arxiv.org/abs/2103.06877)

* ConvMixer - Patches are All You Need?: [implementation](convnets/12-convmixer.ipynb), [annotated paper](annotated_papers/convmixer.pdf), [paper](https://openreview.net/pdf?id=TVHS5Y4dNvM).

* ConvNeXt - A ConvNet for the 2020s: [implementation](convnets/13-convnext.ipynb), [annotated paper](annotated_papers/convnexts.pdf), [paper](https://arxiv.org/abs/2201.03545)

## An Empirical Study of CNN, Transformer, and MLP
CNN与Transformer、MLP性能对比
[paper](https://arxiv.org/pdf/2108.13002.pdf)  [code](https://github.com/microsoft/SPACH)

## Important Notes

The implementations of ConvNets architectures contained in this repository are not optimized for training but rather to understand how those networks were designed, principal components that makes them and how they evolved overtime. LeNet-5[(LeCunn, 1998)](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf) had 5 layers(with learnable weights, pooling layers excluded). AlexNet[(Krizhevsky et al., 2012)](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) had 8 layers. Few years later, Residual Networks[(He et al., 2015)](https://arxiv.org/abs/1512.03385) made the trends after showing that it's possible to train networks of over 100 layers. Today, residual networks are still one of the most widely used architecture across a wide range of visual tasks and they impacted the [design of language architectures](https://arxiv.org/abs/2203.00555). 

Computer vision research community is [very vibrant](https://twitter.com/Jeande_d/status/1446468370182799364).
With the intelligent frameworks and better architectures we have to day, understanding how networks architectures are designed before you can throw them in your dataset is never a neccesity, but it's one of the best ways to stay on top of this vibrant and fast-ever changing field!

If you want to use ConvNets for solving a visual recognition tasks such as image classification or object detection, you can get up running quickly by getting the models (and their pretrained weights) from tools like [Keras](https://keras.io), [TensorFlow Hub](https://tfhub.dev), [PyTorch Vision](https://github.com/pytorch/vision), [Timm PyTorch Image Models](https://github.com/rwightman/pytorch-image-models), [GluonCV](https://cv.gluon.ai), and [OpenMML Lab](https://github.com/open-mmlab).


## References Implementations

* [Keras Applications](https://github.com/keras-team/keras/tree/master/keras/applications)
* [Timm PyTorch Image Models](https://github.com/rwightman/pytorch-image-models)
* [PyTorch Vision](https://github.com/pytorch/vision)
* [Machine Learning Tokyo](https://github.com/Machine-Learning-Tokyo/CNN-Architectures)

## Further Learning

If you would like to learn more about ConvNets/CNNs, below are some few amazing resources:

* [CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu) - [Lecture 5](https://www.youtube.com/watch?v=bNb2fEVKeEo&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=5) and [Lecture 9](https://www.youtube.com/watch?v=DAOcjicFr1Y&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=9)

* [Deep Learning for Computer Vision ](https://www.youtube.com/playlist?list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r) - [Lecture 8](https://www.youtube.com/watch?v=XaZIlVrIO-Q&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r&index=8)

* [Paper With Code - Convolutional Neural Networks](https://paperswithcode.com/methods/category/convolutional-neural-networks)

* [MIT Introduction to Deep Learning](http://introtodeeplearning.com) - [Lecture 3](https://www.youtube.com/watch?v=AjtX1N_VT9E&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=4)

* [CS230 Deep Learning - CNN Cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks)

* [CNNs Interactive Explainer](https://poloclub.github.io/cnn-explainer/)
