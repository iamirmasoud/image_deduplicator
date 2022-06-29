# Automatic Image Deduplicator using PyTorch

## Project Overview

In this project, I'll create a neural network architecture consisting of both CNNs (Encoder) and LSTMs (Decoder) to automatically deduplicate images within a directory.
![Image Captioning Model](assets/encoder.png?raw=true) [Image source](https://arxiv.org/pdf/1411.4555.pdf)


## Preparing the environment
**Note**: I have developed this project on __Linux__. It can surely be run on Windows and Mac with some little changes.

1. Clone the repository, and navigate to the downloaded folder.
```
git clone https://github.com/iamirmasoud/image_deduplicator.git
cd image_deduplicator
```

2. Create (and activate) a new environment, named `deduplicator_env` with Python 3.7. If prompted to proceed with the install `(Proceed [y]/n)` type y.

	```shell
	conda create -n deduplicator_env python=3.7
	source activate deduplicator_env
	```
	
	At this point your command line should look something like: `(deduplicator_env) <User>:image_deduplicator <user>$`. The `(deduplicator_env)` indicates that your environment has been activated, and you can proceed with further package installations.

6. Before you can experiment with the code, you'll have to make sure that you have all the libraries and dependencies required to support this project. You will mainly need Python3.7+, PyTorch and its torchvision, OpenCV, and Matplotlib. You can install  dependencies using:
```
pip install -r requirements.txt
```

7. Navigate back to the repo. (Also, your source environment should still be activated at this point.)
```shell
cd image_deduplicator
```

8. Open the directory of notebooks, using the below command. You'll see all of the project files appear in your local environment; open the first notebook and follow the instructions.
```shell
jupyter notebook
```

9. Once you open any of the project notebooks, make sure you are in the correct `deduplicator_env` environment by clicking `Kernel > Change Kernel > deduplicator_env`.



## Jupyter Notebooks
The project is structured as a series of Jupyter notebooks that should be run in sequential order:

### [0. Dataset Exploration notebook](0_Dataset_Exploration.ipynb) 

This notebook initializes the [COCO API](https://github.com/cocodataset/cocoapi) (the "pycocotools" library) used to access data from the MS COCO (Common Objects in Context) dataset, which is "commonly used to train and benchmark object detection, segmentation, and captioning algorithms."

### [1. Architecture notebook](1_Architecture.ipynb) 

This notebook uses the pycocotools, torchvision transforms, and NLTK to preprocess the images and the captions for network training. It also explores details of EncoderCNN, which is taken pretrained from [torchvision.models, the ResNet50 architecture](https://pytorch.org/docs/master/torchvision/models.html#id3). The implementations of the EncoderCNN and DecoderRNN are found in the [model.py](model.py) file.

The core architecture used to achieve this task follows an encoder-decoder architecture, where the encoder is a pretrained ResNet CNN on ImageNet, and the decoder is a basic one-layer LSTM.

#### Architecture Details
![encoder-decoder-architecture](assets/encoder-decoder.png)

The left half of the diagram depicts the "EncoderCNN", which encodes the critical information contained in a regular picture file into a "feature vector" of a specific size. That feature vector is fed into the "DecoderRNN" on the right half of the diagram (which is "unfolded" in time - each box labeled "LSTM" represents the same cell at a different time step). Each word appearing as output at the top is fed back to the network as input (at the bottom) in a subsequent time step until the entire caption is generated. The arrow pointing right that connects the LSTM boxes together represents hidden state information, which represents the network's "memory", also fed back to the LSTM at each time step.

The architecture consists of a CNN encoder and RNN decoder. The CNN encoder is a pre-trained ResNet on ImageNet, which is a VGG convolutional neural network with skip connections. It has been proven to work really well on tasks like image recognition because the residual connections help model the residual differences before and after the convolution with the help of the identity block. A good pre-trained network on ImageNet is already good at extracting both useful low-level and high-level features for image tasks, so it naturally serves as a feature encoder for the image we want to caption. Since we are not doing the traditional image classification task, we drop the last fully connected layer and replace it without a new trainable fully connected layer to help transform the final feature map to an encoding that is more useful for the RNN decoder.

RNNs have long been shown useful in language tasks due to their ability to model data with sequential nature, such as language. Specifically, LSTMs even incorporate both long-term and short-term information as memories in the network. Thus, we pick an RNN decoder for the captioning task. Specifically, following the spirit of sequence to sequence (seq2seq) models used in translation, I leveraged the architecture choices in [this paper](https://arxiv.org/pdf/1411.4555.pdf) to use an LSTM to generate captions based on the encoded information from the CNN encoder. Specifically, **I first use the CNN encoder output concatenated with the "START" token as the initial input for the RNN decoder.** I apply a fully connected layer on the hidden states at that timestamp to output a softmax probability over the words in our entire vocabulary, where we choose the word with the highest probability as the word generated at that timestamp. Then, we feed this predicted word back again as the input for the next step. We continue so until we generated a caption of max length, or the network generated the "STOP" token, which indicates the end of the sentence.

#### LSTM Decoder
In the project, we pass all our inputs as a sequence to an LSTM. A sequence looks like this: first a feature vector that is extracted from an input image, then a start word, then the next word, the next word, and so on.

#### Embedding Dimension
The LSTM is defined such that, as it sequentially looks at inputs, it expects that each individual input in a sequence is of a consistent size and so we embed the feature vector and each word so that they are `embed_size`.

#### Using my trained model
You can [download](https://drive.google.com/file/d/1s3aRAdt8ZMqn53UUSSFLCDBJ_F-KNbpE/view?usp=sharing) my trained models by unzipping the `captioning_models.zip` file in the `models` directory of project for your own experimentation.

Feel free to experiment with alternative architectures, such as bidirectional LSTM with attention mechanisms!

### [2. Training notebook](2_Training.ipynb) 

This notebook provides the selection of hyperparameter values and EncoderRNN training. The hyperparameter selection is also explained.


#### Configs

- `batch_size` - the batch size of each training batch. It is the number of image-caption pairs used to amend the model weights in each training step. 
- `vocab_threshold` - the minimum word count threshold. Note that a larger threshold will result in a smaller vocabulary, whereas a smaller threshold will include rarer words and result in a larger vocabulary. 
- `vocab_from_file` - a Boolean that decides whether to load the vocabulary from file. 
- `embed_size` - the dimensionality of the image and word embeddings.




#### Image Transformations

In the original [ResNet](https://arxiv.org/pdf/1512.03385.pdf) paper, which is the ResNet architecture that our CNN encoder uses, it scales the shorter edge of images to 256, randomly crops it at 224, randomly samples, and horizontally flips the images, and performs batch normalization. Thus, to keep the best performance of the original ResNet model, it makes the most sense to keep the image preprocessing and transforms the same as the original model. Thus, I use the default `transform_train` as follows:

```
transform_train = transforms.Compose([ 
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.RandomCrop(224),                      # get 224x224 crop from random location
    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])
```
If you are gonna modifying this transform, keep in mind that:
- The images in the dataset have varying heights and widths, and 
- When using a pre-trained model, it must perform the corresponding appropriate normalization.


