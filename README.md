## Project: Build a Traffic Sign Recognition Program

Overview
---
In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

We have included an Ipython notebook that contains further instructions 
and starter code. Be sure to download the [Ipython notebook](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb). 

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting three files: 
* the Ipython notebook with the code
* the code exported as an html file
* a writeup report either as a markdown or pdf file

The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

All of the rubrics are self explained in the jupyter notebook. So I will focus on "What I Found" in the next part.

What I Found
------------
* Stuck

In the beginning, I thought the project would be simple, considering I have done similar project before.

It didn't take long for me to get stuck. In order to make my network different from LeNet, the first thing across my mind was to change the depth of the network and also add more neurons so that it could "understand" a dataset that is more complicated than MNIST. So I did it. but the result was that the network simply cannot train, no matter how many epochs I train, the loss stays almost at a constant value no more than 10, no less than 3; and the accuracy stays at 0.0***. I was stuck here for 3 days, so later I decided to switch back to LeNet, and Boom! It worked! I couldn't understand how that happend until I try to modify the network again.

* Possible causes

During trouble shooting my network, I also read a lot of examples from other people. There are some ideas I wanted to share.

1. Optimizer

![alt text][./writeup_pics/Long Valley - Imgur.gif]

Above is an amazing demonstration on how different optimizer behave. What I saw was the Adadelta optimizer was very good at dealing with some case that the network was hard to train. So I switched to Adadelta optimizer. Magic happened! Although in the beginning, it was also not learning, but after quite a few epochs, suddenly it start to learn, and soon the accuracy jump to more than 80 %.

2. Learning rate

From the optimizer lesson, I learnt either the network won't train at all, or once it start to train, it will be very easy to continue.

So I start to test learning rate. In the Udacity course, it mentioned that the good learning rate can range from 0.1 to 0.0001, I tried all, didn't work well, than I started to get bold, why not change it to 5! :) humm, not working, later I found in my network, the learning rate was good at 0.5, or even 1 or maybe something higher (although I didn't try). It could reduce the period that the loss could not be minimized.Next time! Don't be squeamish about using a special learning rate!


3. Batch size

At the beginning, there was a bug in my coding, I accidentally fixed the batch_size to be 128, although I left a global BATCH_SIZE constant there. I changed it many times but the GPU usage was always below 20%. Until I found this bug, I could train my network much faster and without a lot of spikes in the loss and accuracy in each batch, which means the result would be more stable.

4. Batch normalization

The project rubrics asked me to do normalization. in return, I did it in batch. It made the learning curve much more smooth and less over-fitting.

5. When to stop training

I was considering this question, so I let the network run for like forever. In the training set, the accuracy can reach 100% and the loss can be almost 0. In the validation set, the accuracy also decrease, than get stable, but the cross entropy loss will first reduce than start to increase. I figured the over-fitting started to get stronger once the validation set loss start to increase, so that should be the time to stop training, even when the validation accuracy still can decrease. Later I checked on the internet, many people also agreed this method.

6. Tensorboard

I implemented Tensorboard features in my project, it really helped me to troubleshoot. When my network could not train, all the parameters never spread out to any distribution other than my initialization. I could also see the feature maps viewed as images. It's a powerful tool, kind of making the neural network black box visible.

7. Cloud compute

AWS is expensive, so I tried several Elastic Compute Clouds in China. At beginning I used Meituan Cloud, it was very cheap, to get a Tesla M40 single core. but later the resource got really tight, I couldn't create instance after delete my old one. Than I had to move on to Aliyun, it seems more expensive, 5 times price comparing to Meituan Cloud, but offers Tesla P4. To my surprise, they offer bidding package, which was only 10% of the actual price most of the time. anyone interested can check below link.

https://promotion.aliyun.com/ntms/act/ambassador/sharetouser.html?userCode=hn0apcri&utm_source=hn0apcri

8. Docker

After learning how to use docker, it becomes very easy to deploy project in the cloud, which can save a lot of money.

This project is a small but great project for me, not because what I achieved in the result, but it led me to vast knowledge, making me knows what to explore.

Visualize using Tensorboard
---------------------------

[//]: # (Image References)

[image1]: ./writeup_pics/Long_Valley_Imgur.gif "optimizer"
[image2]: ./writeup_pics/network.png "Neural Network"
[image3]: ./writeup_pics/color_code.png "Colors"
[image4]: ./writeup_pics/overview.png "overview"
[image5]: ./writeup_pics/validation_acc.png
[image6]: ./writeup_pics/conv1.png
[image7]: ./writeup_pics/train_acc.png
[image8]: ./writeup_pics/validation_loss.png
[image9]: ./writeup_pics/conv1_structure.png
[image0]: ./writeup_pics/train_loss.png

* The Graph

flattern node is disconnected for better view.

![alt text][image2]

below is the conv1 structure, the bn & bn_1 node are batch normalization. There are 3 batch norm layers in the graph.

![alt text][image9]

* Loss & Accuracy in Training vs. Validation

1.Below is the color represent different runs

![alt text][image3]

| Run   |batch size | batch norm    |learn rate |
|:-----:|:---------:|:-------------:|:---------:|
| 1     |128        |No		        |0.1        |
| 2     |128        |No             |0.5        |
| 3	    |4096       |Yes            |0.5        |
| 4		|8192       |Yes       		|0.8        |

2. in run1, it is continued until training accuracy is almost always 1, and the loss is near 0. but you can see the validation loss start to increase after some time.

Note: in the validation accuracy and loss curve, the last data points are actually from the test set, so you can see the accuracy dropped and the loss increased.

![alt text][image4]

3. training loss and accuracy curves are very smooth in run2 and run3, because the implementation of batch normalization and bigger batch size, although with bigger learning rate, they were still smooth. Due to training stopped before validation loss increase, the over-fitting problem is managed.

Note: These are smoothed curves for better viewing, the actual spicky curves are light colored.

![alt text][image0]

![alt text][image7]

4. In the validation loss image below, you can see without batch_norm and with small batch size, model can get premature over-fitting. That is, first, validation loss start to increase before it drop to the lowest comparing to run3 and run4, second, the last data point, which is the test set loss, gives too much increase in the loss. the same can be concluded from the validation accuracy chart.

![alt text][image8]

![alt text][image5]


### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```

### Requirements for Submission
Follow the instructions in the `Traffic_Sign_Classifier.ipynb` notebook and write the project report using the writeup template as a guide, `writeup_template.md`. Submit the project code and writeup document.

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

