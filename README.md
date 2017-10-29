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

Design and Test a Model Architecture
------------------------------------

### preprocessing techniques

The data was processed using batch_normalization technic, which is realized by tensorflow tf.nn.batch_normalization() function inside the computation graph. There are 3 batch normalization layers. 

Let's consider the 1st layer to be the preprocessing (in order to meet the project rubric). The batch normalization actually involves 2 of the following:

1. Normalization
2. Standardization
3. Average over a minibatch


What I Found
------------

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
[image11]: ./writeup_pics/pics_from_internet.png

### Stuck

In the beginning, I thought the project would be simple, considering I have done similar project before.

It didn't take long for me to get stuck. In order to make my network different from LeNet, the first thing across my mind was to change the depth of the network and also add more neurons so that it could "understand" a dataset that is more complicated than MNIST. So I did it. but the result was that the network simply cannot train, no matter how many epochs I train, the loss stays almost at a constant value no more than 10, no less than 3; and the accuracy stays at 0.0***. I was stuck here for 3 days, so later I decided to switch back to LeNet, and Boom! It worked! I couldn't understand how that happend until I try to modify the network again.

### Possible causes

During trouble shooting my network, I also read a lot of examples from other people. There are some ideas I wanted to share.

#### Optimizer

![alt text][image1]

Above is an amazing demonstration on how different optimizer behave. What I saw was the Adadelta optimizer was very good at dealing with some case that the network was hard to train. So I switched to Adadelta optimizer. Magic happened! Although in the beginning, it was also not learning, but after quite a few epochs, suddenly it start to learn, and soon the accuracy jump to more than 80 %.

#### Learning rate

From the optimizer lesson, I learnt either the network won't train at all, or once it start to train, it will be very easy to continue.

So I start to test learning rate. In the Udacity course, it mentioned that the good learning rate can range from 0.1 to 0.0001, I tried all, didn't work well, than I started to get bold, why not change it to 5! :) humm, not working, later I found in my network, the learning rate was good at 0.5, or even 1 or maybe something higher (although I didn't try). It could reduce the period that the loss could not be minimized.Next time! Don't be squeamish about using a special learning rate!


#### Batch size

At the beginning, there was a bug in my coding, I accidentally fixed the batch_size to be 128, although I left a global BATCH_SIZE constant there. I changed it many times but the GPU usage was always below 20%. Until I found this bug, I could train my network much faster and without a lot of spikes in the loss and accuracy in each batch, which means the result would be more stable.

#### Batch normalization

The project rubrics asked me to do normalization. in return, I did it in batch. It made the learning curve much more smooth and less over-fitting.

#### When to stop training

I was considering this question, so I let the network run for like forever. In the training set, the accuracy can reach 100% and the loss can be almost 0. In the validation set, the accuracy also decrease, than get stable, but the cross entropy loss will first reduce than start to increase. I figured the over-fitting started to get stronger once the validation set loss start to increase, so that should be the time to stop training, even when the validation accuracy still can decrease. Later I checked on the internet, many people also agreed this method.

#### Tensorboard

I implemented Tensorboard features in my project, it really helped me to troubleshoot. When my network could not train, all the parameters never spread out to any distribution other than my initialization. I could also see the feature maps viewed as images. It's a powerful tool, kind of making the neural network black box visible.

#### Cloud compute

AWS is expensive, so I tried several Elastic Compute Clouds in China. At beginning I used Meituan Cloud, it was very cheap, to get a Tesla M40 single core. but later the resource got really tight, I couldn't create instance after delete my old one. Than I had to move on to Aliyun, it seems more expensive, 5 times price comparing to Meituan Cloud, but offers Tesla P4. To my surprise, they offer bidding package, which was only 10% of the actual price most of the time. anyone interested can check below link.

https://promotion.aliyun.com/ntms/act/ambassador/sharetouser.html?userCode=hn0apcri&utm_source=hn0apcri

#### Docker

After learning how to use docker, it becomes very easy to deploy project in the cloud, which can save a lot of money.

This project is a small but great project for me, not because what I achieved in the result, but it led me to vast knowledge, making me knows what to explore.

Visualize using Tensorboard
---------------------------

### The Graph

flattern node is disconnected for better view.

![alt text][image2]

below is the conv1 structure, the bn & bn_1 node are batch normalization. There are 3 batch norm layers in the graph.

![alt text][image9]

###Loss & Accuracy in Training vs. Validation

Below is the color represent different runs

![alt text][image3]

| Run   |batch size | batch norm    |learn rate |
|:-----:|:---------:|:-------------:|:---------:|
| 1     |128        |No		        |0.1        |
| 2     |128        |No             |0.5        |
| 3	    |4096       |Yes            |0.5        |
| 4		|8192       |Yes       		|0.8        |

#### Overview
In run1, it is continued until training accuracy is almost always 1, and the loss is near 0. but you can see the validation loss start to increase after some time.

Note: in the validation accuracy and loss curve, the last data points are actually from the test set, so you can see the accuracy dropped and the loss increased.

![alt text][image4]

#### Training
training loss and accuracy curves are very smooth in run2 and run3, because the implementation of batch normalization and bigger batch size, although with bigger learning rate, they were still smooth. Due to training stopped before validation loss increase, the over-fitting problem is managed.

Note: These are smoothed curves for better viewing, the actual spicky curves are light colored.

![alt text][image0]

![alt text][image7]

#### Validation & Test
In the validation loss image below, you can see without batch_norm and with small batch size, model can get premature over-fitting. That is, first, validation loss start to increase before it drop to the lowest comparing to run3 and run4, second, the last data point, which is the test set loss, gives too much increase in the loss. the same can be concluded from the validation accuracy chart.

![alt text][image8]

![alt text][image5]


Test a Model on New Images
--------------------------

#### Test model with image from internet

I put 12 images in the demo_pics folder
 
the first 6 images are properly centered and sized, although there're some limitations. 

|image No.|pros				| cons			|
|:-------:|:-------------:|:-------------:|
|1			|zoom & center	|lost details|
|2    		|zoom & center	|nois white spot|
|3			|zoom & center	|lost details|
|4			|zoom & center	|shadow, and low resolution|
|5			|zoom & center	|dark|
|6			|zoom & center	|blur|
|7			|clear				|a bit out center, very bright and vivid|
|8			|zoom & center	|a bit gray|
|9			|clear				|doesn'te exist in the class|
|10			|zoom & center	|blur|
|11			|clear	|very small not center|
|12			|clear	|not center and got other sign below it|

The first 6 signs looks similar to the dataset(properly centered and zoomed, a bit dull in the color). While the last 6 pics could incure challenge(they are clear but looks different from the dataset in some aspects), for example, the 7th picture is very bright, with vivid color, and a bit out centered, the 9th picture does not exist in the classes during training (End of speed limit (100km/h)), but the it's similar to class 6 which is "End of speed limit (80km/h)", so I assume the classifier would recognize it as class 6. In the 11th picture, the sign is very tiny, it is very different from the training dataset, probably the classifier would fail here. Finally, the 12th picture is also not centered or zoomed, and got something below it.

![alt text][image11]

#### The performance on the new imags

after feeding the images into the network, the classifier output as below:

the predicted classes:
[26 40  4 35 14 17  2 23  6 14  3 25]

Actual class should be:
[26, 40, 4, 35, 14, 17, 2, 23, 6, 14, 4, 26]

Accuracy for these images is: 83.33333134651184 %

We can see, the classifier got good response in the first 10 images, and the 9th image which is not include in the trained class got a similar return --- End of speed limit (80km/h). It was good result, it means my network can deal with poor lighting or poor camera/photographer.

The only difficulties happens in the last 2 images, which means my network has not learnt to do zoom and center. It make sense, becasue I did not train for it. So if i want my classifier to be more robust, I need to let it train for dealing with images that are not zoomed or centered.

I haven't put images with distortion or rotation, or add special blockage in front of the sign, I believe it won't give any meaning for result.

#### Top 5 probabilities

From the output below, in the first 11 pictures, the network was very confident with his result, although it got wrong in the 11th picture, it made me check what is it actually points to, the result was class 3 "Speed limit (60km/h)", almost there!! The last picture, 25 "Road work", No.... Notice the actual class are not even in the top 5, so it's kind of random guess...

TopKV2(values=array(

[[  9.99998808e-01,   6.92192089e-07,   4.31497057e-07,
          6.41800000e-08,   1.46565435e-08],
       [  9.99619842e-01,   3.80067824e-04,   6.60506956e-08,
          1.91607086e-09,   7.90609966e-10],
       [  9.99999881e-01,   9.58321564e-08,   2.95874170e-08,
          2.44568401e-08,   1.23732380e-09],
       [  9.99999642e-01,   3.05283208e-07,   7.84031340e-10,
          7.35486172e-10,   3.96830901e-10],
       [  9.99896169e-01,   5.93730983e-05,   1.88053782e-05,
          1.21305202e-05,   1.02408976e-05],
       [  1.00000000e+00,   1.97430207e-13,   5.31566220e-16,
          3.28441624e-17,   6.72244018e-18],
       [  9.99995232e-01,   4.59754756e-06,   1.37037929e-07,
          7.34646690e-08,   1.60690607e-08],
       [  1.00000000e+00,   8.54227855e-10,   5.10739895e-10,
          5.90200686e-13,   2.13553987e-13],
       [  9.99561369e-01,   3.97805299e-04,   3.88155458e-05,
          1.87778051e-06,   3.11875468e-08],
       [  9.99954224e-01,   2.14593401e-05,   1.85765421e-05,
          2.09246150e-06,   8.43388705e-07],
       [  6.54774904e-01,   1.19296946e-01,   1.17136344e-01,
          1.08741641e-01,   1.55527105e-05],
       [  5.93723714e-01,   3.22844625e-01,   4.64011990e-02,
          1.23350089e-02,   1.22207869e-02]], dtype=float32),

indices=array(

       [[26, 29, 18, 22, 25],
       [40, 37, 18, 33, 39],
       [ 4, 19, 39, 26, 15],
       [35, 34, 40, 12, 38],
       [14,  0, 17,  8,  1],
       [17, 14, 25, 42, 41],
       [ 2,  1,  5, 13, 38],
       [23, 11, 30, 19, 21],
       [ 6, 42, 16,  5, 40],
       [14,  8,  9,  5, 15],
       [ 3,  9, 35, 36, 16],
       [25, 12, 38, 11, 32]], dtype=int32))

Actual class should be:
[26, 40, 4, 35, 14, 17, 2, 23, 6, 14, 4, 26]

Others
------


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

