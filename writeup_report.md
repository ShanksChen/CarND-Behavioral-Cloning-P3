# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup-pictures/dropout-after-first-fully-connected-layer.png "Architecture With Dropout After First Fully Conneced Layers"
[image2]: ./writeup-pictures/with-3-dropout.png "Architecture With Dropout After First Three Fully Conneced Layers"
[image3]: ./writeup-pictures/without-dropout.png "Architecture Without Dropout"
[image4]: ./writeup-pictures/problem1.png "Problem Detail 1"
[image5]: ./writeup-pictures/problem2.png "Problem Detail 2"
[image6]: ./writeup-pictures/LeNet-5.png "LeNet-5 Architecture"
[image7]: ./writeup-pictures/3-dropout-epcoh-10.png "Architecture with 3 dropout for epcoh=10"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
Recording a video.mp4 of my vehicle driving autonomously around the track by executing
```sh
pthon drive.py model.h5 run1
```
when the car finished one circle, to create a chronological video of the agent driving by executing
```sh
pthon video.py run1
```
the video's name is run1.mp4, but I rename it as "video.mp4".

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The model I uesd is which published by the autonomous vehicle team at NVIDIA using for training a real car to drive autonomously. The model consists a normalization layer, followed by five convolutional layers, followed by four fully connected layers.

### Model Architecture and Training Strategy

When I started to do the project, I tried to do the project on my local machine. But some problem occured. When I executing
```sh
python drive.py model.h5
```
And in the terminal showed this

![alt text][image4]

Then I run the simulator, the terminal showed this

![alt text][image5]

And I ask the mentor in the student hub about this problem, he can't give me effective solution. So I can only use the workspace provided by udacity.

In addition, the connection to the workspace and the simulator in the VM is very bad. After many attempts, I can only choose to use the sample drive data in the workspace for the project.

#### 1. A tried architecture

At the beginning, I use the LeNet-5 architecture. It consists of a normalization layer, followed by two convolutional layers, followed by three fully connected layers. After every convolutional layers, I add a max pooling layer. Moreover I add dropout function after first two fully connected layers to avoid over fitting(model.py lines 47-59). The normalization layer I used in the appropriate model architecture is same as in this architecture. So I will explain it later.
The loss of this architecture shows in below picture

![alt text][image6]

The training result seems good. But when test it in the autonomous mode, It can't achieve goals. The video file name is "video-lenet.mp4"

#### 2. An appropriate model architecture has been employed

My model which published by the autonomous vehicle team at NVIDIA consists of  a normalization layer, followed by five convolutional layers, followed by four fully connected layers. (model.py lines 62-80) 

The data is normalized in the model using a Keras lambda layer (model.py line 64) and a Cropping2D function (code line 65).

The first three convolutional layers in the model using stride 2 and "ReLU" activation function (model.py lines 67-69).

The other two convolutional layers in the model using default stride and "ReLU" activation function (model.py lines 70-71).

After the convolutional layers, I added a flatten layer (model.py lines 72).

Next there were four fulluy connected layers (model.py lines 74-80).

#### 3. Attempts to reduce overfitting in the model

I tried to add dropout layers in the model (model.py lines 77,77,79).But the Result is not good.

The picture below shows the result that using three dropout layers after first three convolution layers

![alt text][image2]

By this situation, the autonomous mode video shows in the "video-3-dropout-epoch5.mp4".

The picture below shows the result that using one dropout layer after the first convolution layer

![alt text][image1]

By this situation, the autonomous mode video shows in the "video-dropout-after-first-fully-connected-layer.mp4".

The picture below shows the result without using dropout layer

![alt text][image3]

By this situation, the autonomous mode video shows in the "video.mp4".

In order to make a better comparison, I added using three dropout layers after first three convolution layers but with ten epcoh.
The below picture shows the result

![alt text][image7]

By this situation, the autonomous mode video shows in the "video-3-dropout-epoch10.mp4".

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

Only the model which without dropout layer was tested by running it through the simulator and ensuring that the vehicle could stay on the track. Other three did not ensure that the vehicle could stay on the track.

In my opinion, under normal circumstances, the addition of dropout layer should be better able to ensure that the vehicle is driving in the car. But this is not the case. Compare the one dropout layer and three dropout layer, the result obtained by using only one dropout layer is better, at least the vehicle can go further in the track, although the car was drove out of the track at the second sharp turn and into the lake. But the three dropout layer situation even worse, the car was drove into the mud road and never came out. 

The same situation happended in the three dropout layers with 10 epcoh.

#### 4. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 82).

#### 5. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the sample drive data which provided by udacity. I have explained the reason.

Although, I created my training data on my local machine. It didn't work when I finished the training on local machine, then upload the h5 file to the worspace.

Let's talk about the sample drive data. *Using this method carries two benefits. One, we have more data to use for training the network. And two, the data we uesd for training the network is more comprehensive. (Explained by David Silver)*

Therefore, I did data augementation. 

### Result

I just keep the car on the first track.

I really want to finish this project on my local computer, but the problem I mentioned before has plagued me for nearly three days. I have to choose to complete this project on the workspace.

I got a lot of driving data on my computer, including driving two laps and trying to stay in the center, returning to the center of the lane from the edge of the lane, collecting data on the second track, collecting data clockwise and counterclockwise, etc.

When I solved the problem mentioned earlier, I will continue to try these training data and neural networks on my computer again. In order to complete this project better and even complete the challenge of Track two.