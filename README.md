
# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./data_visualized.png "Visualization"
[image2]: ./sample.png "Sample"
[image3]: ./sample_g.png "Grayscale"
[image4]: ./testimg/12c.jpg "Traffic Sign 1"
[image5]: ./testimg/21c.jpg "Traffic Sign 2"
[image6]: ./testimg/30c.jpg "Traffic Sign 3"
[image7]: ./testimg/40c.jpg "Traffic Sign 4"
[image8]: ./testimg/32c.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/buiducanh/traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

I chose to visualize the count of each sign in the training data set. I also visualize a sample of one sign and its grayscale transformation.

![alt text][image2]

![alt text][image3]

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

First I normalize the images in order for gradient descent to have a better convergence surface.

Finally, I grayscale the images because looking at the signs, they all have distinct features that do not have to do with colors. Thus, I hope that removing the colors will help the neural network learn these features. Since colors can be deceptive when there are distortions like sunlight, snow or shadows in the picture, I believe grayscale image will help the neural network be more general.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]
![alt text][image3]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x24 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x24 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Fully connected		| connecting 400 nodes to 120 neurons	        |
| Dropout				| with 50% probability        					|
| RELU					|												|
| Fully connected		| connecting 120 nodes to 84 neurons	        |
| Dropout				| with 50% probability        					|
| RELU					|												|
| Fully connected		| connecting 84 nodes to 43 neurons	            |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used AdamOptimizer as the gradient descent algorithm, with a batch size of 128, 10 epochs, and 0.001 learning rate. The loss function is the softmax cross entropy calculated from the logits at the final layer.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.951
* validation set accuracy of 0.951
* test set accuracy of 0.924

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I chose Lenet becaues Lenet works well with image data and it is easy to adapt.
* What were some problems with the initial architecture?
The architecture didn't have a good regularizer to help with overfitting.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I added dropout layers after each fully connected layers that are not the output layer. This gives me an easy way to adjust regularization strength of the model.
* Which parameters were tuned? How were they adjusted and why?
I tuned on the depth of the convolutional layers, and the learning rate. I decided to make a deep first convolutional layer because I want the neural network to learn many small features from the signs, since there are 43 signs in total and each of them has small features that can be differentiated. Then I play with the learning rate until the model converge nicely.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
I chose to add dropout layers because I want to regularize the model and encourage it to learn more general features, since the signs in reality can be distorted by the weather. It helps to have a way to control how much the model trusts the data.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because the angle of the picture is from the bottom up, so the sign is slanted. The second is straightforward. The third is hard because there is snow covering the sign. The fourth is hard because it is a drawing of the sign. The fifth one is straightforward.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			         |     Prediction	        					| 
|:----------------------:|:--------------------------------------------:| 
| Priority Road     	 | Yield   									    | 
| Double Curve     		 | Double Curve									|
| Beware of Ice/ Snow	 | Pedestrians									|
| End of all speed limits| Priority Road					 			|
| Roundabout			 | Roundabout        							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This is bad performance compared to the test set performance, but it shows that our concerns for the difficult qualities of the images we chose were true. All of the difficult images were predicted incorrectly. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is extremely sure that this is a yield sign (probability of 0.99), but the image is a priority road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Yield sign   									| 
| .0013   				| Turn right ahead								|
| .000083				| Keep left										|
| .0000022     			| No Entry   					 				|
| .0000012			    | Turn Left Ahead     							|


For the second image, the model is extremely sure that this is a double curve sign (probability of 0.95), and the image is a double curve sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .95         			| Double curve sign								| 
| .047   				| Wild animals crossing							|
| .00079				| Slippery road									|
| .00070     			| Right of way at next intersection				|
| .00066			    | Dangerous curve to the left					|


For the third image, the model is confused since the top prediction is 50% being pedestrian, but the image is a beware ice sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .50         			| Pedestrian sign   							| 
| .13   				| Right of way at next intersection				|
| .10				    | Roundabout									|
| .088       			| General Caution				 				|
| .049   			    | Speed limit (100 km/h)						|


For the fourth image, the model is relatively sure that this is a priority road sign (probability of 0.75), but the image is an end of all speed limits sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .75         			| Priority road sign							| 
| .079   				| Yield Sign     								|
| .053   				| Road work  									|
| .052      			| Stop Sign    					 				|
| .016  			    | Go straight or right 							|


For the fifth image, the model is relatively sure that this is a roundabout sign (probability of 0.84), and the image is a roundabout sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .84         			| Roundabout sign   							| 
| .15   				| Speed limit (100 km/h)						|
| .0059  				| Priority Road									|
| .0041     			| Speed limit (120 km/h)    	 				|
| .00015			    | Speed limit (80 km/h)							|



```python

```
