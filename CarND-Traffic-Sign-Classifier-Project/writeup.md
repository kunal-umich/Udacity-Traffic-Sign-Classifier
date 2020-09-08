# **Traffic Sign Recognition Project** 

## My Writeup

---


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://view5f1639b6.udacity-student-workspaces.com/view/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.html)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the len() and .shape method to get the data set summary.
The code for the same is available in the 2nd code cell in the Traffic_Sign_Classifier.ipynb notebook.

* The size of training set : 34799
* The size of test set : 12630
* The shape of a traffic sign image : (32, 32, 3)
* The number of unique classes/labels in the data set : 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 
Here are some random images from the training dataset displayed along with its classes:

![Training set image](https://view5f1639b6.udacity-student-workspaces.com/view/CarND-Traffic-Sign-Classifier-Project/Screenshots/Screenshot_1.png)

Here are bar charts showing the distribution of the images present in training, validation and test dataset different classes.
We can see that there is an uneven distribution, as there are more images present for certain classes which can make the CNN more biased towards these classes.

The code for the same is available in the 3-7 code cells in the Traffic_Sign_Classifier.ipynb notebook.

![Bar chart for training Dataset](https://view5f1639b6.udacity-student-workspaces.com/view/CarND-Traffic-Sign-Classifier-Project/Screenshots/Screenshot_2.png)

![Bar chart for validation Dataset](https://view5f1639b6.udacity-student-workspaces.com/view/CarND-Traffic-Sign-Classifier-Project/Screenshots/Screenshot_3.png)

![Bar chart for test Dataset](https://view5f1639b6.udacity-student-workspaces.com/view/CarND-Traffic-Sign-Classifier-Project/Screenshots/Screenshot_4.png)


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because for traffic sign classification, color doesn't play an important role and by using grayscale images we can improve the performance of the network as it has to consider only 1 channel compared to 3 in RGB images.

After grayscaling the image, I performed normalization on the images to make the pixel intensity distribution more even throughout the image. This makes the network converge faster by making the optimization more numerically efficient.

Here are examples of traffic sign images after grayscaling and normalizing.

![Processed images](https://view5f1639b6.udacity-student-workspaces.com/view/CarND-Traffic-Sign-Classifier-Project/Screenshots/Screenshot_6.png)

I also tried out of other techniques like augmenting datasets by rotating the images to create additional images and I also tried Histogram Equalization technique to increase the global contrast of the images, but since I couldn't achieve improvement in my training model using these techniques, I decided not to implement it in my final solution.

The code for the same is available in the 8-12 code cells in the Traffic_Sign_Classifier.ipynb notebook.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I chose Lenet Architecture for implementing my model.

My final model consisted of the following layers:

| Layer         		 |     Description	        					| 
|:---------------------: |:--------------------------------------------:| 
| Input                  | 32x32x1 grayscale image 						| 
| Convolution Layer 1 	 | 1x1 stride, Valid padding. Output = 28x28x6.	|
| Activation function    | RELU 										|
| Normalization function | Batch normalization  						|
| Max pooling          	 | 2x2 stride,  outputs 14x14x6  				|
| Convolution Layer 2 	 | 1x1 stride, Valid padding. Output = 10x10x16.|
| Activation function    | RELU 										|
| Normalization function | Batch normalization  						|
| Max pooling          	 | 2x2 stride,  outputs 5x5x16      			|
| Fully connected Layer 1| Input = 400. Output = 120.    				|
| Activation function    | RELU 										|
| Normalization function | Dropout 										|
| Fully connected Layer 2| Input = 120. Output = 84.    				|
| Activation function    | RELU 										|
| Normalization function | Dropout 										| 
| Fully connected Layer 3| Input = 84. Output = 43.     				|

 
The code for the model Architecture is available in the 13th cell in the Traffic_Sign_Classifier.ipynb notebook.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

1) To train the model, I first shuffled the images in the training set to ensure that the order of the images does affect the training process.

The code for the same is available in the 14th code cell in the Traffic_Sign_Classifier.ipynb notebook.

2) Then I set the Epoch and batch sizes for the training.

The code for the same is available in the 15th code cell in the Traffic_Sign_Classifier.ipynb notebook.

3) After that, I set my placeholder variables to supply images of different batch sizes, placholder for Batch normalization (To indicate  whether the model is in train or test phase) and probability placeholder for the Dropout function.

   I also on-hot encoded the labels using Tensorflow's built-in function.
   
   The code for the same is available in the 16th code cell in the Traffic_Sign_Classifier.ipynb notebook.
   
4) I set my learning rate and then called the Lenet Architecture that I defined for training. Then, I calculated the cross entropy between      the logits obtained from the Lenet Architecture and the one-hot encoded labels. The cross-entropy defines how different the logits          (predictions) are from the actual labels.

The mean of the cross-entropy from all the training dataset images is then calculated and stored as our loss which we need to minimize.
    
Then, I used Adam Optimizer to minimize our loss function.
    
The code for the same is available in the 17th code cell in the Traffic_Sign_Classifier.ipynb notebook.
   
5) The model is then trained on the training images and the accuracy of the model is calculated on the validation dataset.
    
   The code for the same is available in the 16-17 code cells in the Traffic_Sign_Classifier.ipynb notebook.


   Based on the validation accuracy the hyperparameters are tuned and final parameters that were chosen are as follows:
   
   Epochs = 20
   
   Batch size = 64
   
   Learn rate = 0.0005
   
   keep_prob = 0.6
    
   
#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 96.6%
* test set accuracy of 94.2%


1) I started by implementing the Lenet architecture from the lesson and got accuracy up to 89% with the following hyperparameter values:

   Epochs = 10
   
   Batch size = 128
   
   Learn rate = 0.001
   
2) Then, I preprocessed the image by converting it to grayscale and normalizing the image.

  As mentioned earlier I also tried some other image pre-processing methods like rotation and histogram equalization, but since I             couldn't achieve improvement in my training model using these techniques, I decided not to implement it in my final solution.
   
3) Then I added normalization techniques to prevent overfitting. I decided to apply dropouts, but then I came across several resources on the net, that dropouts don't work well on CNN, and should only be applied in the fully connected layers of the Lenet model.
So, I applied batch normalization after the convolution layers and dropout after the fully connected layers.

Reference : https://www.kdnuggets.com/2018/09/dropout-convolutional-networks.html

4) After that, I started tuning one parameter at a time to improve accuracy. 

  First I tuned the keep probability parameter between 0.4 to 0.6 and settled at 0.6.
  
  Then, I reduced the learning rate and found that I could achieve better accuracy.
  
  Then, I reduced the batch size, and even though it made the processing slower, I was able to achieve better accuracy.
  
  Finally, I tuned the number of Epochs to identify at which point the training accuracy reached a saturation point.
  
  Final parameters after tuning:    
   
  Epochs = 20
   
  Batch size = 64
   
  Learn rate = 0.0005
   
  keep_prob = 0.6
  
5) After tuning the parameters I was able to achieve a 96% accuracy in the validation set and 94% in the test set, which suggests the model performed decently for the traffic sign classification task.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Img 1: Beware of Ice](https://view5f1639b6.udacity-student-workspaces.com/view/CarND-Traffic-Sign-Classifier-Project/New_images/3.jpg)

![Img 2: Speed Limit](https://view5f1639b6.udacity-student-workspaces.com/view/CarND-Traffic-Sign-Classifier-Project/New_images/4.jpg)

![Img 3: Dangerous curve to left](https://view5f1639b6.udacity-student-workspaces.com/view/CarND-Traffic-Sign-Classifier-Project/New_images/1.jpg)

![Img 4: Dangerous curve to right](https://view5f1639b6.udacity-student-workspaces.com/view/CarND-Traffic-Sign-Classifier-Project/New_images/2.jpg)

![Img 5: No Vehicles](https://view5f1639b6.udacity-student-workspaces.com/view/CarND-Traffic-Sign-Classifier-Project/New_images/5.jpg)


Here are all the 5 images with after resizing to 32x32:

![New images](https://view5f1639b6.udacity-student-workspaces.com/view/CarND-Traffic-Sign-Classifier-Project/Screenshots/Screenshot_5.png)




- The first image (Beware of Ice) might be difficult to classify because the snowflake pattern on the sign is a complex pattern, which the network might find a difficult identity.

- The second image (Speed limit of 50) might be difficult to classify because the model has to identify numbers in the image and the image was taken at an angle which will make it more challenging.

- The third (Dangerous curve to left) and fourth image (Dangerous curve to the right) were selected because the signs are similar and it would be interesting to see if the model is able to distinguish between the left and right direction signs. Also, the fourth image is taken at an angle making it difficult to identify.

- The fifth image (No Vehicles) was chosen as it has no distinctive pattern, so it will be interesting to see it is correctly classified.



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:
(Note: The prediction are taken from indices of the top 5 probability obtained using softmax function)


Predictions:
Image 1 : [11 30 12 34 16]

Image 2 : [ 9  3 15 10 38] 

Image 3 : [19 23  9 30 21]

Image 4 : [10  9 19 42  7]

Image 5 : [15  2 13 12 38]


| Image(one-hot label)         |     Prediction(one-hot label)   				 	| 
|:---------------------:       |:--------------------------------------------------:| 
| Beware of Ice (30) 	       | Right-of-way at the next intersection (11)   		| 
| Speed Limit 50 km/h (2)	   | No passing (9)  								  	|
| Dangerous curve to left (19) | Dangerous curve to left (19)     					|
| Dangerous curve to left (20) | No passing for vehicles over 3.5 metric tons (10)	|
| No Vehicles (15)			   | No Vehicles (15)             						|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. Given that all the new images that were selected for testing had low frequency in the training dataset as can be seen from the bar chart shown earlier and also some of the images provided were taken at different angles, I feel the model performed decently. With data augmentation techniques and other image pre-processing methods, accuracy can be improved further. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 33rd cell of the Ipython notebook.


| Probability         	|     Prediction     	        					| 
|:---------------------:|:-------------------------------------------------:| 
|  1.0         			| Right-of-way at the next intersection (11) 		| 
|  0.44   				| No passing (9)     							  	|
|  1.0					| Dangerous curve to left (19)        				|
|  0.99      			| No passing for vehicles over 3.5 metric tons (10)	|
|  1				    | No Vehicles (15)             						|


- For the first image, the model is very sure that this is a 'Right-of-way at the next intersection' sign (probability of 1.0), and the model fails to identify the complex pattern of the snowflake.

- For the second image, the model lacking confidence and prdeicts that this is a 'No passing ' sign (probability of 0.44), and the model probably fails because of the angle at which the traffic sign is present.

- For the third image, the model is very sure that this is a 'Dangerous curve to left' sign (probability of 1.0), and it is correctly predicted.

- For the fourth image, the model is very sure that this is a 'No passing for vehicles over 3.5 metric tons' sign  (probability of 0.99), and again the model fails to identify the sign when the image is at an angle.

- For the fifth image, the model is very sure that this is a 'No Vehicles ' sign (probability of 1.0), and it is correctly predicted.


