# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.
My pipeline consists of 5 steps. 
1. Convert the image to grayscale and apply Gaussian smoothing 
2. Apply Canny edge detection  
3. Mask the region of interest
3. Apply Hough transformation to find the lines
4. Draw the lane lines on the original image

In order to draw a single line on the left and right lanes, I modified the draw_lines() function with following steps.
1. Identify and separate left lines and right lines based on their slope
2. Calculate average slope for both left line and right line in a single frame 
3. Filter the slope to reduce impact of noise
4. Find the top and bottom point in detetected lane lines and extend the line with average slope been found
5. Record the current average slope and used in the next frame.

For the chanllenge project:
I noticed that the previous pipeline didn't work as there is a sudden change of light intensity, shadow of trees and the frontal part of the car in the video. To deal with the light and shadow issues, i convert the image to HLS color space and use S channel only instead of grayscale. However, it does not works perfectly, sometimes lane lines are not detected. Therefore, when no lines are detected, i use the previous detected lines and slopes. 

Note: 

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 
1. For the challenge project, if many lines are actually not been detected, this method will not work. 
2. It can only detected straight lines (where i consider curved lines to be straight in certain distance). If there is a 90 degree turnning it will be problematic
3. Not really robust enough..




### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...
1. Dynamic thersholding
2. Implement some other algorithm and combine them together to create redundancy, could be more robust.


