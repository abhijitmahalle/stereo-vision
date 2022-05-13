# Stereo Vision 
This repository contains code to find disparity and depth maps of two image sequences of a given subject using the concepts of epipolar geometry, Fundamental Matrix, Essential Matrix and its decomposition to get Rotation and Translation matrices, epipolar lines, rectification, and correspondence using SSD. Note that no OpenCV inbuilt function was used while implementing these concepts and they were coded from scratch.  
### Instructions to run the file:
```sh
python project3.py

```
Results can be found in the "results" directory


![](results/curule/unrectified_epilines.png)  
                            

![](results/curule/rectified_epilines.png)


<img src =https://github.com/abhijitmahalle/stereo_vision/blob/master/results/curule/disparity_heatmap.png width = 49% height = 50% /> <img src =https://github.com/abhijitmahalle/stereo_vision/blob/master/results/curule/depth_heatmap.png width = 49% height = 50% />


