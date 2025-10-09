# Steren 
A repository for a codebase to analyse planetary surfaces using artificial intelligence

<p align="center">
  <img src="steren.png" alt="Steren Logo" width="30%">
</p>

### Model Training
Features are best annotated in GIS software such as [QGIS](https://qgis.org/).
Once the annotations are curated the GeoCOCO Pyhton package can be used to convert the annotations to a coco format dataset.
This COCO dataset can then be viewed using fiftyone or converted to other formats such as YOLO format, using the ultralytics api. A yolo model for example can then be trained for object detection
\

### Model Inference
Since satellite the images are very large and continous, most object detections models such as the yolo series by ultralytics require the imaged to be sliced. The sahi package can be used for this.

### Model Performance Measurement
Often the model performance measurements are availible during training however in the case of repurposing models on new data for which there exists accurate annotations, the model performance can be quantified.
The model outputs detections with a given confidence. Normally the confidence is thresholded to obtain concrete set of predictions. The model performance is not one dimensional precision the proportion of objects identified are that are and recall the proportion of annotated objects that where detected are independant dimensions of performance. 

In the case of craters to study recall, I focus on the annotated objects for each object I find if it is detected and find the confidence then I plot the graph of object size against confidence. For precision I look at each detected object and determine if it is in the annotated data 
