# DASbench-data-preparation
Using unsupervised deep learning model to process the raw DAS data from Groß Schönebeck survey for DASbench


## How to perform the workflow?
  1) Run the [downloading script](https://github.com/cuiyang512/DASbench-data-preparation/blob/main/Get_data.sh) to get raw profiles. It takes a while to dowanload (~ 1GB each profile).
  2) Preprocess and split the data with the [notebook script](https://github.com/cuiyang512/DASbench-data-preparation/blob/main/Data_preprocessing.ipynb). Here, I only release the script to preprocess single raw data. Therefore, you need to change the file name very time when you split different raw data. However, I will put all the preprocessing workflow in an iteration later.
  3) Process the samples generated above using the trained model from our recent publications. Run the [notebook script](https://github.com/cuiyang512/DASbench-data-preparation/blob/main/USL_dataset_processing.ipynb).
