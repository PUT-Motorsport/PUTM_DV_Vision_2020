# PUT Motorsport Driverless Vision
Computer vision repository for cones detection methods for Formula Student Driverless competition.

### Resources
1. [MIT Formula Student Driverless Dataset](https://storage.cloud.google.com/mit-driverless-open-source/YOLO_Dataset.zip?authuser=1)
2. CSV files with annotations:
  - [all labels csv](https://storage.cloud.google.com/mit-driverless-open-source/yolov3-training/all.csv?authuser=1)
  - [train labels csv](https://storage.cloud.google.com/mit-driverless-open-source/yolov3-training/train.csv?authuser=1)
  - [validate labels csv](https://storage.cloud.google.com/mit-driverless-open-source/yolov3-training/validate.csv?authuser=1)
3. [YOLOv4 initial weights](https://drive.google.com/file/d/1JKF-bdIklxOOVy-2Cr5qdvjgGpmGfcbp/view)


### YOLO dataset format
```python
python parsers/csv2yolo_parser.py --csv_path <PATH TO CSV FILES DIRECTORY> --dataset_path <PATH TO DATASET DIRECTORY>
```

### COCO dataset format
```python
python parsers/csv2coco_parser.py --file_path <PATH TO DIRECTORY FOR ANNOTATION FILES> --csv_path <PATH TO CSV FILES DIRECTORY>
```