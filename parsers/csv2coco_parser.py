import json
import click
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


def add_info_dict():
    info_dict = {
      "description": "MIT CVC-YOLOv3 Dataset with Formula Student Standard ", 
      "url": "https://github.com/cv-core/MIT-Driverless-CV-TrainingInfra", 
      "version": "1.0", 
      "year": 2020, 
      "contributor": "Mateusz Piechocki", 
      "date_created": "2020/09/19"
    }
    return info_dict


def add_categories_list():
    categories_list = [
      {
      "supercategory": "objects", 
      "id": 0, 
      "name": "cone"
      }
    ]
    return categories_list


def add_image_dict(filename, height, width, image_id):
    image_dict = {
      "license": 0, 
      "file_name": f"dataset/{filename}", 
      "coco_url": "", 
      "height": height, 
      "width": width, 
      "date_captured": "", 
      "flickr_url": "", 
      "id": image_id
    }
    return image_dict


def add_annotation_dict(bbox, image_id, id):
    x, y, h, w = bbox
    annotation_dict = {
      "id": id,
      "segmentation": [], 
      "area": h*w, 
      "iscrowd": 0, 
      "image_id": image_id, 
      "bbox": [x, y, w, h], 
      "category_id": 0, 
    }
    
    return annotation_dict


def parse_csv2coco(file_path: str, csv_path: str):
    df = pd.read_csv(Path(csv_path), header=1)
    images_list = []
    annotations_list = []

    annotation_index = 0

    for idx, row in tqdm(enumerate(df.iterrows())):
        filename = row[1][0]
        width = int(row[1][2])
        height = int(row[1][3])
        bboxes = [bbox.replace('[', '').replace(']', '').split(', ') for bbox in row[1][5:].values if not pd.isna(bbox)]
        bboxes = [[int(coord) for coord in bbox] for bbox in bboxes]

        image_dict = add_image_dict(filename, height, width, idx)
        images_list.append(image_dict)

        for bbox in bboxes:
            annotation_dict = add_annotation_dict(bbox, idx, annotation_index)
            annotations_list.append(annotation_dict)
            annotation_index+=1
            
    info_dict = add_info_dict()
    categories_list = add_categories_list()
    
    dataset_dict = {
      'type': 'instances', 
      'info': info_dict, 
      'categories': categories_list, 
      'images': images_list, 
      'annotations': annotations_list, 
    }
    
    with open(Path(file_path), 'w') as file:
        json.dump(dataset_dict, file)


@click.command()
@click.option('--file_path', help='Summary json files path', required=True)
@click.option('--csv_path', help='CSV annotation files path', required=True)
def main(file_path: str, csv_path: str):
	parse_csv2coco(file_path+'/train.json', csv_path+'/train.csv')
	parse_csv2coco(file_path+'/val.json', csv_path+'/validate.csv')


if __name__ == '__main__':
    main()
