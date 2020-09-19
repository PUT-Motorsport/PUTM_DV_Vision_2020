import os
import cv2
import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from typing import List


def add_csv_bbox_to_img(img: np.ndarray, bbox: List):
    bbox = np.array(bbox)
    pt1 = bbox[:2]
    pt2 = bbox[:2] + bbox[2:][::-1]
    cv2.rectangle(img, tuple(pt1), tuple(pt2), (0,0,255), 2)


def add_yolo_bbox_to_img(img: np.ndarray, bbox: List):
    bbox = np.array(bbox)
    height, width = img.shape[:2]
    pt1 = np.multiply(np.subtract(bbox[:2], bbox[2:]/2), [width, height]).astype(np.int32)
    pt2 = np.multiply(np.add(bbox[:2], bbox[2:]/2), [width, height]).astype(np.int32)
    cv2.rectangle(img, tuple(pt1), tuple(pt2), (255,0,0), 2)


def create_summary_txt_file(file_path: str, csv_path: str, dataset_path: str):
    df = pd.read_csv(Path(csv_path), header=1)
    with open(Path(file_path), 'w') as file:
        for filename in tqdm(df['Name'].values):
            file_path = Path(f'{dataset_path}/{filename}')
            if os.path.exists(file_path):
                file.write(f'{file_path}\n')


def normalize_bbox(bbox: List, height: int, width: int) -> List:
    x, y, h, w = bbox
    x_norm = (x + w/2) / width
    y_norm = (y + h/2) / height
    w_norm = w / width
    h_norm = h / height
    return [x_norm, y_norm, w_norm, h_norm]


def parse_csv2yolo(csv_path: str, dataset_path: str):
    df = pd.read_csv(Path(csv_path), header=1)
    for row in tqdm(df.iterrows()):
        filename = row[1][0].replace('jpg', 'txt')
        width = int(row[1][2])
        height = int(row[1][3])
        bboxes = [bbox.replace('[', '').replace(']', '').split(', ') for bbox in row[1][5:].values if not pd.isna(bbox)]
        bboxes = [[int(coord) for coord in bbox] for bbox in bboxes]
        bboxes = list(map(lambda bbox: normalize_bbox(bbox, height, width), bboxes))
        with open(Path(f'{dataset_path}/{filename}'), 'w') as txt_file:
            for bbox in bboxes:
                txt_file.write(f'0 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n')


@click.command()
@click.option('--csv_path', help='CSV annotation files path', required=True)
@click.option('--dataset_path', help='Image dataset directory', required=True)
def main(csv_path: str, dataset_path: str):
    all_csv_path = Path(f'{csv_path}/all.csv')
    train_csv_path = Path(f'{csv_path}/train.csv')
    val_csv_path = Path(f'{csv_path}/validate.csv')

    parse_csv2yolo(all_csv_path, dataset_path)
    create_summary_txt_file('train.txt', train_csv_path, dataset_path)
    create_summary_txt_file('val.txt', val_csv_path, dataset_path)


if __name__ == "__main__":
    main()