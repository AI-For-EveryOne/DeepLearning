import numpy as np
import boto3
import json
import cv2
import os
import io
import matplotlib.pyplot as plt
from PIL import Image

import sys
sys.setrecursionlimit(5000)


SIMILARITY_THRESHOLD = 70

def img_to_byte(file_name):
    with open(file_name, 'rb') as img:
        img_byte = img.read()
        return img_byte


def bounding_box_positions(imageHeight, imageWidth, box, rotation): 
    left = 0
    top = 0

    if rotation == 'ROTATE_0':
        left = imageWidth * box['Left']
        top = imageHeight * box['Top']
    
    if rotation == 'ROTATE_90':
        left = imageHeight * (1 - (box['Top'] + box['Height']))
        top = imageWidth * box['Left']

    if rotation == 'ROTATE_180':
        left = imageWidth - (imageWidth * (box['Left'] + box['Width']))
        top = imageHeight * (1 - (box['Top'] + box['Height']))

    if rotation == 'ROTATE_270':
        left = imageHeight * box['Top']
        top = imageWidth * (1- box['Left'] - box['Width'] )

    print('Left: ' + '{0:.0f}'.format(left))
    print('Top: ' + '{0:.0f}'.format(top))
    print('Face Width: ' + '{0:.0f}'.format(imageWidth * box['Width']))
    print('Face Height: ' + '{0:.0f}'.format(imageHeight * box['Height']))

    left = int('{0:.0f}'.format(left))
    top = int('{0:.0f}'.format(top))
    width = int('{0:.0f}'.format(imageWidth * box['Width']))
    height = int('{0:.0f}'.format(imageHeight * box['Height']))

    return left, top, width, height


def main():
    #boto3のclientを作成し、rekognitionとリージョンを指定
    client = boto3.client('rekognition', 'ap-northeast-1')

    # 画像読み込み
    # 全体の写真
    target_file = 'DataSet/イメージ.jpeg'
    target_byte = img_to_byte(target_file)

    # 特定したい人の写真
    source_file = 'DataSet/イメージ 2.jpeg'
    source_byte = img_to_byte(source_file)

    response = client.compare_faces(
        SourceImage = {
            'Bytes':source_byte
        },
        TargetImage = {
            'Bytes':target_byte
        },
        SimilarityThreshold = SIMILARITY_THRESHOLD
    )

    response = json.dumps(response, ensure_ascii=False, indent=4)
    print(response)
    response = json.loads(response)

    if 'FaceMatches' in response:
        box = response['FaceMatches'][0]['Face']['BoundingBox']

        image = Image.open(open(target_file, 'rb'))
        width, height = image.size

        stream = io.BytesIO()
        if 'exif' in image.info:
            exif = image.info['exif']
            image.save(stream, format=image.format, exif=exif)
        else:
            image.save(stream, format=image.format)
        image_binary = stream.getvalue()

        img = cv2.imread(target_file)
        result_img = 'result/result.jpg'
        x, y, w, h = bounding_box_positions(height, width, box , 'ROTATE_0')
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # 矩形画像作成
        cv2.imwrite(result_img, img)
            
        #出力画像の表示
        plt.show(plt.imshow(np.asarray(Image.open(result_img))))


if __name__ == '__main__':
    main()
