#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 12:17:38 2018

@author: github.com/GustavZ
"""
import os
import tarfile
from six.moves import urllib
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.client import device_lib
print("版本:", tf.__version__)
print("型号:", device_lib.list_local_devices())
print(tf.test.is_gpu_available())

import socket
import yaml
import cv2
from stuff.helper import FPS2, WebcamVideoStream
from skimage import measure
from threading import Thread
import subprocess

## LOAD CONFIG PARAMS ##
if (os.path.isfile('config.yml')):
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
else:
    with open("config.sample.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

VIDEO_INPUT1    = cfg['video_input_rstp1']
VIDEO_INPUT2    = cfg['video_input_rstp2']
FPS_INTERVAL    = cfg['fps_interval']
ALPHA           = cfg['alpha']
MODEL_NAME      = cfg['model_name']
MODEL_PATH      = cfg['model_path']
IMG_SAVE_PATH1   = cfg['img_save_path1']
IMG_SAVE_PATH2   = cfg['img_save_path2']
DOWNLOAD_BASE   = cfg['download_base']
BBOX            = cfg['bbox']
MINAREA         = cfg['minArea']

# Hardcoded COCO_VOC Labels
LABEL_NAMES = np.asarray([
    '', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'])

def create_colormap(seg_map, type, video_input):
    """
    Takes A 2D array storing the segmentation labels.
    Returns A 2D array where each element is the color indexed
    by the corresponding element in the input label to the PASCAL color map.
    """
    # colormap = np.zeros((256, 3), dtype=int)
    # ind = np.arange(256, dtype=int)
    #
    # for shift in reversed(range(8)):
    #     for channel in range(3):
    #         colormap[:, channel] |= ((ind >> channel) & 1) << shift
    #     ind >>= 3
    #
    # # if type == "person":
    # #     colormap[:, 0] = 136
    # #     colormap[:, 1] = 246
    # #     colormap[:, 2] = 136

    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)


    # colored_seg_map[i, j] = colormap[seg_map[i, j]]
    # Generate the PASCAL colormap for all labels not set to 'person'
    # for shift in reversed(range(8)):
    #     for channel in range(3):
    #
    #         if type == "person":
    #             # Set a specific color for 'person' type
    #             person_color = [136, 246, 136]  # Blue color
    #             colormap[15] = person_color  # Assuming the label for 'person' is 15
    #
    #         if type != "person" or (type == "person" and channel == 2):  # Skip if 'person' type and not blue channel
    #             colormap[:, channel] |= ((ind >> channel) & 1) << shift
    #     ind >>= 3

    # Apply the colormap to the segmentation map
    orange_color = [255, 165, 0]

    height, width = seg_map.shape
    colored_seg_map = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            if j > 4 * width // 5 and video_input == VIDEO_INPUT2:
                colormap[15] = [255, 165, 0]
            else:
                colormap[15] = [136, 246, 136]
            colored_seg_map[i, j] = colormap[seg_map[i, j]]
            # else:
            #     colored_seg_map[i, j] = orange_color

    return colored_seg_map

def file_clean(path):
    # 系统启动时，清空文件夹内所有文件
    img_files_list = os.listdir(path)  # 读入文件夹
    for file in img_files_list:
        oldpath = os.path.join(path, file)
        os.remove(oldpath)
def img_save(img, path, loop_counter):
    loop_counter = str(loop_counter).rjust(6, '0')
    cv2.imwrite(path + '/seg' + str(loop_counter) + '.jpg', img)
    # 控制文件数量
    img_files_list = os.listdir(path)  # 读入文件夹
    num_img = len(img_files_list) # 统计文件数量
    if num_img > 100:
        oldpath = os.path.join(path, img_files_list[0])
        os.remove(oldpath)


# Download Model from TF-deeplab's Model Zoo
def download_model():
    model_file = MODEL_NAME + '.tar.gz'
    if not os.path.isfile(MODEL_PATH):
        print('> Model not found. Downloading it now.')
        opener = urllib.request.URLopener()
        opener.retrieve(DOWNLOAD_BASE + model_file, model_file)
        tar_file = tarfile.open(model_file)
        for file in tar_file.getmembers():
          file_name = os.path.basename(file.name)
          if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd() + '/models/')
        os.remove(os.getcwd() + '/' + model_file)
    else:
        print('> Model found. Proceed.')


# Visualize Text on OpenCV Image
def vis_text(image,string,pos):
    cv2.putText(image,string,(pos),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

# Load frozen Model
def load_frozenmodel():
    print('> Loading frozen model into memory')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        seg_graph_def = tf.GraphDef()
        with tf.gfile.GFile(MODEL_PATH, 'rb') as fid:
            serialized_graph = fid.read()
            seg_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(seg_graph_def, name='')
    return detection_graph

# 有点卡，尝试写成多线程
def segmentation(detection_graph, label_names, input):
    # fixed input sizes as model needs resize either way
    vs = WebcamVideoStream(input, 640,480).start()

    resize_ratio = 1.0 * 513 / max(vs.real_width,vs.real_height)
    target_size = (int(resize_ratio * vs.real_width), int(resize_ratio * vs.real_height))

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    fps = FPS2(FPS_INTERVAL).start()
    print("> Starting Segmentaion")

    cv2.namedWindow('segmentation1', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO);
    cv2.resizeWindow('segmentation1', 960, 540)

    loop_counter = 0

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            while vs.isActive():
                image = cv2.resize(vs.read(),target_size)

                batch_seg_map = sess.run('SemanticPredictions:0',
                                feed_dict={'ImageTensor:0': [cv2.cvtColor(image, cv2.COLOR_BGR2RGB)]})

                # visualization
                seg_map = batch_seg_map[0]

                vis_text(image,"fps: {}".format(fps.fps_local()),(10,30))

                # boxes (ymin, xmin, ymax, xmax)
                region = None
                if BBOX:
                    map_labeled = measure.label(seg_map, connectivity=1)
                    for region in measure.regionprops(map_labeled):
                        if region.area > MINAREA:
                            box = region.bbox
                            p1 = (box[1], box[0])
                            p2 = (box[3], box[2])
                            if (label_names[seg_map[tuple(region.coords[0])]] == "person"):

                                cv2.rectangle(image, p1, p2, (255,255,255), 2)
                                vis_text(image,label_names[seg_map[tuple(region.coords[0])]],(p1[0],p1[1]-10))

                if region is not None:
                    seg_image = create_colormap(seg_map, label_names[seg_map[tuple(region.coords[0])]],
                                                VIDEO_INPUT1).astype(np.uint8)
                else:
                    continue
                cv2.addWeighted(seg_image, ALPHA, image, 1 - ALPHA, 0, image)

                cv2.imshow('segmentation1',image)
                #cv2.moveWindow('segmentation1', 0, 0)
                img_save(image, IMG_SAVE_PATH2, loop_counter)
                loop_counter += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                fps.update()
    fps.stop()
    vs.stop()
    cv2.destroyAllWindows()

def segmentation2(detection_graph, label_names, input):
    # fixed input sizes as model needs resize either way
    vs = WebcamVideoStream(input, 640,480).start()

    resize_ratio = 1.0 * 513 / max(vs.real_width,vs.real_height)
    target_size = (int(resize_ratio * vs.real_width), int(resize_ratio * vs.real_height))

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    fps = FPS2(FPS_INTERVAL).start()
    print("> Starting Segmentaion")

    cv2.namedWindow('segmentation2', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO);
    cv2.resizeWindow('segmentation2', 960, 540)

    loop_counter = 0
    #pipe_init = ffmpeg_init()

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            while vs.isActive():
                image = cv2.resize(vs.read(),target_size)

                batch_seg_map = sess.run('SemanticPredictions:0',
                                feed_dict={'ImageTensor:0': [cv2.cvtColor(image, cv2.COLOR_BGR2RGB)]})

                # visualization
                seg_map = batch_seg_map[0]

                vis_text(image,"fps: {}".format(fps.fps_local()),(10,30))

                # boxes (ymin, xmin, ymax, xmax)
                region = None
                if BBOX:
                    map_labeled = measure.label(seg_map, connectivity=1)
                    for region in measure.regionprops(map_labeled):
                        if region.area > MINAREA:
                            box = region.bbox
                            p1 = (box[1], box[0])
                            p2 = (box[3], box[2])
                            if (label_names[seg_map[tuple(region.coords[0])]] == "person"):

                                cv2.rectangle(image, p1, p2, (255,255,255), 2)
                                vis_text(image,label_names[seg_map[tuple(region.coords[0])]],(p1[0],p1[1]-10))

                # rtsp_push(pipe_init, image)

                if region is not None:
                    seg_image = create_colormap(seg_map, label_names[seg_map[tuple(region.coords[0])]],
                                                VIDEO_INPUT2).astype(np.uint8)
                else:
                    continue

                cv2.addWeighted(seg_image, ALPHA, image, 1 - ALPHA, 0, image)

                cv2.imshow('segmentation2', image)

                img_save(image, IMG_SAVE_PATH1, loop_counter)
                loop_counter += 1
                #cv2.moveWindow('segmentation2', 0, 0)
                #right_image = cv2.imread('C://Users//34748//Desktop//1700642303092.png')
                #cv2.imshow('right_image', right_image)
                #cv2.moveWindow('right_image', 980, 0)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                fps.update()
    fps.stop()
    vs.stop()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    download_model()
    graph = load_frozenmodel()

    file_clean(IMG_SAVE_PATH1)
    file_clean(IMG_SAVE_PATH2)

    t1 = Thread(target=segmentation, args=(graph, LABEL_NAMES, VIDEO_INPUT1))  # 64
    t2 = Thread(target=segmentation2, args=(graph, LABEL_NAMES, VIDEO_INPUT2)) # 14

    t1.start()
    t2.start()

    # 等待所有线程执行完毕
    t1.join()  # join() 等待线程终止，要不然一直挂起
    t2.join()