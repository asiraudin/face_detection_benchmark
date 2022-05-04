import argparse
from face_detectors import *
from predictions import make_predictions
from evaluation import evaluation
import os
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--detector')
    parser.add_argument('-i', '--image_dir')
    parser.add_argument('-p', '--pred_dir')
    parser.add_argument('-g', '--gt_dir', default='ground_truth')

    args = parser.parse_args()
    wd = os.getcwd()

    if args.detector == 'haar':
        face_detector = OpenCVHaarFaceDetector()
    elif args.detector == 'caffe':
        face_detector = OpenCVCaffeeFaceDetector()

    start_pred = time.time()

    make_predictions(face_detector, os.path.join(wd, args.image_dir), os.path.join(wd, args.pred_dir))

    end_pred = time.time()

    delta = end_pred - start_pred
    print("==================== Prediction time ====================")
    print(f"{delta//60} minutes, {round(delta%60)} seconds.")
    print("=========================================================")

    start_eval = time.time()

    evaluation(os.path.join(wd, args.pred_dir), os.path.join(wd, args.gt_dir))

    end_eval = time.time()
    delta = end_eval - start_eval
    print(f"Evaluation time: {delta//60} minutes, {round(delta%60)} seconds")
