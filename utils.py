import cv2 as cv
import numpy as np
import pandas as pd
import requests
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans

GDRIVE_KAZE_100_NORM_ID = '1bu58-ycXTjFPhFatXMnA8E_TP84Hux5F'
GDRIVE_SIFT_100_NORM_ID = '1EuWxaW4OqhBt94eb1TsEXQgvoDRD4CPk'
GDRIVE_SUFT_100_NORM_ID = '1W1YIXWwNYNEV_qwl74FIxu58q5f9PAqi'


def feature_extractor(img_path, detector, ret_both=False):
    img = cv.imread(img_path, 0)
    kp, des = detector.detectAndCompute(img, None)
    if ret_both:
        return kp, des
    return des


def getX(names, detector=cv.xfeatures2d.SURF_create()):
    outs = [feature_extractor(o, detector) for o in names]
    outs1 = [o for o in outs if o is not None]  # remove some invalid output
    X = np.concatenate(outs1, axis=0)
    return X


class Vocab:
    def __init__(self, suft_npy, sift_npy, kaze_npy):
        # make sure centers and detectors have same order of suft, sift and kaze
        self.centers = [np.load(suft_npy),
                        np.load(sift_npy),
                        np.load(kaze_npy)]

        self.detectors = [cv.xfeatures2d.SURF_create(),
                          cv.xfeatures2d.SIFT_create(),
                          cv.KAZE_create()]

    def query_id(self, img):
        """
        img: image path
        return list of ((x,y), id)
        """
        start_pos = 0
        out_kp, out_id = [], []
        for i in range(len(self.centers)):
            detector = self.detectors[i]
            center = self.centers[i]
            kps, des = feature_extractor(img, detector, ret_both=True)
            # normalize des
            des = preprocessing.normalize(des, norm="l2")
            ids = [start_pos + self.get_min_pos(center, o) for o in des]
            # check if outs is None
            start_pos += center.shape[0]

            out_kp.extend([o.pt for o in kps])
            out_id.extend(ids)

        return zip(out_kp, out_id)

    def get_min_pos(self, centers, vec):
        tmp = centers - vec  # broadcast
        distances = np.sum(tmp ** 2, axis=1)
        return np.argmin(distances)


def download_file_from_google_drive(gid, dest):
    """
    credit: https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive
    :param gid: google file id
    :param dest: file path to save
    :return:
    """

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, dest):
        CHUNK_SIZE = 32768

        with open(dest, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': gid}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': gid, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, dest)
