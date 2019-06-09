import numpy as np
import cv2 as cv
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing

def feature_extractor(img_path, detector, ret_both=False):
    img = cv.imread(img_path,0)
    kp, des = detector.detectAndCompute(img,None)
    if ret_both:
        return kp, des
    return des

def getX(names, detector = cv.xfeatures2d.SURF_create()):
    outs = [feature_extractor(o, detector) for o in names]
    outs1 = [o for o in outs if o is not None] # remove some invalid output
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

        
    def query_id(self,img):
        """
        img: image path
        return list of ((x,y), id)
        """
        start_pos = 0
        out_kp, out_id = [], []
        for i in range(len(self.centers)):
            detector = self.detectors[i]
            center = self.centers[i]
            kps,des = feature_extractor(img, detector, ret_both=True)
            # normalize des
            des = preprocessing.normalize(des, norm="l2")
            ids = [start_pos + self.get_min_pos(center, o) for o in des]
            # check if outs is None
            start_pos += center.shape[0]
            
            out_kp.extend([o.pt for o in kps])
            out_id.extend(ids)

        return zip(out_kp, out_id)
    
    def get_min_pos(self, centers, vec):
        tmp = centers - vec # broadcast
        distances = np.sum(tmp ** 2, axis=1)
        return np.argmin(distances)
