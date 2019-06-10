from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import requests
from gensim import matutils
from gensim.corpora import Dictionary
from gensim.models import HdpModel
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

GDRIVE_KAZE_100_NORM_ID = '1bu58-ycXTjFPhFatXMnA8E_TP84Hux5F'
GDRIVE_SIFT_100_NORM_ID = '1EuWxaW4OqhBt94eb1TsEXQgvoDRD4CPk'
GDRIVE_SUFT_100_NORM_ID = '1W1YIXWwNYNEV_qwl74FIxu58q5f9PAqi'

# docs-data gid
GDRIVE_DOCS_DATA = '14Wf72-hsRwkMk3xGlg7fR1vwpKE4B24L'

# models
GDRIVE_BOW_SVM_CLF = '1_cZBoqAjKwh_WkQaRZnqNyzSberOCdAP'
GDRIVE_HDP_SVM_CLF = '1vEp-CawLi50teyKwB9C-Rl0ZFLabzV2L'


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


def build_docs(vocab, fname_list, labels):
    """
    Build document from image files
    :param vocab:
    :param fname_list: image file list
    :param labels: labels of images in list (must same size as fname_list)
    :return: [[ids]] one for each file
    """
    docs, targets = [], []
    for fname, label in zip(fname_list, labels):
        try:
            out = list(vocab.query_id(fname))
            ids = [o for _, o in out]
            docs.append(ids)
            targets.append(label)
        except:
            pass  # some error file
    return docs, np.array(targets)


def build_tf_idf(vocab, fname_train, labels_train):
    """
    make tf-idf of images features
    :param vocab:
    :param fname_train: files
    :param labels_train: labels
    :return: [vector tf-idf], [labels]
    """
    train_docs, train_targets = build_docs(vocab, fname_train, labels_train)
    train_docs_str = [' '.join([str(a) for a in o]) for o in train_docs]

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train_docs_str)

    tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)

    return X_train_tf, train_targets


def build_tf_idf_docs(docs, targets):
    train_docs_str = [' '.join([str(a) for a in o]) for o in docs]

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train_docs_str)

    tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)

    return X_train_tf, targets


def pre_processing(vocab, fname_train, labels_train, fname_test, labels_test):
    # train
    X_train_tf, train_targets = build_tf_idf(vocab, fname_train, labels_train)
    # test
    X_test_tf, test_targets = build_tf_idf(vocab, fname_test, labels_test)
    return X_train_tf, train_targets, X_test_tf, test_targets


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


def download_and_cache(gid):
    """
    Download and cache Google Drive file (public) in /tmp
    :param gid:
    :return:
    """
    cache_file_path = Path('/tmp/{}'.format(gid))
    if not cache_file_path.is_file():
        download_file_from_google_drive(gid, cache_file_path.absolute())
    return cache_file_path


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    credit: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        # print('Confusion matrix, without normalization')
        pass

    # print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def build_hdp_vec(docs, targets, dct=None, hdp=None):
    docs = [[str(o) for o in one] for one in docs]

    if dct is None:  # train set
        dct = Dictionary(docs)
        for one in docs:
            dct.add_documents([[str(o) for o in one]])

    copus = [dct.doc2bow(o) for o in docs]
    if hdp is None:  # train
        hdp = HdpModel(copus, dct)

    v = [hdp[o] for o in copus]
    v_d = matutils.corpus2dense(v, num_terms=len(dct.token2id)).T

    return copus, v_d, targets, dct, hdp


def balanced_subsample(x, y, subsample_size=1.0):
    """
    credit: https://stackoverflow.com/questions/23455728/scikit-learn-balanced-subsampling
    """
    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems * subsample_size)

    xs = []
    ys = []

    for ci, this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs, ys
