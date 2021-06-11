import torch
import cv2
import json
import pandas as pd
import copy
import random
import numpy as np
import pickle
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from skimage import transform
from skimage.transform import rotate, AffineTransform
from skimage.util import random_noise
from skimage.filters import gaussian
from scipy import ndimage
import random
import os


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, jsonPath = '../gt_train.json', missingPath = '../images/missing_images.csv', shuffle = True, base = '../images/'):

        missing = (pd.read_csv(missingPath, names = ['img', 'src'])['img']).astype(str)
        jsonString = open(jsonPath, 'r').readlines()[0]
        data = json.loads(jsonString)
        
        data = {x: data[x] for x in data if not sum(missing.isin([x]))}
        self.data = data
        keys = list(self.data.keys())
        random.shuffle(keys)
        if shuffle:
            self.data = {x: self.data[x] for x in keys}
        self.imSize = (224, 224)
        self.root = base
        self.keys = list(self.data.keys())
        self.save = False

        print("Train set ready with {} samples".format(len(self)))

        self.transformations = [self.nothing, self.noise_img, self.bright]

        return None

    def nothing(self, img):
        return img
    def rotate_img(self, img):
        return rotate(img, angle=random.randint(1, 90))

    def noise_img(self, img):
        return random_noise(img, var=0.1**2)

    def bright(self, img):
        return img + (100/255)

    def __len__(self):
        return len(self.data.keys())

    def loadImage(self, path, normalize = True):
        img = cv2.imread(path, 1) #0 means grayscale. Maybe it's a TODO: use grayscale so the model isnt biased
        #assert type(img) != type(None)
        if type(img) == type(None):
            0/0
        img = cv2.resize(img, self.imSize)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2,0,1)
        img = img/255

        transformations = random.sample(self.transformations, random.randint(0, len(self.transformations)))
        for tr in transformations:
            img = tr(img)
        return img

    def __getitem__(self, index):

        file_ = self.keys[index]
        path = self.root + file_

        try:
            image = self.loadImage(path)
            if self.save:
                file_ = file_.split('.')
                file_[-1] = 'npz'
                file_ = '.'.join(file_)
                path = self.root + file_
                np.savez(path, image)
                
            return image, self.data[self.keys[index]]
        except ZeroDivisionError:
            return self[random.randint(0, len(self)-1)]

class TestSet(torch.utils.data.Dataset):
    def __init__(self, jsonPath = '../gt_test.json', missingPath = '../images/missing_images.csv', shuffle = True, base = '../images/'):

        missing = (pd.read_csv(missingPath, names = ['img', 'src'])['img']).astype(str)
        jsonString = open(jsonPath, 'r').readlines()[0]
        data = json.loads(jsonString)
        
        data = {x: data[x] for x in data if not sum(missing.isin([x]))}
        self.data = data
        keys = list(self.data.keys())
        random.shuffle(keys)
        if shuffle:
            self.data = {x: self.data[x] for x in keys}
        self.imSize = (224, 224)
        self.root = base
        self.keys = list(self.data.keys())
        self.save = False
        
        print("Train set ready with {} samples".format(len(self)))

        self.transformations = [self.nothing, self.noise_img, self.bright]

        return None

    def nothing(self, img):
        return img
    def rotate_img(self, img):
        return rotate(img, angle=random.randint(1, 90))

    def noise_img(self, img):
        return random_noise(img, var=0.1**2)

    def bright(self, img):
        return img + (100/255)

    def __len__(self):
        return len(self.data.keys())

    def loadImage(self, path, normalize = True):
        img = cv2.imread(path, 1) #0 means grayscale. Maybe it's a TODO: use grayscale so the model isnt biased
        #assert type(img) != type(None)
        if type(img) == type(None): #or os.stat(path).st_size == 2051: TODO: Activar axò en cas d'experiment
            0/0
        img = cv2.resize(img, self.imSize)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2,0,1)
        img = img/255

        transformations = random.sample(self.transformations, random.randint(0, len(self.transformations)))
        for tr in transformations:
            img = tr(img)
        return img

    def __getitem__(self, index):

        file_ = self.keys[index]
        path = self.root + file_

        try:
            image = self.loadImage(path)
            if self.save:
                file_ = file_.split('.')
                file_[-1] = 'npz'
                file_ = '.'.join(file_)
                path = self.root + file_
                np.savez(path, image)
                
            return image, self.data[self.keys[index]]
        except ZeroDivisionError:
            return self[random.randint(0, len(self)-1)]

class FastTrainSet(TrainSet):
    def __init__(self, jsonPath = '../gt_train.json', missingPath = '../images/missing_images.csv', shuffle = True, base = '../images/'):
        missing = (pd.read_csv(missingPath, names = ['img', 'src'])['img']).astype(str)
        jsonString = open(jsonPath, 'r').readlines()[0]
        data = json.loads(jsonString)
        
        data = {x: data[x] for x in data if not sum(missing.isin([x]))}
        self.data = data
        keys = list(self.data.keys())
        random.shuffle(keys)
        if shuffle:
            self.data = {x: self.data[x] for x in keys}
        self.imSize = (224, 224)
        self.root = base
        self.keys = list(self.data.keys())
        self.save = False
        

        print("Test set ready with {} samples".format(len(self)))
    def __getitem__(self, index):
        file_ = self.keys[index].split('.')
        file_[-1] = 'npz'
        file_ = '.'.join(file_)
        path = self.root + file_
        try:
            return np.load(path)['arr_0'], self.data[self.keys[index]]
        except FileNotFoundError:
            return self[random.randint(0, len(self)-1)]

class FastTestSet(TestSet):
    def __init__(self, jsonPath = '../gt_test.json', missingPath = '../images/missing_images.csv', shuffle = True, base = '../images/'):
        
        missing = (pd.read_csv(missingPath, names = ['img', 'src'])['img']).astype(str)
        jsonString = open(jsonPath, 'r').readlines()[0]
        data = json.loads(jsonString)
        
        data = {x: data[x] for x in data if not sum(missing.isin([x]))}
        self.data = data
        keys = list(self.data.keys())
        random.shuffle(keys)
        if shuffle:
            self.data = {x: self.data[x] for x in keys}
        self.imSize = (224, 224)
        self.root = base
        self.keys = list(self.data.keys())
        self.save = False
        

        print("Test set ready with {} samples".format(len(self)))
    def __getitem__(self, index):
        file_ = self.keys[index].split('.')
        file_[-1] = 'npz'
        file_ = '.'.join(file_)
        path = self.root + file_
        try:
            return np.load(path)['arr_0'], self.data[self.keys[index]]
        except FileNotFoundError:
            return self[random.randint(0, len(self)-1)]
    
def get_data_files():
    try:
        
        train_file = pickle.load(open('./Datasets/' + 'train.pkl', 'rb'))
        test_file = pickle.load(open('./Datasets/' + 'test.pkl', 'rb'))
        print('Files found, loaded...')
    except:
        print('Loading and copying files')
        train_file = TrainSet()
        test_file = TestSet()

        pickle.dump(train_file, open('./Datasets/'+'train.pkl', 'wb'))
        pickle.dump(test_file, open('./Datasets/'+'test.pkl', 'wb'))
    return train_file, test_file

def get_data_filesFaster():
    try:
        
        train_file = pickle.load(open('./Datasets/' + 'fast_train.pkl', 'rb'))
        test_file = pickle.load(open('./Datasets/' + 'fast_test.pkl', 'rb'))
        print('Files found, loaded...')
    except:
        print('Loading and copying files')
        train_file = datasets.FastTrainSet()
        test_file = datasets.FastTestSet()

        pickle.dump(train_file, open('./Datasets/'+'fast_train.pkl', 'wb'))
        pickle.dump(test_file, open('./Datasets/'+'fast_test.pkl', 'wb'))
    return train_file, test_file
if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt
    data, _ = get_data_files()

    def experiment(data, name):
        limit = 200
        times = []
        numbers = []
        for i in range(limit):
            t0 = time.time()
            numbers.append(i+1)
            for j in range(i):
                data[j]
            t = time.time()
            times.append(t-t0)
        plt.scatter(numbers, times)
        plt.savefig(name)
        plt.clf()
    
    def retake(img):
        img /= img.max()
        img -= img.min()
        img *= 255
        return img.astype(int).transpose(1, 2, 0)

    for i in range(len(data)):
        image, year = data[i]
        plt.imshow(retake(image))
        plt.savefig('./outTransform/{}.png'.format(i))
    #experiment(dataFast, './fast.png')
    #experiment(data, './slow.png')


    




    '''
    import multiprocessing as mp
    data, dataTest = get_data_files()

    def recorrer_dataset(data):
        data.save = True
        
        def _recorrer(data, process, nProcess):
            lenght = len(data)
            for i in range(process, lenght, nProcess):
                data[i]
                percent = (i+1)/lenght * 100
                print(percent, '% images saved')
        p = 4
        pool = [mp.Process(target=_recorrer, args = (data, x, p)) for x in range(p)]
        [process.start() for process in pool]
        [process.join() for process in pool]

    p1 = mp.Process(target = recorrer_dataset, args = (data, ))
    p2 = mp.Process(target = recorrer_dataset, args = (dataTest, ))
    p1.start(), p2.start()
    p1.join(), p1.join()
    '''
