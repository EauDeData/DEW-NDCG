import json
from annoy import AnnoyIndex
import os
import pickle
import torch
import torch.nn as nn
import numpy as np
import cv2
import torchvision

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = torchvision.models.resnet101(pretrained = True)
        self.conv.fc = nn.Linear(2048, 128)
        self.reg = nn.Linear(128, 1, bias = True)



    def forward(self, x):
        y = self.conv(x)
        y_value = self.reg(y) + 1975
        return y, y_value

base = '../images/'

def load_data():

    try:
        return pickle.load(open('./trainDict.pkl', 'rb'))
    except FileNotFoundError:
        
        jsonPath ='../gt_train.json'
        jsonString = open(jsonPath, 'r').readlines()[0]
        data = json.loads(jsonString)
        isEdible = lambda x: os.path.exists(base + x) and os.stat(base + x).st_size != 2051
        dataset = {}
        for num, path in enumerate(data):
            print(num, 'out of', len(data))
            dataset[num] = {'_id': num, 'path': path, 'gt': data[path], 'missing': not isEdible(path)}

        pickle.dump(dataset, open('./trainDict.pkl', 'wb'))
        return dataset

def load_model():
    
    epoch = 30
    model = torch.load(open('./models/{}-model.pkl'.format(epoch), 'rb'))
    model.eval()

    return model
    
def loadImage(path):
    img = cv2.imread(path, 1)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2,0,1)
    img = img/255
    return img.astype(np.float32)

def load_trees():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    embSize = 128
    tree = AnnoyIndex(embSize, 'angular')
    try:
        tree.load('./annTrees/test.ann')
        return tree
    except OSError:
        data = load_data()
        model = load_model()

        lenData = len(data)
        for key in data:

            print(key, 'out of', lenData, end = ' \r')
            img = data[key]
            if img['missing']:
                continue
            path = base + img['path']
            npImg = torch.from_numpy(np.expand_dims(loadImage(path), 0)).to(device)
            emb = model(npImg)[0][0].to('cpu').detach().numpy()
            tree.add_item(key, emb)

        tree.build(10) # 10 trees
        tree.save('./annTrees/test.ann')
        return tree

def retrieval_image(path, tree, model, data, k, weighted = False):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    npImg = torch.from_numpy(np.expand_dims(loadImage(path), 0)).to(device)
    emb = model(npImg)[0][0].to('cpu').detach().numpy()
    
    kNN = tree.get_nns_by_vector(emb, k, include_distances = weighted)
    if weighted:
        kNN, sims = kNN[0], [1/np.exp(x) for x in kNN[1]]
        return [data[x] for x in kNN], sims
    return [data[x] for x in kNN], None


def get_data_files():
    try:
        
        train_file = pickle.load(open('./Datasets/' + 'train.pkl', 'rb'))
        test_file = pickle.load(open('./Datasets/' + 'test.pkl', 'rb'))
        print('Files found, loaded...')
    except:
        print('Loading and copying files')
        train_file = datasets.TrainSet()
        test_file = datasets.TestSet()

        pickle.dump(train_file, open('./Datasets/'+'train.pkl', 'wb'))
        pickle.dump(test_file, open('./Datasets/'+'test.pkl', 'wb'))
    return train_file, test_file

def predict_year(retrieval, gt, weighted = None):
    #print(retrieval)
    if weighted:
        gt_retrieved = [x['gt'] * z  for x, z in zip(retrieval, weighted)]
        prediction = sum(gt_retrieved)/sum(weighted)
        return prediction, abs(prediction - gt)
    
    gt_retrieved = [x['gt'] for x in retrieval]
    prediction = sum(gt_retrieved)/len(gt_retrieved)
    return prediction, abs(prediction - gt)

def get_metrics(retrieval, gt):
    return None, None, None


if __name__ == '__main__':
    weighted = not False
    import matplotlib.pyplot as plt

    tree = load_trees()
    llista_ours = pickle.load(open('./retrievals.pkl', 'rb'))
    data = load_data()
    _, test = get_data_files()
    model = load_model()
    
    arrayError = []
    Xs = []
    MAE = 0
    n = 0
    k = 20
    for query in llista_ours:
        
        if len(llista_ours[query]) == 0:
            continue
        #print(query)
        ret, sims = retrieval_image(base + query, tree, model, data, k, weighted = weighted)
        #print(ret)
        y = test.data[query]
        pred, error = predict_year(ret, y, sims if weighted else None)
        
        MAE += error
        n += 1

    print(MAE/n, end = '\n')

        


