import torch
import torch.nn as nn
import dataLoader as datasets
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
#from torchvision import datasets
import time
import matplotlib.pyplot as plt
import numpy as np

from loss import DGCLoss
from loss import MAPLoss
from utils import show_batch, cosine_similarity_matrix, ndcg, meanave
import pickle
import seaborn as sns

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

'''
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = torchvision.models.googlenet(pretrained = True)
        self.conv.fc = nn.Linear(1024, 128)
        self.reg = nn.Linear(128, 1, bias = True)

    def forward(self, x):
        y = self.conv(x)
        y_value = self.reg(y) + 1975
        return y, y_value
'''

def train(model, device, train_loader, optim, lossf, epoch, lossf_optional = None, dtype = torch.FloatTensor):
    model.train()
    start_time = time.time()
    train_loss = 0
    for step, (data, targets) in enumerate(train_loader):
        #with torch.autograd.detect_anomaly():
        optim.zero_grad()
        data = data.type(dtype).to(device)
        targets = targets.type(dtype).to(device)
        output, prediction = model(data) 
        loss = lossf(output, targets) + (0 if (not lossf_optional) else 0.1 * lossf_optional(prediction.squeeze(), targets))
        train_loss += loss.item() * data.shape[0]
        loss.backward()
        optim.step()
    train_loss /= len(train_loader.dataset)
    print(
                f'EPOCH: {epoch}',
                f'STEP: {step+1:0>{len(str(len(train_loader)))}}/{len(train_loader)}',
                f'LOSS: {train_loss:.4f}',
    )
    end_time = time.time()
    print(f'TOTAL-TIME: {round(end_time-start_time)}', end='\n')
    return train_loss


def test(model, device, test_loader, lossf, criterion, epoch, dtype = torch.FloatTensor, lossf_optional = None):
    model.eval()
    test_loss = 0
    gc, ap = [], []
    with torch.no_grad():
        for step, (data, targets) in enumerate(test_loader):
            data, targets = data.type(dtype).to(device), targets.type(dtype).to(device)
            
            output, prediction  = model(data)

            test_loss +=lossf(output, targets).item() * data.shape[0] + (0 if (not lossf_optional) else 0.1 * lossf_optional(prediction.squeeze(), targets))
            gc.append(criterion(output, targets, reducefn='none'))
            ap.append(meanave(output, targets, reducefn='none'))
            if step == 0:
                show_batch(data, output, title = f'NDCG: {epoch}')

    test_loss /= len(test_loader.dataset)

    ndgc = np.mean(np.concatenate(gc))
    meap = np.mean(np.concatenate(ap))

    print('\nTest set: Average loss: {:.4f}; smooth-NDCG: {:.4f}; NDCG: {:.4f}; mAP: {:.4f}\n'.format(
        test_loss, 1-test_loss, ndgc, meap))
    return test_loss

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
def main(args):
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    print("USING", device)
    # Prepare Data
    train_file, test_file = get_data_files()
    train_loader = DataLoader(num_workers=8,
        dataset=train_file,
        batch_size=args.batch_size,
        shuffle=True
    )
    test_loader = DataLoader(num_workers=8,
        dataset=test_file,
        batch_size=args.batch_size,
        shuffle=False
    )

    model = CNN().to(device)
    optim = torch.optim.Adam(model.parameters(), args.learning_rate)
    lossf = DGCLoss()
    criterion = ndcg

    plt.ion()
    trainLosses = []
    testLosses = []
    for epoch in range(1, args.epochs+1):
        #loss_optional = nn.MSELoss()
        loss_optional = None
        loss_train = train(model, device, train_loader, optim, lossf, epoch, lossf_optional = loss_optional )
        torch.save(model, './models/{}-model.pkl'.format(epoch))
        loss_test = test(model, device, test_loader, lossf, criterion, epoch, lossf_optional = loss_optional)

        trainLosses.append(loss_train)
        testLosses.append(loss_test)

        plt.plot(trainLosses)
        plt.plot(testLosses)
        plt.legend(['train', 'test'])
        plt.savefig('./loss.png')
        plt.clf()



        
    if args.save_tensors:
        #### GUARDEM LES REPRESENTACIONS ####
        handler = open("./tensors.tsv", "w+")
        handler.write("LABEL\t" + "\t".join([str(x+1) for x in range(128)]))
        for step, (data, targets) in enumerate(test_loader):
                data, targets = data.to(device), targets.to(device)
                output = model(data)
                try:
                    for index in range(args.batch_size):
                        line = "\n" + str(targets[index].detach().numpy().astype(str))+"\t"+"\t".join(output[index].detach().numpy().astype(str))
                        handler.write(line)
                except:
                    continue
        handler.close()
        root = './' #Change
        data = pd.read_csv(root + 'tensors.tsv', delimiter = '\t').groupby('LABEL').mean() 
        dataTensor = torch.Tensor(data.to_numpy())
        matrix = cosine_similarity_matrix(dataTensor, dataTensor).numpy()

        frame = pd.DataFrame(matrix, columns = range(10)) 
        sns.heatmap(frame)
        plt.title("NDCG+MAVEP Cluster Distribution")
        plt.savefig(root + 'distanceMatrix.png')
            

    


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Train MNIST ranking', add_help=False)
    #parser.add_argument('--datasets', type=str)
    parser.add_argument('--epochs', default=64, type=int)
    parser.add_argument('--learning_rate', '-lr', default=1e-3, type=float)
    parser.add_argument('--batch_size', '-bz', default=32, type=int)
    parser.add_argument('--save_tensors', '-s', default = True, type = bool)
    args = parser.parse_args()
    main(args)

