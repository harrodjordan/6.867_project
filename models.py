import sklearn 
import matplotlib.pyplot as plt 
import numpy as np
import os 
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import precision_score, roc_curve, auc 
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.utils.data as data_utils
import torch.optim as optim
from random import shuffle



class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.log_softmax(x)

class CNN():

    def __init__(self):
        self.model = ConvNet()

    def train(self, model, device, train_loader, optimizer, epoch):
        self.model.train()
        sum_num_correct = 0
        sum_loss = 0
        num_batches_since_log = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct = pred.eq(target.view_as(pred)).sum().item()
            sum_num_correct += correct
            sum_loss += loss.item()
            num_batches_since_log += 1
            
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{:05d}/{} ({:02.0f}%)]\tLoss: {:.6f}\tAccuracy: {:02.0f}%'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), sum_loss / num_batches_since_log, 
                    100. * sum_num_correct / (num_batches_since_log * train_loader.batch_size))
                )
                sum_num_correct = 0
                sum_loss = 0
                num_batches_since_log = 0
    
    def test(self, model, device, test_loader, dataset_name="Test set"):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(dataset_name, test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

    def training_procedure(self, X, y, name_list):

        args = dict()
        args["seed"] = 73912
        args["no_cuda"] = False
        args["log_interval"] = 100
        args["batch_size"] = 32
        args["test-batch-size"] = 1000
        

        params = dict()
        params["epochs"] = 10
        params["lr"] = 0.1
        

        torch.manual_seed(args["seed"])
        use_cuda = not args["no_cuda"] and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        

        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        X_train, X_test, y_train, y_test = train_test_split(X, y, name_list, test_size = 0.2)

        train_loader = data_utils.TensorDataset(X_train, y_train)
        test_loader = data_utils.TensorDataset(X_test, y_test)
        
        
        model = ConvNet().to(device)
        optimizer = optim.SGD(model.parameters(), lr=params["lr"])

        # Train the model
        for epoch in range(1, params["epochs"] + 1):
            self.train(model=model, device=device, train_loader=train_loader, optimizer=optimizer, epoch=epoch)
            self.test(model, device, test_loader)
 


class SVM():

    def __init__(self, C=0.001):
        # initialize classifier
        self.clf = svm.SVC(C=C, class_weight='balanced', probability = True)


    def train(self, X, y, name_list):
        # fit the classifier
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, name_list, test_size=0.2)
        self.clf = self.clf.fit(X,y)

    def test(self, X, true_labels):
        prediction = self.clf.predict(X)
        # get mean accuracy (not a good metric!)
        score = self.clf.score(X, true_labels)
        precision = precision_score(true_labels, prediction)
        return score, precision 

    def roc_auc(self, X, true):
        pred = self.clf.predict(X)
        fpr, tpr, thresholds = roc_curve(true, pred)
        auc_var = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,label='ROC (AUC = %0.2f)' % (auc_var))
        plt.show()

    def plot_margin(self, X, y):
        plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

        # plot the decision function
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # create grid to evaluate model
        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = self.clf.decision_function(xy).reshape(XX.shape)

        # plot decision boundary and margins
        ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
                   linestyles=['--', '-', '--'])
        # plot support vectors
        ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
                   linewidth=1, facecolors='none', edgecolors='k')
        plt.show()



# TODO add to parameter list
class RandomForest():

    def __init__(self, n_estimators=10):
        # initialize classifier
        self.clf = RandomForestClassifier(n_estimators=n_estimators)

    def train(self, X, y, name_list):
        # fit the classifier
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, name_list, test_size=0.2)
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.clf = self.clf.fit(X_train, y_train)

    # do we want to just spit out labels or testing accuracies?
    # kind of an arbitrary choice
    # we also care about things like TPR etc
    def test(self, X, true_labels):
        # predict labels
        prediction = self.clf.predict(X)
        # get mean accuracy (not a good metric!)
        score = self.clf.score(X, true_labels)
        precision = precision_score(true_labels, prediction)

        return score, precision 

def train_test_split(X, y, name_list, test_size=0.2, randomize=True) :
    # right now this just splits at the 80% line (no randomness)

    # need to make sure data from a single patient are all in the same category
    # each label is VXX.YYY, but with one to two Xs zero to three Ys
    # so we take XXYYY to be a unique label for a patient (is this true??) (maybe lol)
    # we then sort the data by those labels so that all the data from a given
    # patient is reunited with its source. 

    sort_by = list()

    for x in name_list :
    	labels_split = x.split('.')
    	
    	part1 = labels_split[1][1:]
    	if len(labels_split) == 2 :
    		part2 = ''
    	else :
    		part2 = labels_split[2]

    	sort_by.append(int(part1 + part2))

    # get a list with one of each name
    names_unique = list(set(sort_by))

    def shuffle_helper(elem) :
        return names_unique.index(elem[0])


    # if desired, randomize the order
    if randomize :
        shuffle(names_unique)

    indexing = range(0,len(sort_by))

    sort_by, indexing = (list(t) for t in zip(*sorted(zip(sort_by, indexing),key=shuffle_helper)))

    order = np.asarray(indexing)

    X_sorted = X[order]
    y_sorted = y[order]

    # we split at the closest multiple of 23 to the requested split point

    split_at = int(int(int(X.shape[0] * (1 - test_size)) / 23.0) * 23)

    X_train = X[:split_at,:]
    X_valid = X[split_at:,:]
    y_train = y[:split_at]
    y_valid = y[split_at:]
    return X_train, X_valid, y_train, y_valid

    
