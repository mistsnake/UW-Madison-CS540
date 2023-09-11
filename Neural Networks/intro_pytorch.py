import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training=True):
    """
    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    dataset = None
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    if training:  # return training set
        dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    else:  # return testing set
        dataset = datasets.FashionMNIST('./data', train=False, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size = 64)


def build_model():
    """
    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    return nn.Sequential(nn.Flatten(),  # takes a 28 x 28 image and flattens
           nn.Linear(784, 128),
           nn.ReLU(),
           nn.Linear(128, 64),
           nn.ReLU(),
           nn.Linear(64, 10))


def train_model(model, train_loader, criterion, T):
    """

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """

    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()  # set model to train mode
    criterion = nn.CrossEntropyLoss()

    # outer loop iterates over epochs
    for epoch in range(T):
        # inner loop iterates over (images, labels) pairs from train DataLoader
        running_loss = 0.0
        total_size = 0
        match = 0  # amt of correct predictions

        for index, pair in enumerate(train_loader, start=0):
            image, label = pair  # get current pair

            opt.zero_grad()  # set gradients to zero

            output = model(image)
            loss = criterion(output, label)
            loss.backward()
            opt.step()

            running_loss = running_loss + loss.item()*train_loader.batch_size

            temp, predicted = torch.max(output, 1)
            total_size = total_size + label.size(0)
            match = match + (predicted == label).sum().item()

        accura_percent = (match/total_size) * 100
        print_loss = running_loss/len(train_loader.dataset)

        print(f"Train Epoch: {epoch}\t Accuracy: {match}/{total_size}({accura_percent:.2f}%)\t Loss: {print_loss: .3f}")


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model.eval()
    match = 0
    total = 0
    with torch.no_grad():
        running_loss = 0.0
        for data, labels in test_loader:
            output = model(data)
            temp, predicted = torch.max(output, 1)
            loss = criterion(output, labels)
            running_loss = running_loss + loss.item()*test_loader.batch_size

            # collect the correct predictions for each class
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    match = match + 1
                total = total + 1
    accuracy_perc = (match/total)*100
    if show_loss:
        print(f"Average loss: {running_loss/len(test_loader.dataset) : .4f}")
    print(f"Accuracy: {accuracy_perc: .2f}%")


def predict_label(model, test_images, index):
    """
    INPUT: 
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    class_names = ['T - shirt / top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

    test_image = test_images[index]  # get the image at the index
    prob = F.softmax(model(test_image), dim=1)  # input logits of test_images into softmax
    most_likely = torch.topk(prob, 3)
    most_probs = most_likely.values[0]
    most_indices = most_likely.indices[0]

    print(f"{class_names[most_indices[0]]}: {most_probs[0]*100:.2f}%\n{class_names[most_indices[1]]}: {most_probs[1]*100:.2f}%\n{class_names[most_indices[2]]}: {most_probs[2]*100:.2f}%")


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''



