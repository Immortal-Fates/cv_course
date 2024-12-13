import os
import hashlib
import requests
import gzip
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# LeNet-5 Model
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_data(batch_size, root='./dataset/MNIST'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(root=root, 
                                               train=True, 
                                               transform=transform, 
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root=root, 
                                              train=False, 
                                              transform=transform, 
                                              download=True)

    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True)

    test_loader = DataLoader(dataset=test_dataset, 
                             batch_size=batch_size, 
                             shuffle=False)
    return train_loader, test_loader

def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    model.train()
    loss_list = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i+1) % 100 == 0:
                avg_loss = running_loss / 100
                loss_list.append(avg_loss)
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {avg_loss:.4f}')
                running_loss = 0.0
    return loss_list

def test_model(model, test_loader, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy of the model on the 10000 test images: {100 * correct / total} %')

def show_predictions(model, test_loader, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        images, labels = next(iter(test_loader))
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        fig = plt.figure(figsize=(15, 15))
        for i in range(10):
            ax = fig.add_subplot(2, 5, i+1)
            ax.imshow(images[i].cpu().numpy().squeeze(), cmap='gray')
            ax.set_title(f'True: {labels[i].item()}\nPred: {predicted[i].item()}')
            ax.axis('off')
        plt.show()

def main():
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_loader, test_loader = load_data(batch_size, './dataset/MNIST')
    model = LeNet5()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # # Train the model
    # loss_list = train_model(model, train_loader, criterion, optimizer, num_epochs, device)
    
    # # Save the model checkpoint
    # torch.save(model.state_dict(), './model/lenet5_model1.pth')
    # print('Model saved to lenet5_model.pth')

    # Load the trained model
    model.load_state_dict(torch.load('./model/lenet5_model1.pth'))
    print('Model loaded from lenet5_model.pth')
    
    # Test the model
    test_model(model, test_loader, device)

    # Show predictions for the first 10 images in the test set
    show_predictions(model, test_loader, device)

    # # Plot the loss curve
    # plt.figure()
    # plt.plot(loss_list)
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.title('Training Loss Curve')
    # plt.show()

if __name__ == "__main__":
    main()