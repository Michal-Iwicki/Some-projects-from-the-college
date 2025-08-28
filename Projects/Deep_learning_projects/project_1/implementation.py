import torch
import torch.nn as nn
import torch.nn.functional as F
class CNNClassifier(nn.Module):
    def __init__(self, num_classes, w1 =64, w2 = 128, w3 = 256,dropout_rate = 0.25, use_bn= True):
        super(CNNClassifier, self).__init__()
        
        self.conv1 = nn.Conv2d(3, w1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(w1, w2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(w2, w3, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(w1) if use_bn else nn.Identity()
        self.bn2 = nn.BatchNorm2d(w2) if use_bn else nn.Identity()
        self.bn3 = nn.BatchNorm2d(w3) if use_bn else nn.Identity()
        self.pool = nn.MaxPool2d(2, 2)
        self.fun = F.relu
        self.fc1 = nn.Linear(w3 * 4 * 4, 128)
        self.drop =  nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool(self.fun(self.bn1(self.conv1(x))))
        x = self.pool(self.fun(self.bn2(self.conv2(x))))
        x = self.pool(self.fun(self.bn3(self.conv3(x))))
        x = self.drop(torch.flatten(x, 1))
        x = self.fun(self.fc1(x))
        x = self.fc2(self.drop(x))
        return x

def train_model(model, train_loader, val_loader, optimizer, epochs=10, printer = True, patience = 3,tracking = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_val_loss = float('inf')
    criterion = nn.CrossEntropyLoss()
    p_counter = 0
    train_losses=[]
    val_losses=[]
    for epoch in range(epochs):
        if p_counter >= patience:
            print("Patience triggered. End of learning")
            break
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        accuracy = 100 * correct / total
        if tracking:
            train_losses.append(running_loss/len(train_loader))
            val_losses.append(val_loss)
        if printer:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, "
                f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            p_counter=0
        else:
            p_counter+=1

    model.load_state_dict(best_model)
    if tracking:
        return train_losses, val_losses

def evaluate(model, test_loader):
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100 * correct / total
    return test_loss, accuracy