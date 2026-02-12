import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.models import efficientnet_b0


def get_pretrained_model():
    model = efficientnet_b0(weights='EfficientNet_B0_Weights.IMAGENET1K_V1')

    original_conv = model.features[0][0]
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=original_conv.bias is not None
    )
    with torch.no_grad():
        new_conv.weight[:] = original_conv.weight.mean(dim=1, keepdim=True)
    model.features[0][0] = new_conv

    return model


def train_model(model, optimizer, train_loader, val_loader, num_epochs=20, patience=3, verbose=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()
    model.to(device)

    counter = 0
    best_val_acc = 0
    best_model = model.state_dict()

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        if not patience is None and counter >= patience:
            print('Patience trigger. End of learning.')
            break

        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        train_acc = 100 * train_correct / train_total

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_acc = 100 * val_correct / val_total

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if verbose:
            print(f"Epoch [{epoch + 1}/{num_epochs}]")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%\n")

        if val_acc > best_val_acc:
            counter = 0
            best_val_acc = val_acc
            best_model = model.state_dict()
        else:
            counter += 1

    model.load_state_dict(best_model)

    return 


def test_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()
    model.to(device)

    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_acc = 100 * test_correct / test_total

    return test_loss / len(test_loader), test_acc
