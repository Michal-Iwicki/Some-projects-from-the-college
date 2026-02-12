import csv
import datetime
import os
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from EfficientNet_implementation import get_pretrained_model, train_model, test_model
from CNN_transfomers_implementation import Mel_transformer, train_transformer, evaluate_model
from new_loader import TorchTensorFolderDataset
import matplotlib.pyplot as plt

def get_model(model_type):
    return get_pretrained_model() if model_type == "EfficientNet" else Mel_transformer()

def get_models_functions(model_type):
    train_f = train_model if model_type == "EfficientNet" else train_transformer
    test_f = test_model if model_type == "EfficientNet" else evaluate_model

    return train_f, test_f


def get_loader(data_size: str = "sample", denoised: bool = False, use_mel: bool = True, target_data: str = "train", batch_size: int = 16):
    path = os.path.join(os.getcwd(), "data", "preprocessed", data_size, "denoised" if denoised else "standard", "mel" if use_mel else "raw", target_data)

    dataset = TorchTensorFolderDataset(path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader


def save_to_csv(filename, column_names, data):
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H.%M")
    filename = f"{filename}_{timestamp}.csv"

    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(column_names)
        writer.writerows(data)

    return


# def test_learning_rates(model_type, times=3, learning_rates=[0.1, 0.01, 0.001], data_size="sample", denoised=False, use_mel=True):
#     train_model, test_model = get_models_functions(model_type)

#     val_loader = get_loader(data_size, denoised, use_mel, "validation")
#     test_loader = get_loader(data_size, denoised, use_mel, "test")

#     result = []
#     column_names = ["learning_rate", "train_loss", "train_acc", "val_loss", "val_acc", "test_loss", "test_acc"]
#     print(*column_names)

#     for learning_rate in learning_rates:
#         for _ in range(times):
#             train_loader = get_loader(data_size, denoised, use_mel, "train")

#             model = get_model(model_type)
#             optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#             train_model(model, optimizer, train_loader, val_loader, num_epochs=10)

#             train_loss, train_acc = test_model(model, train_loader)
#             val_loss, val_acc = test_model(model, val_loader)
#             test_loss, test_acc = test_model(model, test_loader)

#             result.append((learning_rate, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc))

#             print(*result[-1])

#     save_to_csv("results/learning_rates_test.csv", column_names, result)

#     return


# def test_batch_sizes(model_type, times=3, batch_sizes=[16, 32, 64], data_size="sample", denoised=False, use_mel=True):
#     train_model, test_model = get_models_functions(model_type)

#     val_loader = get_loader(data_size, denoised, use_mel, "validation")
#     test_loader = get_loader(data_size, denoised, use_mel, "test")

#     result = []
#     column_names = ["batch_size", "train_loss", "train_acc", "val_loss", "val_acc", "test_loss", "test_acc"]
#     print(*column_names)

#     for batch_size in batch_sizes:
#         for _ in range(times):
#             train_loader = get_loader(data_size, denoised, use_mel, "train", batch_size)

#             model = get_model(model_type)
#             optimizer = optim.Adam(model.parameters(), lr=0.001)

#             train_model(model, optimizer, train_loader, val_loader, num_epochs=10)

#             train_loss, train_acc = test_model(model, train_loader)
#             val_loss, val_acc = test_model(model, val_loader)
#             test_loss, test_acc = test_model(model, test_loader)

#             result.append((batch_size, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc))

#             print(*result[-1])

#     save_to_csv("results/batch_sizes_test.csv", column_names, result)

#     return


# def test_weights_decays(model_type, times=3, weights_decays=[0.1, 0.5, 0.9], data_size="sample", denoised=False, use_mel=True):
#     train_model, test_model = get_models_functions(model_type)

#     val_loader = get_loader(data_size, denoised, use_mel, "validation")
#     test_loader = get_loader(data_size, denoised, use_mel, "test")

#     result = []
#     column_names = ["weights_decays", "train_loss", "train_acc", "val_loss", "val_acc", "test_loss", "test_acc"]
#     print(*column_names)

#     for weights_decay in weights_decays:
#         for _ in range(times):
#             train_loader = get_loader(data_size, denoised, use_mel, "train")

#             model = get_model(model_type)
#             optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=weights_decay)

#             train_model(model, optimizer, train_loader, val_loader, num_epochs=10)

#             train_loss, train_acc = test_model(model, train_loader)
#             val_loss, val_acc = test_model(model, val_loader)
#             test_loss, test_acc = test_model(model, test_loader)

#             result.append((weights_decay, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc))

#             print(*result[-1])

#     save_to_csv("results/weights_decays_test.csv", column_names, result)

#     return

def get_confusion_matrix(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()
    
    all_predicted = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
    
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predicted.append(predicted)
            all_labels.append(labels)
    
    # Now concatenate all batches into a single tensor
    all_predicted = torch.cat(all_predicted)
    all_labels = torch.cat(all_labels)

    cm = confusion_matrix(all_labels.cpu(), all_predicted.cpu())
    labels = list(test_loader.dataset.class_to_idx.keys())

    fig, ax = plt.subplots(figsize=(12, 12))  # większy wykres
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Oranges', ax=ax, colorbar=False, xticks_rotation=90)

    # Usuń etykiety z zerami
    for i in range(disp.text_.shape[0]):
        for j in range(disp.text_.shape[1]):
            if disp.text_[i, j] is not None and disp.text_[i, j].get_text() == '0':
                disp.text_[i, j].set_text('')

    plt.tight_layout()

    return
