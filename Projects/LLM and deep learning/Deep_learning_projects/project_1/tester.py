import torch.optim as optim
from torchvision import transforms
import numpy as np
import torch.nn as nn
from loader import load_png_images
from implementation import CNNClassifier, train_model,evaluate
from torchvision import models
import pandas as pd
import csv
import os
import datetime
import matplotlib.pyplot as plt

MISZA = False
SAMPLE = False

if MISZA:
    sample_train_path = os.path.join(os.getcwd(), "data", "sample", "train")
    sample_valid_path = os.path.join(os.getcwd(), "data", "sample", "valid")
    sample_test_path = os.path.join(os.getcwd(), "data", "sample", "test")
    
    full_train_path = os.path.join(os.getcwd(), "data", "full", "train")
    full_valid_path = os.path.join(os.getcwd(), "data", "full", "valid")
    full_test_path = os.path.join(os.getcwd(), "data", "full", "test")
else:
    sample_train_path = "data/train_sample"
    sample_valid_path = "data/valid_sample"
    sample_test_path = "data/test_sample"
    
    full_train_path = "data/train"
    full_valid_path = "data/valid"
    full_test_path = "data/test"

if SAMPLE:
    train_path = sample_train_path
    valid_path = sample_valid_path
    test_path = sample_test_path
else:
    train_path = full_train_path
    valid_path = full_valid_path
    test_path = full_test_path

def save_to_csv(data, filename, column_names):
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H.%M")
    filename = f"{filename}_{timestamp}.csv"
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(column_names)  
        writer.writerows(data)

def aggregate_csv_files(folder_path = "./results"):
    all_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    dataframes = []
    
    for file in all_files:
        df = pd.read_csv(os.path.join(folder_path, file))
        group_col = df.columns[0]  
        aggregated_df = df.groupby(group_col).agg(["mean", "std"]).round(3)
        aggregated_df.columns = ["_".join(col).lower() for col in aggregated_df.columns]
        dataframes.append(aggregated_df)
    
    return dataframes

def test_batchsizes(times=3, batchsizes=[16, 32, 64]):
    test_loader = load_png_images(test_path, batch_size=1024, shuffle=False)[0]
    val_loader = load_png_images(valid_path, batch_size=1024, shuffle=False)[0]
    result = []
    column_names = ["batchsize", "train_loss", "train_accuracy", "test_loss", "test_accuracy"]
    print(*column_names)
    for batchsize in batchsizes:
        for i in range(times):
            train_loader, num_classes = load_png_images(train_path, batch_size=batchsize)
            model = CNNClassifier(num_classes)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            train_model(model, train_loader, val_loader, optimizer, epochs=10, printer=False)
            r_loss, r_acc = evaluate(model, train_loader)
            t_loss, t_acc = evaluate(model, test_loader)
            values = [batchsize,r_loss, r_acc,t_loss, t_acc]
            result.append(values)
            print(*values)
    save_to_csv(result, "results/batchsize_test.csv", column_names)

def test_lr(times=3, learning_rates=[0.0001, 0.001, 0.01]):
    test_loader = load_png_images(test_path, batch_size=1024, shuffle=False)[0]
    val_loader = load_png_images(valid_path, batch_size=1024, shuffle=False)[0]
    result = []
    column_names = ["learning_rate", "train_loss", "train_accuracy", "test_loss", "test_accuracy"]
    print(*column_names)
    for lr in learning_rates:
        for i in range(times):
            train_loader, num_classes = load_png_images(train_path, batch_size=32)
            model = CNNClassifier(num_classes)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            train_model(model, train_loader, val_loader, optimizer, epochs=10, printer=False)
            r_loss, r_acc = evaluate(model, train_loader)
            t_loss, t_acc = evaluate(model, test_loader)
            values = [lr, r_loss, r_acc, t_loss, t_acc]
            result.append(values)
            print(*values)
    save_to_csv(result, "results/lr_test.csv", column_names)

def test_dropout(times=3, dropout_rates=[0.25, 0.5, 0.75]):
    test_loader = load_png_images(test_path, batch_size=1024, shuffle=False)[0]
    val_loader = load_png_images(valid_path, batch_size=1024, shuffle=False)[0]
    result = []
    column_names = ["dropout_rate", "train_loss", "train_accuracy", "test_loss", "test_accuracy"]
    print(*column_names)
    for dr in dropout_rates:
        for i in range(times):
            train_loader, num_classes = load_png_images(train_path, batch_size=32)
            model = CNNClassifier(num_classes, dropout_rate=dr)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            train_model(model, train_loader, val_loader, optimizer, epochs=10, printer=False)
            r_loss, r_acc = evaluate(model, train_loader)
            t_loss, t_acc = evaluate(model, test_loader)
            values = [dr, r_loss, r_acc, t_loss, t_acc]
            result.append(values)
            print(*values)
    save_to_csv(result, "results/dropout_rate_test.csv", column_names)

def test_w_decay(times=3, w_decays=[0.001,0.01,0.1]):
    test_loader = load_png_images(test_path, batch_size=1024, shuffle=False)[0]
    val_loader = load_png_images(valid_path, batch_size=1024, shuffle=False)[0]
    result = []
    column_names = ["weight_decay", "train_loss", "train_accuracy", "test_loss", "test_accuracy"]
    print(*column_names)
    for w_decay in w_decays:
        for i in range(times):
            train_loader, num_classes = load_png_images(train_path, batch_size=32)
            model = CNNClassifier(num_classes)
            optimizer = optim.AdamW(model.parameters(), weight_decay=w_decay)
            train_model(model, train_loader, val_loader, optimizer, epochs=10, printer=False)
            r_loss, r_acc = evaluate(model, train_loader)
            t_loss, t_acc = evaluate(model, test_loader)
            values = [w_decay, r_loss, r_acc, t_loss, t_acc]
            result.append(values)
            print(*values)
    save_to_csv(result, "results/weight_decay_test.csv", column_names)    


def find_best_cnn(w_decays=[0.001,0.01,0.1], lr_rates=[0.0001,0.01,0.1]):
    val_loader = load_png_images(valid_path, batch_size=1024, shuffle=False)[0]
    best_loss= float('inf')
    best_w = 0
    best_lr=0.001 #default
    for w_decay in w_decays:
        train_loader, num_classes = load_png_images(train_path, batch_size=32)
        model = CNNClassifier(num_classes)
        optimizer = optim.AdamW(model.parameters(), weight_decay=w_decay)
        val_loss = train_model(model, train_loader, val_loader, optimizer, epochs=10, printer=False,tracking=True)[1][-1]
        if best_loss>val_loss:
            best_loss = val_loss
            best_w = w_decay
            best_model = model.state_dict()
    for lr in lr_rates:
        train_loader, num_classes = load_png_images(train_path, batch_size=32)
        model = CNNClassifier(num_classes)
        optimizer = optim.AdamW(model.parameters(), lr= lr,weight_decay=best_w)
        val_loss = train_model(model, train_loader, val_loader, optimizer, epochs=10, printer=False,tracking=True)[1][-1]
        if best_loss>val_loss:
            best_lr = lr
            best_loss = val_loss
            best_model = model.state_dict()
    return model.load_state_dict(best_model), best_lr,best_w

def find_best_type(type, w_decays=[0.001,0.01,0.1], lr_rates=[0.0001,0.01,0.1]):#0.001 is default so already tested
    if type == "DenseNet":
        model_pre = models.densenet121
        initial = "DenseNet121_Weights.IMAGENET1K_V1"
        num_features=model_pre(weights=initial).classifier.in_features
    else:
        model_pre= models.wide_resnet50_2
        initial = "Wide_ResNet50_2_Weights.DEFAULT"
        num_features=model_pre(weights=initial).fc.in_features
    val_loader = load_png_images(valid_path, batch_size=1024, shuffle=False)[0]
    best_loss= float('inf')
    best_w = 0
    best_lr = 0.001 #default
    for wd in w_decays:
        train_loader, num_classes = load_png_images(train_path, batch_size=32)
        model = model_pre(weights=initial)
        model.fc = nn.Linear(num_features, num_classes)   
        optimizer = optim.AdamW(model.parameters(), weight_decay=wd)
        val_loss = train_model(model, train_loader, val_loader, optimizer, epochs=10, printer=False,tracking=True)[1][-1]
        if best_loss>val_loss:
            best_loss = val_loss
            best_w = wd
            best_model = model.state_dict()
    for lr in lr_rates:
        train_loader, num_classes = load_png_images(train_path, batch_size=32)
        model = model_pre(weights=initial)
        model.fc = nn.Linear(num_features, num_classes)   
        optimizer = optim.AdamW(model.parameters(), lr= lr, weight_decay=best_w)
        val_loss = train_model(model, train_loader, val_loader, optimizer, epochs=10, printer=False,tracking=True)[1][-1]
        if best_loss>val_loss:
            best_loss = val_loss
            best_lr=lr
            best_model = model.state_dict()
    return model.load_state_dict(best_model), best_lr, best_w


def plot_training(type, epochs, w_decay, lr):
    val_loader = load_png_images(valid_path, batch_size=1024, shuffle=False)[0]
    train_loader,num_classes = load_png_images(train_path, batch_size=32)
    if type == "DenseNet":
        model = models.densenet121(weights="DenseNet121_Weights.IMAGENET1K_V1")
        model.fc = nn.Linear(model.classifier.in_features, num_classes)   
    elif type == "WideResNet":
        model=models.wide_resnet50_2(weights="Wide_ResNet50_2_Weights.DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, num_classes)   
    else:
        model = CNNClassifier(num_classes)

    optimizer = optim.AdamW(model.parameters(), lr= lr,weight_decay=w_decay)
    train_loss, val_loss = train_model(model, train_loader, val_loader, optimizer, epochs=epochs,patience=epochs, printer=False,tracking=True)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1,epochs+1), train_loss, label='Train Loss', marker='o', linestyle='-')
    plt.plot(range(1,epochs+1), val_loss, label='Validation Loss', marker='s', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
    return model


def test_augmentation(times=3, transformations={
    "no-transform": [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4788952171802521, 0.4722793698310852, 0.43047481775283813], std=[0.24205632507801056, 0.2382805347442627, 0.25874853134155273])
    ],
    "flips": [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4788952171802521, 0.4722793698310852, 0.43047481775283813], std=[0.24205632507801056, 0.2382805347442627, 0.25874853134155273])
    ], 
    "auto-augment": [
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4788952171802521, 0.4722793698310852, 0.43047481775283813], std=[0.24205632507801056, 0.2382805347442627, 0.25874853134155273])
    ],
    "rand-augment": [
        transforms.RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4788952171802521, 0.4722793698310852, 0.43047481775283813], std=[0.24205632507801056, 0.2382805347442627, 0.25874853134155273])
    ]
}):
    test_loader = load_png_images(test_path, batch_size=1024, shuffle=False)[0]
    val_loader = load_png_images(valid_path, batch_size=1024, shuffle=False)[0]
    result = []
    column_names = ["transformation", "train_loss", "train_accuracy", "test_loss", "test_accuracy"]
    print(*column_names)
    for name, transformation in transformations.items():
        for i in range(times):
            train_loader, num_classes = load_png_images(train_path, transform=transformation, batch_size=32)
            model = CNNClassifier(num_classes)
            optimizer = optim.Adam(model.parameters())
            train_model(model, train_loader, val_loader, optimizer, epochs=10, printer=False)
            r_loss, r_acc = evaluate(model, train_loader)
            t_loss, t_acc = evaluate(model, test_loader)
            values = [name, r_loss, r_acc, t_loss, t_acc]
            result.append(values)
            print(*values)
    save_to_csv(result,"results/augmentation_test.csv", column_names)