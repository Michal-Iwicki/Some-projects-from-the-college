import torch.optim as optim

from tester import get_loader,save_to_csv
from CNN_transfomers_implementation import Mel_transformer, train_transformer, evaluate_model
from EfficientNet_implementation import get_pretrained_model, train_model, test_model

def find_best_model(model_type, learning_rates=[0.001, 0.0001, 0.00001], weights_decays=[ 0.01, 0.001, 0.0001], times = 3, denoised=True):
    # retrive respective functions to train and to test model based on model's type
    train_f = train_model if model_type == "EfficientNet" else train_transformer
    test_f = test_model if model_type == "EfficientNet" else evaluate_model
    
    # retrive respective validation loader based on model's type
    val_loader = get_loader('', denoised, True if model_type =="EfficientNet" else False, "validation")
    train_loader = get_loader('', denoised, True if model_type =="EfficientNet" else False, "train")
    
    # initiate default values
    best_val_loss = float("+inf")
    best_learning_rate = 0.001
    best_weights_decay = 0
    columns = ["lr", "train_loss","train_acc", "val_loss","val_acc"]
    values =[]
    # look for the best learning rate
    for learning_rate in learning_rates:
        for i in range(times):
            model = get_pretrained_model() if model_type == "EfficientNet" else Mel_transformer()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # train model
            train_f(model, optimizer, train_loader, val_loader, num_epochs=15)
            
            
            train_loss, train_acc = test_f(model, train_loader)
            val_loss, val_acc = test_f(model, val_loader)
            print(columns)
            print(learning_rate,train_loss, train_acc, val_loss, val_acc)
            values.append([learning_rate,train_loss, train_acc, val_loss, val_acc])
            # update if it is better
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict()
                best_learning_rate = learning_rate
            
    
    save_to_csv("results/lr_"+model_type,columns,values)
    columns = ["wd", "train_loss","train_acc", "val_loss","val_acc"]
    values = []
    for weights_decay in weights_decays:
        for i in range(times):
            # create respective model based on model's type and its optimizer
            model = get_pretrained_model() if model_type == "EfficientNet" else Mel_transformer()
            optimizer = optim.AdamW(model.parameters(), lr=best_learning_rate, weight_decay=weights_decay)
            
            # train model
            train_f(model, optimizer, train_loader, val_loader, num_epochs=15)
            
            train_loss, train_acc = test_f(model, train_loader)
            val_loss, val_acc = test_f(model, val_loader)
            print(columns)
            print(weights_decay,train_loss, train_acc, val_loss, val_acc)
            values.append([weights_decay,train_loss, train_acc, val_loss, val_acc])
            # update if it is better
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict()
                best_weights_decay = weights_decay
    
    save_to_csv("results/wd_"+model_type,columns,values)
    # return best model configuration, best learning rate and weights decay rate
    model.load_state_dict(best_model)
    return model, best_learning_rate, best_weights_decay
            