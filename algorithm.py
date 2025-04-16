import torch
import numpy as np
from data_utils import *
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import sys
import tqdm
from sklearn.linear_model import LinearRegression

def regression(dataset):
    """
    fit a linear regression model to the dataset
    """
    X_train, Y_train, Z_train, XZ_train = dataset.X_train, dataset.Y_train, dataset.Z_train, dataset.XZ_train
    X_test, Y_test, Z_test, XZ_test = dataset.X_test, dataset.Y_test, dataset.Z_test, dataset.XZ_test
    reg = LinearRegression().fit(X_train, Y_train)
    Yhat = reg.predict(X_test)
    # Training RMSE
    rmse = np.sqrt(np.mean((Y_train - reg.predict(X_train))**2))
    print(f"Training RMSE of the model: {rmse:4.2f}")
    # Testing RMSE
    rmse = np.sqrt(np.mean((Y_test - Yhat)**2))
    print(f"Testing RMSE of the model: {rmse:4.2f}")
    return reg


def trainer_h(model, dataset, optimizer, device, n_epochs, batch_size):
    """
    train the ground truth classifier h: utilizing all features, using a MLP model
    """
    train_tensors, test_tensors = dataset.get_dataset_in_tensor()

    _, Y_train, Z_train, XZ_train = train_tensors
    _, Y_test, Z_test, XZ_test = test_tensors
    train_dataset = Dataset_prep(XZ_train, Y_train, Z_train)
    test_dataset = Dataset_prep(XZ_test, Y_test, Z_test)
    train_dataset, val_dataset = random_split(train_dataset,[int(0.9*len(train_dataset)),len(train_dataset)-int(0.9*len(train_dataset))])
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    
    tau = 0.5
    loss_func = torch.nn.BCELoss(reduction = 'mean')
    for epoch in tqdm.trange(n_epochs, desc="Training", unit="epochs"):
        for _, (x_batch, y_batch, z_batch) in enumerate(train_loader):
            x_batch, y_batch, z_batch = x_batch.to(device), y_batch.to(device), z_batch.to(device)
            Yhat = model(x_batch)
            cost = loss_func(Yhat.reshape(-1), y_batch)
            optimizer.zero_grad()
            if (torch.isnan(cost)).any():
                continue
            cost.backward()
            optimizer.step()
    
    
    Yhat_val = model(val_dataset.dataset.X[val_dataset.indices]).reshape(-1).detach().numpy()
    Y = np.array(val_dataset.dataset.Y[val_dataset.indices])
    Yhat_labels = np.array(1*(Yhat_val >= tau))
    true_preds = (Y == Yhat_labels).sum()
    acc = true_preds/len(Yhat_labels)
    print(f"Validation Accuracy of the model: {100.0*acc:4.2f}%")

    Yhat_test = model(test_dataset.X).reshape(-1).detach().numpy()
    Yhat_labels = np.array(1*(Yhat_test >= tau))
    true_preds = (np.array(Y_test) == Yhat_labels).sum()
    acc_test = true_preds/len(Yhat_labels)
    print(f"Testing Accuracy of the model: {100.0*acc_test:4.2f}%")
    return acc_test*100.0




def Grad_effort(model, x,  We, c_idx, nc_idx):
    """
    1. calculate the gradient of the model with respect to the input: g
    2. effort budget is fixed to delta
    3. effort conversion matrix We: 1 effort -> We unit of feature
    3. calculate the effort allocation: g * We / ||g * We|| * delta to ensure the norm of total effort is delta, and the effort is proportional to the gradient   
    """
    x.requires_grad = True
    Yhat = model(x).sum()
    g, = torch.autograd.grad(Yhat, x, create_graph=True)
    all_idx = c_idx + nc_idx
    k = len(all_idx)
    g = g[:,all_idx]
    efforts = (g*(We))/(torch.norm(g*(We),p=2,dim=1)+1e-8).reshape(-1,1).repeat(1,k)
    efforts = torch.nan_to_num(efforts)
    # the new feature vector after manipulation and improvement
    x_star = x.clone()
    x_star[:,all_idx] = x[:,all_idx] + efforts * We
    # the new feature vector after improvement
    x_improve = x_star.clone()
    # reverse back for the manipulated features
    if len(nc_idx) > 0:
        x_improve[:,nc_idx] = x[:,nc_idx]
    return efforts, x_star, x_improve