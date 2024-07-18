import torch
import numpy as np
from torch.autograd import grad
from types import SimpleNamespace
from utils import *
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import sys
sys.path.append("..")
from br import *
import tqdm



def trainer_h(model, dataset, optimizer, device, n_epochs, batch_size):
    """
    train the ground truth classifier h: utilizing all features, using a MLP model
    """
    train_tensors, test_tensors = dataset.get_dataset_in_tensor()
    X_train, Y_train, Z_train, XZ_train = train_tensors
    X_test, Y_test, Z_test, XZ_test = test_tensors
    train_dataset = FairnessDataset(XZ_train, Y_train, Z_train)
    test_dataset = FairnessDataset(XZ_test, Y_test, Z_test)
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







def trainer_baselines(model, h, dataset, optimizer, device, n_epochs, batch_size, z_blind, metric, lambda_, delta_effort=1, ctv=None):
    """
    Training functions for baseline algos:
    1. Regular ERM
    2. algorithms for fairness under strategic classification: EI, BE
    3. a baseline algorithm only considering "safety": SAFE
    """

    train_tensors, test_tensors = dataset.get_dataset_in_tensor()
    X_train, Y_train, Z_train, XZ_train = train_tensors
    X_test, Y_test, Z_test, XZ_test = test_tensors

    sensitive_attrs = dataset.sensitive_attrs
    if z_blind is True:
        train_dataset = FairnessDataset(X_train, Y_train, Z_train)
        test_dataset = FairnessDataset(X_test, Y_test, Z_test)
    else:
        train_dataset = FairnessDataset(XZ_train, Y_train, Z_train)
        test_dataset = FairnessDataset(XZ_test, Y_test, Z_test)
    
    train_dataset, val_dataset = random_split(train_dataset,[int(0.8*len(train_dataset)),len(train_dataset)-int(0.8*len(train_dataset))])
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    
    tau = 0.5


    results = SimpleNamespace()
    loss_func = torch.nn.BCELoss(reduction = 'mean')
    relu = torch.nn.ReLU()

    # prediction loss
    d_losses = []
    # other metric loss
    f_losses = []

    # decision-maker welfare
    accuracies = []
    # social welfare
    imp_all = []
    imp_a = []
    imp_b = []
    sfty = []
    # applicant welfare
    aw_a = []
    aw_b = []
    aw_all = []


    # fairness
    dp_disparities = []
    eo_disparities = []
    eodd_disparities = []
    ei_disparities = []
    be_disparities = []



    for epoch in tqdm.trange(n_epochs, desc="Training", unit="epochs"):
        
        local_d_loss = []
        local_f_loss = []

        for _, (x_batch, y_batch, z_batch) in enumerate(train_loader):
            x_batch, y_batch, z_batch = x_batch.to(device), y_batch.to(device), z_batch.to(device)
            Yhat = model(x_batch)
            
            cost = 0
            f_loss = 0

            # decision-maker loss
            d_loss = loss_func(Yhat.reshape(-1), y_batch)

            # Equal Improvability
            if metric == 'EI' and torch.sum(Yhat<tau)>0:
                x_batch_e = x_batch[(Yhat<tau).reshape(-1),:]
                z_batch_e = z_batch[(Yhat<tau).reshape(-1)]
                Yhat_max, Yhat_total = Grad_effort(model, h, x_batch_e, delta_effort, ctv=ctv)

                loss_mean = loss_func(Yhat_max.reshape(-1), torch.ones(len(Yhat_max)))
                loss_z = torch.zeros(len(sensitive_attrs), device = device)
                for z in sensitive_attrs:
                    z = int(z)
                    group_idx = z_batch_e == z
                    if group_idx.sum() == 0:
                        continue
                    loss_z[z] = loss_func(Yhat_max.reshape(-1)[group_idx], torch.ones(group_idx.sum()))
                    f_loss += torch.abs(loss_z[z] - loss_mean)
                cost += lambda_*f_loss
                cost += (1-lambda_)*d_loss
            
            # Bounded Effort
            elif metric == 'BE':
                x_batch_e = x_batch[(Yhat<tau).reshape(-1),:]
                z_batch_e = z_batch[(Yhat<tau).reshape(-1)]
                Yhat_max, Yhat_total = Grad_effort(model, h, x_batch_e, delta_effort, ctv = ctv)

                loss_mean = (len(x_batch_e)/len(x_batch))*loss_func(Yhat_max.reshape(-1), torch.ones(len(Yhat_max)))
                loss_z = torch.zeros(len(sensitive_attrs), device = device)
                for z in sensitive_attrs:
                    z = int(z)
                    group_idx = z_batch_e == z
                    if group_idx.sum() == 0:
                        continue
                    loss_z[z] = (z_batch_e[z_batch_e==z].shape[0]/z_batch[z_batch==z].shape[0])*loss_func(Yhat_max.reshape(-1)[group_idx], torch.ones(group_idx.sum()))
                    f_loss += torch.abs(loss_z[z] - loss_mean)
                cost += lambda_*f_loss
                cost += (1-lambda_)*d_loss

            # SAFE
            elif metric == 'SAFE':
                Yhat_max, Yhat_total = Grad_effort(model, h, x_batch, delta_effort, ctv = ctv)

                f_loss += torch.mean(relu(h(x_batch) - Yhat_total))
                cost += (lambda_*f_loss)
                cost += (1-lambda_)*d_loss

            # ERM
            else:
                cost += d_loss


            optimizer.zero_grad()
            if (torch.isnan(cost)).any():
                continue
            cost.backward()
            optimizer.step()

            local_d_loss.append(d_loss.item())

            # EI and BE have f_loss
            if metric == 'EI' or metric == 'BE' or metric == 'SAFE':
                if hasattr(f_loss,'item'):
                    local_f_loss.append(f_loss.item())      

        # the average loss of training
        d_losses.append(np.mean(local_d_loss))
        f_losses.append(np.mean(local_f_loss))

        # prediction
        Yhat_train = model(train_dataset.dataset.X[train_dataset.indices]).reshape(-1).detach().numpy()

        # get the final best responses
        Yhat_max_train, Yhat_total_train = Grad_effort(model, h, train_dataset.dataset.X[train_dataset.indices], delta_effort, ctv = ctv)
        
        x_start = train_dataset.dataset.X[train_dataset.indices]
        y_true = h(x_start).detach().numpy()
        z_true = train_dataset.dataset.Z[train_dataset.indices].detach().numpy()

        # get improvement measures
        imp_all.append(np.mean(Yhat_total_train.detach().numpy()-y_true))

        if epoch == n_epochs - 1:
            wf_a, wf_b, wf_all = aw_calc(z_true,Yhat_train,y_true)
            aw_a.append(wf_a)
            aw_b.append(wf_b)
            aw_all.append(wf_all)

        # improvement disparity
        aimp, bimp = 0,0
        y0 = h(x_start[z_true == 0]).detach().numpy()
        y1 = h(x_start[z_true == 1]).detach().numpy()
        ystar0 = Yhat_total_train[z_true == 0].detach().numpy()
        ystar1 = Yhat_total_train[z_true == 1].detach().numpy()

        aimp += np.mean((ystar0-y0))
        bimp += np.mean((ystar1-y1))
        m0  = ystar0 - y0
        m1 = ystar1 - y1
        m0 = np.where(m0 > 0, 0, m0)
        m1 = np.where(m1 > 0, 0, m1)
        saf = len(m0)/(len(m0)+len(m1))*np.mean(m0) + len(m1)/(len(m0)+len(m1))*np.mean(m1)

        sfty.append(saf)

        imp_a.append(aimp)
        imp_b.append(bimp)

        Yhat_max_train = Yhat_max_train.reshape(-1).detach().numpy()

        # still can get fairness measures
        accuracy, dp_disparity, eo_disparity, eodd_disparity, ei_disparity, be_disparity = model_performance(train_dataset.dataset.Y[train_dataset.indices].detach().numpy(), train_dataset.dataset.Z[train_dataset.indices].detach().numpy(), Yhat_train, Yhat_max_train, tau)
        accuracies.append(accuracy)
        dp_disparities.append(dp_disparity)
        eo_disparities.append(eo_disparity)
        eodd_disparities.append(eodd_disparity)
        ei_disparities.append(ei_disparity)
        be_disparities.append(be_disparity)



    # store the list of training results into the 
    results.train_imp_all = imp_all
    results.train_imp_a = imp_a
    results.train_imp_b = imp_b
    results.train_acc_hist = accuracies
    results.train_d_loss_hist = d_losses
    results.train_f_loss_hist = f_losses
    results.train_dp_hist = dp_disparities      
    results.train_eo_hist = eo_disparities  
    results.train_eodd_hist = eodd_disparities  
    results.train_ei_hist = ei_disparities  
    results.train_be_hist = be_disparities
    results.train_aw_a = aw_a
    results.train_aw_b = aw_b
    results.train_aw_all = aw_all
    results.train_safety = sfty

    # Validation set
    Yhat_val = model(val_dataset.dataset.X[val_dataset.indices]).reshape(-1).detach().numpy()
    Yhat_max_val, Yhat_total_val = Grad_effort(model, h, val_dataset.dataset.X[val_dataset.indices], delta_effort, ctv = ctv)


    imp_all= 0
    # get improvement measures
    x_start = val_dataset.dataset.X[val_dataset.indices]
    y_true = h(x_start).detach().numpy()
    z_true = val_dataset.dataset.Z[val_dataset.indices].detach().numpy()
    imp_all = np.mean(Yhat_total_val.detach().numpy()-y_true)

    imp_a, imp_b = 0,0
    y0 = h(x_start[z_true==0]).detach().numpy()
    y1 = h(x_start[z_true==1]).detach().numpy()
    ystar0 = Yhat_total_val[z_true == 0].detach().numpy()
    ystar1 = Yhat_total_val[z_true == 1].detach().numpy()
    # safety
    m0  = ystar0 - y0
    m1 = ystar1 - y1
    m0 = np.where(m0 > 0, 0, m0)
    m1 = np.where(m1 > 0, 0, m1)
    sfty = len(m0)/(len(m0)+len(m1))*np.mean(m0) + len(m1)/(len(m0)+len(m1))*np.mean(m1)

    imp_a += np.mean((ystar0-y0))
    imp_b += np.mean((ystar1-y1))

    Yhat_max_val = Yhat_max_val.reshape(-1).detach().numpy()

    results.val_imp_all = imp_all
    results.val_imp_a = imp_a
    results.val_imp_b = imp_b
    aw_a, aw_b, aw_all =  aw_calc(z_true,Yhat_val,y_true)
    results.val_aw_a = aw_a
    results.val_aw_b = aw_b
    results.val_aw_all = aw_all
    results.val_safety = sfty

    results.val_acc, results.val_dp, results.val_eo, results.val_eodd, results.val_ei, results.val_be = model_performance(val_dataset.dataset.Y[val_dataset.indices].detach().numpy(), val_dataset.dataset.Z[val_dataset.indices].detach().numpy(), Yhat_val, Yhat_max_val, tau)

    # Testing set
    Yhat_test = model(test_dataset.X).reshape(-1).detach().numpy()
    Yhat_max_test, Yhat_total_test = Grad_effort(model, h, test_dataset.X, delta_effort,ctv=ctv)


    imp_all = 0   
    # get improvement measures
    x_start = test_dataset.X
    y_true = h(x_start).detach().numpy()
    z_true = Z_test.detach().numpy()
    imp_all = np.mean(Yhat_total_test.detach().numpy()-y_true)
    aw_a, aw_b, aw_all =  aw_calc(z_true,Yhat_test,y_true)

    imp_a, imp_b = 0,0
    y0 = h(x_start[z_true == 0]).detach().numpy()
    y1 = h(x_start[z_true == 1]).detach().numpy()
    ystar0 = Yhat_total_test[z_true == 0].detach().numpy()
    ystar1 = Yhat_total_test[z_true == 1].detach().numpy()

    imp_a += np.mean((ystar0-y0))
    imp_b += np.mean((ystar1-y1))
    # safety
    m0  = ystar0 - y0
    m1 = ystar1 - y1
    m0 = np.where(m0 > 0, 0, m0)
    m1 = np.where(m1 > 0, 0, m1)
    sfty = len(m0)/(len(m0)+len(m1))*np.mean(m0) + len(m1)/(len(m0)+len(m1))*np.mean(m1)

    results.test_imp_all = imp_all
    results.test_imp_a = imp_a
    results.test_imp_b = imp_b
    results.test_aw_a = aw_a
    results.test_aw_b = aw_b
    results.test_aw_all = aw_all
    results.test_safety = sfty
    Yhat_max_test = Yhat_max_test.reshape(-1).detach().numpy()
    results.test_acc, results.test_dp, results.test_eo, results.test_eodd, results.test_ei, results.test_be = model_performance(Y_test.detach().numpy(), Z_test.detach().numpy(), Yhat_test, Yhat_max_test, tau)

    return results



def trainer_new(model, h, dataset, optimizer, device, n_epochs, batch_size,z_blind, delta_effort=1,lambda2_=0,lambda3_=0,ctv = None,add_safe=False):
    '''
    Training function for fairness algos: EI, BE;
    Training function for SAFE, MaxImp, TotalImp, together with an applicant welfare guranttee (h(x) and f(x) have high corr)
    Also the applicant welfare regularizer can be added to prevent worse applicants get higher probability to be admitted
    '''

    relu = torch.nn.ReLU()
    train_tensors, test_tensors = dataset.get_dataset_in_tensor()
    X_train, Y_train, Z_train, XZ_train = train_tensors
    X_test, Y_test, Z_test, XZ_test = test_tensors

    sensitive_attrs = dataset.sensitive_attrs
    if z_blind is True:
        train_dataset = FairnessDataset(X_train, Y_train, Z_train)
        test_dataset = FairnessDataset(X_test, Y_test, Z_test)
    else:
        train_dataset = FairnessDataset(XZ_train, Y_train, Z_train)
        test_dataset = FairnessDataset(XZ_test, Y_test, Z_test)
    
    train_dataset, val_dataset = random_split(train_dataset,[int(0.8*len(train_dataset)),len(train_dataset)-int(0.8*len(train_dataset))])
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    
    tau = 0.5


    results = SimpleNamespace()
    loss_func = torch.nn.BCELoss(reduction = 'mean')

    # decision-maker welfare loss
    d_losses = []
    # improvement loss
    imp_losses = []
    # safety loss
    sf_losses = []
    # agent welfare loss
    aw_losses = []


    # decision-maker welfare
    accuracies = []
    # social welfare
    imp_all = []
    imp_a = []
    imp_b = []
    sfty = []
    # applicant welfare
    aw_a = []
    aw_b = []
    aw_all = []


    # fairness
    dp_disparities = []
    eo_disparities = []
    eodd_disparities = []
    ei_disparities = []
    be_disparities = []


    for epoch in tqdm.trange(n_epochs, desc="Training", unit="epochs"):
        
        local_d_loss = []
        local_imp_loss = []
        local_sf_loss = []
        local_aw_loss = []

        for _, (x_batch, y_batch, z_batch) in enumerate(train_loader):
            x_batch, y_batch, z_batch = x_batch.to(device), y_batch.to(device), z_batch.to(device)
            Yhat = model(x_batch)
            
            cost = 0
            imp_loss = 0
            sf_loss = 0
            aw_loss = 0

            # L_DW
            d_loss = loss_func(Yhat.reshape(-1), y_batch)
            cost += d_loss

            # L_imp
            Yhat_max, Yhat_total = Grad_effort(model, h, x_batch, delta_effort, ctv = ctv)

            imp_loss = loss_func(Yhat_total.reshape(-1), torch.ones(len(Yhat_total)))               
            cost += (lambda2_*imp_loss)

            # L_sf
            Yhat_violate = Yhat_total[h(x_batch)> Yhat_total]
            sf_loss = loss_func(Yhat_violate.reshape(-1), torch.ones(len(Yhat_violate)))
            if add_safe and (epoch > n_epochs/4 or n_epochs == 50):
                cost += (lambda2_*sf_loss)

            # L_aw
            model_violate = model(x_batch)[h(x_batch)> model(x_batch)]               
            aw_loss = loss_func(model_violate.reshape(-1), torch.ones(len(model_violate)))
            cost += (lambda3_* aw_loss)

            # optimize step
            optimizer.zero_grad()
            if (torch.isnan(cost)).any():
                continue
            cost.backward()
            optimizer.step()

            local_d_loss.append(d_loss.item())
            local_imp_loss.append(imp_loss.item())
            local_sf_loss.append(sf_loss.item())
            local_aw_loss.append(aw_loss.item())

                        
        d_losses.append(np.mean(local_d_loss))
        imp_losses.append(np.mean(local_imp_loss))
        sf_losses.append(np.mean(local_sf_loss))
        aw_losses.append(np.mean(local_aw_loss))
        

        Yhat_train = model(train_dataset.dataset.X[train_dataset.indices]).reshape(-1).detach().numpy()

        # get the final best responses
        Yhat_max_train, Yhat_total_train = Grad_effort(model, h, train_dataset.dataset.X[train_dataset.indices], delta_effort, ctv=ctv)
        
        
        x_start = train_dataset.dataset.X[train_dataset.indices]
        y_true = h(x_start).detach().numpy()
        z_true = train_dataset.dataset.Z[train_dataset.indices].detach().numpy()

        # get improvement measures
        imp_all.append(np.mean(Yhat_total_train.detach().numpy() - y_true))

        if epoch == n_epochs - 1:
            welfare_a, welfare_b, welfare_all = aw_calc(z_true,Yhat_train,y_true)
            aw_a.append(welfare_a)
            aw_b.append(welfare_b)
            aw_all.append(welfare_all)

        # imp_a, imp_b
        aimp, bimp = 0,0
        y0 = h(x_start[z_true == 0]).detach().numpy()
        y1 = h(x_start[z_true == 1]).detach().numpy()
        ystar0 = Yhat_total_train[z_true == 0].detach().numpy()
        ystar1 = Yhat_total_train[z_true == 1].detach().numpy()

        aimp += np.mean((ystar0-y0))
        bimp += np.mean((ystar1-y1))
        m0  = ystar0 - y0
        m1 = ystar1 - y1
        m0 = np.where(m0 > 0, 0, m0)
        m1 = np.where(m1 > 0, 0, m1)
        saf = len(m0)/(len(m0)+len(m1))*np.mean(m0) + len(m1)/(len(m0)+len(m1))*np.mean(m1)

        sfty.append(saf)
        imp_a.append(aimp)
        imp_b.append(bimp)

        Yhat_max_train = Yhat_max_train.reshape(-1).detach().numpy()
        # still can get fairness measures
        accuracy, dp_disparity, eo_disparity, eodd_disparity, ei_disparity, be_disparity = model_performance(train_dataset.dataset.Y[train_dataset.indices].detach().numpy(), train_dataset.dataset.Z[train_dataset.indices].detach().numpy(), Yhat_train, Yhat_max_train, tau)
        accuracies.append(accuracy)
        dp_disparities.append(dp_disparity)
        eo_disparities.append(eo_disparity)
        eodd_disparities.append(eodd_disparity)
        ei_disparities.append(ei_disparity)
        be_disparities.append(be_disparity)

    results.train_imp_a = imp_a
    results.train_imp_b = imp_b
    results.train_imp_all = imp_all
    results.train_acc_hist = accuracies
    results.train_d_loss_hist = d_losses
    results.train_imp_loss_hist = imp_losses
    results.train_sf_loss_hist = sf_losses
    results.train_aw_loss_hist = aw_losses

    results.train_dp_hist = dp_disparities      
    results.train_eo_hist = eo_disparities
    results.train_eodd_hist = eodd_disparities  
    results.train_ei_hist = ei_disparities  
    results.train_be_hist = be_disparities
    results.train_aw_a = aw_a
    results.train_aw_b = aw_b
    results.train_aw_all = aw_all
    results.train_safety = sfty


    Yhat_val = model(val_dataset.dataset.X[val_dataset.indices]).reshape(-1).detach().numpy()

    Yhat_max_val, Yhat_total_val = Grad_effort(model, h, val_dataset.dataset.X[val_dataset.indices], delta_effort, ctv = ctv)


    imp_all = 0

    # get improvement measures
    x_start = val_dataset.dataset.X[val_dataset.indices]
    y_true = h(x_start).detach().numpy()
    z_true = val_dataset.dataset.Z[val_dataset.indices].detach().numpy()
    imp_all = np.mean(Yhat_total_val.detach().numpy()-y_true)

    aw_a, aw_b, aw_all =  aw_calc(z_true,Yhat_val,y_true)

    imp_a, imp_b = 0,0
    y0 = h(x_start[z_true==0]).detach().numpy()
    y1 = h(x_start[z_true==1]).detach().numpy()
    ystar0 = Yhat_total_val[z_true == 0].detach().numpy()
    ystar1 = Yhat_total_val[z_true == 1].detach().numpy()

    aimp += np.mean((ystar0-y0))
    bimp += np.mean((ystar1-y1))

    m0  = ystar0 - y0
    m1 = ystar1 - y1
    m0 = np.where(m0 > 0, 0, m0)
    m1 = np.where(m1 > 0, 0, m1)
    sfty = len(m0)/(len(m0)+len(m1))*np.mean(m0) + len(m1)/(len(m0)+len(m1))*np.mean(m1)

    Yhat_max_val = Yhat_max_val.reshape(-1).detach().numpy()

    results.val_imp_all = imp_all
    results.val_imp_a = aimp
    results.val_imp_b = bimp
    results.val_acc, results.val_dp, results.val_eo, results.val_eodd, results.val_ei, results.val_be = model_performance(val_dataset.dataset.Y[val_dataset.indices].detach().numpy(), val_dataset.dataset.Z[val_dataset.indices].detach().numpy(), Yhat_val, Yhat_max_val, tau)

    results.val_aw_a = aw_a
    results.val_aw_b = aw_b
    results.val_aw_all = aw_all
    results.val_safety = sfty


    Yhat_test = model(test_dataset.X).reshape(-1).detach().numpy()
    Yhat_max_test, Yhat_total_test = Grad_effort(model, h, test_dataset.X, delta_effort, ctv=ctv)


    imp_all = 0
  
    # get improvement measures
    x_start = test_dataset.X
    y_true = h(x_start).detach().numpy()
    z_true = Z_test.detach().numpy()
    imp_all = np.mean(Yhat_total_test.detach().numpy()-y_true)

    aw_a, aw_b, aw_all = aw_calc(z_true,Yhat_test,y_true)


    imp_a, imp_b = 0,0
    y0 = h(x_start[z_true == 0]).detach().numpy()
    y1 = h(x_start[z_true == 1]).detach().numpy()
    ystar0 = Yhat_total_test[z_true == 0].detach().numpy()
    ystar1 = Yhat_total_test[z_true == 1].detach().numpy()

    imp_a += np.mean((ystar0-y0))
    imp_b += np.mean((ystar1-y1))

    # safety
    m0  = ystar0 - y0
    m1 = ystar1 - y1
    m0 = np.where(m0 > 0, 0, m0)
    m1 = np.where(m1 > 0, 0, m1)
    sfty = len(m0)/(len(m0)+len(m1))*np.mean(m0) + len(m1)/(len(m0)+len(m1))*np.mean(m1)


    results.test_imp_a = imp_a
    results.test_imp_b = imp_b
    results.test_imp_all = imp_all
    Yhat_max_test = Yhat_max_test.reshape(-1).detach().numpy()
    results.test_acc, results.test_dp, results.test_eo, results.test_eodd, results.test_ei, results.test_be = model_performance(Y_test.detach().numpy(), Z_test.detach().numpy(), Yhat_test, Yhat_max_test, tau)
    results.test_aw_a = aw_a
    results.test_aw_b = aw_b
    results.test_aw_all = aw_all
    results.test_safety = sfty

    return results

