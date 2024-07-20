import torch 

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import nimfa

##############################################
            # LOGISTIC REGRESSION #
##############################################

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes, bias=False)  # Linear layer for prediction

    def forward(self, x):
        x = self.linear(x)
        return x  # No activation for multiclass (softmax applied in loss function)
    
    def train(self, X, Y, learning_rate=0.01, epochs=100, batch_size=32, verbose=False, device='cpu'):
        self.to(device) 
        #self.linear.weight.data = torch.linalg.solve(X.T@X, X.T@Y).T
        #new part - due to singular matrix

        ####### ORIGINAL ########
        #XTX = X.T @ X + 1e-3 * torch.eye(X.shape[1], device=device)
        #self.linear.weight.data = torch.linalg.solve(XTX, X.T @ Y).T

        ######## PINV ########
        X_pinv = torch.linalg.pinv(X.T @ X)
        self.linear.weight.data = (X_pinv @ X.T @ Y).T

    def eval(self, X, Y):
        correct = 0
        total = Y.size(0)
        pred = self(X)
        for idx in range(total):
            if torch.argmax(pred[idx]) == torch.argmax(Y[idx]):
                correct+=1

        return correct/total
    

class LogisticRegressionGD(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegressionGD, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

    def train_model(self, X, Y, learning_rate=0.01, epochs=100, batch_size=32, verbose=False, device='cpu'):
        self.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        
        dataset = torch.utils.data.TensorDataset(X, Y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            for batch_X, batch_Y in dataloader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                
                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = criterion(outputs, torch.argmax(batch_Y, dim=1))
                loss.backward()
                optimizer.step()
                
            if verbose and (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    def eval_model(self, X, Y):
        with torch.no_grad():
            outputs = self(X)
            _, predicted = torch.max(outputs, 1)
            _, labels = torch.max(Y, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
        
        return correct / total


##############################################
                # COVAR #
##############################################

class COVAR(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(COVAR, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes, bias=False)  # Linear layer for prediction
        self.weighter = nn.Linear(input_dim, 1, bias=False)

    def forward(self, x):
        x = self.linear(x)
        return x  # No activation for multiclass (softmax applied in loss function)
    
    def discriminator(self, x):
        x = self.weighter(x)
        return torch.sigmoid(x)

    def weighting(self, X, Q, learning_rate=0.01, epochs=100, batch_size=32, verbose=False, device='cpu'):
        self.to(device)
        X_new = torch.cat([X.to(device),Q.to(device)],0)
        Y_new = torch.cat([torch.ones(X.size(0),1),torch.zeros(Q.size(0),1)],0)
        self.weighter.weight.data = torch.linalg.solve(X_new.T@X_new, X_new.T@Y_new).T 

    def train(self, X, Y, Q, learning_rate=0.01, epochs=100, batch_size=32, verbose=False, device='cpu'):
        self.to(device)  # Move model to available device
        self.weighting(X, Q, learning_rate=learning_rate, epochs=epochs, batch_size=batch_size, verbose=verbose, device=device)

        weight = (1/self.discriminator(X)) - 1
        Y = weight * Y

        self.linear.weight.data = torch.linalg.solve(X.T@X, X.T@Y).T

    def eval(self, X, Y):
        correct = 0
        total = Y.size(0)
        pred = self(X)
        for idx in range(total):
            if torch.argmax(pred[idx]) == torch.argmax(Y[idx]):
                correct+=1

        return correct/total


##############################################
            # MFREG Gradient #
##############################################

class MFRegGrad(nn.Module):
    def __init__(self, dimx, dimz, dimy, dimw, dimu):
        super(MFRegGrad, self).__init__()
        self.linear_uy = nn.Linear(dimu, dimy)  
        self.linear_uw = nn.Linear(dimu, dimy) 
        self.linear_zu = nn.Linear(dimz, dimu) 
        self.linear_xu = nn.Linear(dimx, dimu)  
        self.linear = nn.Linear(dimx, dimy)
        self.dimu = dimu
        self.dimy = dimy

    def forward(self, x):
        x = self.linear(x)
        return x  # No activation for multiclass (softmax applied in loss function)
    
    def pred(self, x):
        XU = self.linear_xu(x)
        UY = self.linear_uy(F.one_hot(torch.arange(0, self.dimu)).float())
        Y = torch.zeros(x.size(0),self.dimy)
        for i in range(self.dimy):
            for j in range(self.dimu):
                Y[:,i] += UY[j,i]*XU[:,j]
        return Y

        # Y = self.linear(x) #Y = F.softmax(self.linear(x),dim=-1)
        # U = self.linear_xu(x) #U = F.softmax(self.linear_xu(x),dim=-1)
        # Q = torch.cat([P[...,None]@Q[:,i][...,None,None] for i in range(x.size(-1))],-1)
        # return Q

    def first_step(self, Z, Y, W, learning_rate=0.01, epochs=100, batch_size=32, verbose=False, device='cpu'):
        self.to(device)  # Move model to available device

        dataset1 = TensorDataset(Z.to(device), Y.to(device))
        dataloader1 = DataLoader(dataset1, batch_size=batch_size, shuffle=True)

        dataset2 = TensorDataset(Z.to(device), W.to(device))
        dataloader2 = DataLoader(dataset2, batch_size=batch_size, shuffle=True)

        # Define loss function and optimizer (using multinomial logistic loss)
        criterion = nn.CrossEntropyLoss()  # Multinomial logistic loss for multiclass
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            for i, ((data1, targety), (data2, targetw)) in enumerate(zip(dataloader1,dataloader2)):
                # Forward pass
                # z = F.softmax(self.linear_zu(data1),dim=-1)
                z = self.linear_zu(data1)
                w = self.linear_uw(z)
                y = self.linear_uy(z)

                # Calculate loss
                loss = criterion(y, targety) + criterion(w, targetw)

                # Backward pass and update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if verbose and (i % 100 == 0):
                    print(f"First step: {epoch+1}/{epochs} | Batch: {i+1}/{len(dataloader1)} | Loss: {loss.item():.6f}")

    
    def train(self, X, Y, Z, W, learning_rate=0.01, epochs=100, batch_size=32, verbose=False, device='cpu'):
        self.to(device)  # Move model to available device
        self.first_step(Z, Y, W, learning_rate=learning_rate, epochs=epochs, batch_size=batch_size, verbose=verbose, device=device)

    def adapt(self, X, W, learning_rate=0.01, epochs=100, batch_size=32, verbose=False, device='cpu'):
        self.to(device)  # Move model to available device
        for param in self.linear_uw.parameters():
            param.requires_grad = False

        # Convert data to TensorDataset and DataLoader
        dataset = TensorDataset(X.to(device), W.to(device))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Define loss function and optimizer (using multinomial logistic loss)
        criterion = nn.CrossEntropyLoss()  # Multinomial logistic loss for multiclass
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            for i, (data, target) in enumerate(dataloader):
                # Forward pass
                # u = F.softmax(self.linear_xu(data),dim=-1)
                u = self.linear_xu(data)
                w = self.linear_uw(u)

                # Calculate loss
                loss = criterion(w, target)

                # Backward pass and update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if verbose and (i % 100 == 0):
                    print(f"Adapt: {epoch+1}/{epochs} | Batch: {i+1}/{len(dataloader)} | Loss: {loss.item():.6f}")


    def eval_test(self, X, Y):
        correct = 0
        total = Y.size(0)
        pred = self.pred(X)
        for idx in range(total):
            if torch.argmax(pred[idx]) == torch.argmax(Y[idx]):
                correct+=1

        return correct/total

    def eval_train(self, X, Y):
        correct = 0
        total = Y.size(0)
        pred = self(X)
        for idx in range(total):
            if torch.argmax(pred[idx]) == torch.argmax(Y[idx]):
                correct+=1

        return correct/total

    def eval_u(self, X, U):
        correct = 0
        total = U.size(0)
        pred = self.linear_xu(X)
        for idx in range(total):
            if torch.argmax(pred[idx]) == torch.argmax(U[idx]):
                correct+=1

        return correct/total


##############################################
            # MFREG Factorisation #
##############################################

class MFReg(nn.Module):
    def __init__(self, dimx, dimz, dimy, dimw, dimu):
        super(MFReg, self).__init__()
        self.UY = torch.rand(dimu+dimx,dimy)
        self.UW = torch.rand(dimu,dimw)
        self.xZU = torch.rand(dimx,dimz,dimu)
        self.XW = torch.rand(dimx,dimw)
        self.XW_train = torch.rand(dimx,dimw)
        self.dimu = dimu
        self.dimx = dimx
        self.dimz = dimz
        self.dimy = dimy
        self.dimw = dimw 

    def forward(self, x):
        x = self.linear(x)
        return x  # No activation for multiclass (softmax applied in loss function)
    
    def pred(self, x):
        # XW = x@self.XW
        # XU = torch.linalg.solve(self.UW, XW, left=False)
        XU = x@self.XW@torch.linalg.pinv(self.UW)
        # XU @ UW = XW 
        UY = F.one_hot(torch.arange(0, self.dimu)).float()@self.UY
        Y = torch.zeros(x.size(0),self.dimy)
        for i in range(self.dimy):
            for j in range(self.dimu):
                Y[:,i] += UY[j,i]*XU[:,j]
        return Y

    def pred_train(self, x):
        # XW = x@self.XW
        # XU = torch.linalg.solve(self.UW, XW, left=False)
        XU = x@self.XW_train@torch.linalg.pinv(self.UW)
        # XU @ UW = XW 
        UY = F.one_hot(torch.arange(0, self.dimu)).float()@self.UY
        Y = torch.zeros(x.size(0),self.dimy)
        for i in range(self.dimy):
            for j in range(self.dimu):
                Y[:,i] += UY[j,i]*XU[:,j]
        return Y

    def first_step(self, X, Z, Y, W, device='cpu'):
        self.to(device)  # Move model to available device
        # Z = torch.cat([Z,X],1)

        # ZY = torch.linalg.solve(Z.T@Z, Z.T@Y) 
        # ZW = torch.linalg.solve(Z.T@Z, Z.T@W)
        xZY = torch.zeros(self.dimx,self.dimz,self.dimy)
        xZW = torch.zeros(self.dimx,self.dimz,self.dimw)
        for x in range(self.dimx):
            for i in range(self.dimz):
                for j in range(self.dimy):
                    xZY[x,i,j] = torch.logical_and(X[x] == 1,Z[i] == 1, Y[j] == 1).sum().float() / torch.logical_and(X[x] == 1,Z[i] == 1).sum().float()
                for k in range(self.dimw):
                    xZW[x,i,k] = torch.logical_and(X[x] == 1,Z[i] == 1, W[k] == 1).sum().float() / torch.logical_and(X[x] == 1,Z[i] == 1).sum().float()

        xZY = xZY.reshape(self.dimx*self.dimz,self.dimy)
        xZW = xZY.reshape(self.dimx*self.dimz,self.dimw)

        stacked = torch.cat([xZW,xZY],0) 

        nmf = nimfa.SepNmf(stacked.cpu().detach().numpy(), seed='random_vcol', rank=self.dimu, max_iter=10000)
        nmf_fit = nmf()

        xZU = torch.from_numpy(nmf_fit.coef()).float().reshape(self.dimx,self.dimz,self.dimu)
        basis = nmf_fit.basis()
        # print(basis.shape)
        UW,UY = torch.from_numpy(basis[:self.dimw,:]).float(),torch.from_numpy(basis[self.dimw:,:]).float()

        # print(ZU.shape,UW.shape,UY.shape)

        self.xZU = xZU.T
        self.UW = UW.T
        self.UY = UY.T


    
    def train(self, X, Y, Z, W, verbose=False, device='cpu'):
        self.to(device)  # Move model to available device
        self.first_step(X, Z, Y, W, device=device)
        self.XW_train = torch.linalg.solve(X.T@X, X.T@W) 

    def adapt(self, X, W, learning_rate=0.01, epochs=100, batch_size=32, verbose=False, device='cpu'):
        self.to(device)  # Move model to available device
        self.XW = torch.linalg.solve(X.T@X, X.T@W) 

    def eval_test(self, X, Y):
        correct = 0
        total = Y.size(0)
        pred = self.pred(X)
        for idx in range(total):
            if torch.argmax(pred[idx]) == torch.argmax(Y[idx]):
                correct+=1

        return correct/total

    def eval_train(self, X, Y):
        correct = 0
        total = Y.size(0)
        pred = self.pred_train(X)
        for idx in range(total):
            if torch.argmax(pred[idx]) == torch.argmax(Y[idx]):
                correct+=1

        return correct/total

    def eval_u(self, X, U):
        correct = 0
        total = U.size(0)
        pred = X@self.XW@torch.linalg.pinv(self.UW)
        for idx in range(total):
            if torch.argmax(pred[idx]) == torch.argmax(U[idx]):
                correct+=1

        return correct/total

    def eval_w(self, X, W):
        correct = 0
        total = W.size(0)
        pred = X@self.XW
        for idx in range(total):
            if torch.argmax(pred[idx]) == torch.argmax(W[idx]):
                correct+=1

        return correct/total




##############################################
                # ORACLE #
##############################################

class Oracle:
    def oracle(self,x,w,u,theta_yx,theta_yw):
        proby = (torch.tensor([0.1,0.1,0.4,0.4]) * u + torch.tensor([0.4,0.4,0.1,0.1]) * (1-u)).unsqueeze(0)
        y = torch.argmax(proby * (theta_yx @ x.T + theta_yw @ w.T).T, dim=1)
        # print(y)
        return y

    def eval(self, X,W,U,Y,theta_yx,theta_yw):
        correct = 0
        total = Y.size(0)
        for idx in range(total):
            if self.oracle(X[idx],W[idx],U[idx].item(),theta_yx,theta_yw) == torch.argmax(Y[idx]):
                correct+=1

        return correct/total



