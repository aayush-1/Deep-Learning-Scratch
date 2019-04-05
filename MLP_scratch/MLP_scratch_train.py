import math
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



#*********************************************************************************************************

def sigmoid(gamma):
    if gamma < 0:
        return 1 - 1/(1 + math.exp(gamma))
    else:
        return 1/(1 + math.exp(-gamma))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def relu(x):
    if(x>0):
        return x
    elif(x<=0):
        return 0

    
def relu_dev(x):
    if x>0:
        return 1
    else:
        return 0.11
    
def cross_entropy_error(output_vec,y_train):
    e=0
    for i in range(output_vec.shape[0]):
        if (output_vec[i]!=0):
            e=e + (-1)*y_train[i]*np.log10(output_vec[i]+1e-8)
    return e

def tanh_deriv(x):
    return 1.0 - np.tanh(x)**2


#*********************************************************************************************************


def main():
    data=pd.read_csv('pendigits_train.txt',header=None)
    x=data.values[:,:16]
    y=data.values[:,16]

    y1 = np.zeros((y.shape[0], 10))
    y1[np.arange(y.shape[0]), y] = 1

    #**********************************************************************************************************


    #inputs
    MLP_input_size=x.shape[1]  #depends on data
    no_hidden_layers=int(input('\nenter hidden layers for MLP: '))
    size_hidden_layer=[]
    size_hidden_layer.append(MLP_input_size)
    non_linear_fn=[] 
    for i in range(0,no_hidden_layers):
        print("hidden layer: ",i)
        size_hidden_layer.append(int(input('size of corresponding hidden layer: ')))
        non_linear_fn.append(input('non linearity fn(tanh or sigmoid) for corresponding layer: '))

    output_size=10 #depends on data
    size_hidden_layer.append(output_size)
    non_linear_fn.append("softmax")

    #*********************************************************************************************************
    gamma=0.01
    beta=0.01
    weights_MLP=[]
    weights_MLP.append(np.random.normal(0,0.1,(MLP_input_size+1, size_hidden_layer[1])))
    for i in range(1,no_hidden_layers):
        weights_MLP.append(np.random.normal(0,0.01,(size_hidden_layer[i]+1,size_hidden_layer[i+1])))
    weights_MLP.append(np.random.normal(0,0.01,(size_hidden_layer[no_hidden_layers]+1,output_size)))

    layers_MLP=[]
    for i in range(0,no_hidden_layers+2):
        layers_MLP.append(np.zeros((size_hidden_layer[i])))

    optimizer=input("optimization method: ")
    if optimizer=="SGD":
        SGD_train(x,y1,gamma,beta,MLP_input_size,no_hidden_layers,size_hidden_layer,non_linear_fn,weights_MLP,layers_MLP,output_size)
    elif optimizer=="momentum":
        momentum_train(x,y1,gamma,beta,MLP_input_size,no_hidden_layers,size_hidden_layer,non_linear_fn,weights_MLP,layers_MLP,output_size)
    elif optimizer=="nesterov":
        nesterov_train(x,y1,gamma,beta,MLP_input_size,no_hidden_layers,size_hidden_layer,non_linear_fn,weights_MLP,layers_MLP,output_size)
    elif optimizer=="adagrad":
        adagrad_train(x,y1,gamma,beta,MLP_input_size,no_hidden_layers,size_hidden_layer,non_linear_fn,weights_MLP,layers_MLP,output_size)
    elif optimizer=="RMSprop":
        RMSprop_train(x,y1,gamma,beta,MLP_input_size,no_hidden_layers,size_hidden_layer,non_linear_fn,weights_MLP,layers_MLP,output_size)
    elif optimizer=="adam":
        adam_train(x,y1,gamma,beta,MLP_input_size,no_hidden_layers,size_hidden_layer,non_linear_fn,weights_MLP,layers_MLP,output_size)

        

#*********************************************************************************************************

def batchnorm_forward(X, gamma, beta):
    mu = np.mean(X, axis=0)
    var = np.var(X, axis=0)

    X_norm = (X - mu) / np.sqrt(var + 1e-8)
    out = gamma * X_norm + beta

    cache = (X, X_norm, mu, var, gamma, beta)

    return out, mu, var, X_norm


#*********************************************************************************************************

def batchnorm_backward(dout, X,X_norm, mu, var, gamma, beta):
    lr=0.01
    N= X.shape

    X_mu = X - mu
    std_inv = 1. / np.sqrt(var + 1e-8)

    dX_norm = dout * gamma
    dvar = np.sum(dX_norm * X_mu, axis=0) * -.5 * std_inv**3
    dmu = np.sum(dX_norm * -std_inv, axis=0) + dvar * np.mean(-2. * X_mu, axis=0)

    dX = (dX_norm * std_inv) + (dvar * 2 * X_mu / N) + (dmu / N)
    dgamma = np.sum(dout * X_norm, axis=0)
    dbeta = np.sum(dout, axis=0)
    gamma+=-lr*dgamma
    beta+=-lr*dbeta

    return dX,gamma,beta

#*********************************************************************************************************

def MLP(mlp_input,no_hidden_layers,size_hidden_layer,non_linear_fn,output_size,weights_MLP,layers_MLP,gamma,beta):
#     o=np.zeros((size_hidden_layer[0]))
    layers_MLP[0]=mlp_input
    layers_MLP_batchnorm=[]
    layers_MLP_nlf=[]
    mu_list=[]
    var_list=[]
    X_norm_list=[]
    for i in range(0,no_hidden_layers+1):
        if i==0:
            layers_MLP[i+1]=np.matmul(np.transpose(layers_MLP[i]),weights_MLP[i][1:,:])+weights_MLP[i][0,:]
        else:
            layers_MLP[i+1]=np.matmul(np.transpose(layers_MLP_nlf[i-1]),weights_MLP[i][1:,:])+weights_MLP[i][0,:]
        batch_norm, mu, var, X_norm= batchnorm_forward(layers_MLP[i+1], gamma, beta)
        layers_MLP_batchnorm.append(batch_norm)
        mu_list.append(mu)
        var_list.append(var)
        X_norm_list.append(X_norm)
        
        if non_linear_fn[i]=="relu":
            aa=np.zeros((layers_MLP[i+1].shape))
            for h in range(0,size_hidden_layer[i+1]):
                aa[h]=(relu(batch_norm[h]))
            layers_MLP_nlf.append(aa)
        
        if non_linear_fn[i]=="tanh":
            layers_MLP_nlf.append(np.tanh(batch_norm))
            #print("yess")
            
        if non_linear_fn[i]=="softmax":
            layers_MLP_nlf.append(softmax(batch_norm))

    return layers_MLP,layers_MLP_batchnorm,layers_MLP_nlf,mu_list,var_list,X_norm_list,gamma,beta

#layers_MLP,layers_MLP_batchnorm,layers_MLP_nlf,mu_list,var_list,X_norm_list,gamma,beta=MLP(x[0,:],no_hidden_layers,size_hidden_layer,non_linear_fn,output_size,weights_MLP,layers_MLP,gamma,beta)



#*********************************************************************************************************


def MLP_backprop_SGD(y_tr,no_hidden_layers,size_hidden_layer,weights_MLP,layers_MLP,learning_rate,gamma,beta,layers_MLP_batchnorm,layers_MLP_nlf,mu_list,var_list,X_norm_list):
    x=0
    for i in range(no_hidden_layers,-1,-1):
        a=weights_MLP[i].shape[0]
        b=weights_MLP[i].shape[1]
        if x==0:
            dx=np.zeros((size_hidden_layer[i+1]))
            x=1
        else:
            dy=dx
            dx=np.zeros((size_hidden_layer[i+1]))
        
        
        if i==no_hidden_layers:
            dx=(-1)*(y_tr)*(1-(layers_MLP_nlf[i]))
        else:
            for j in range(0,b):
                dx[j]=np.dot(dy,weights_MLP[i+1][j+1,:])*relu_dev(layers_MLP_nlf[i][j])
               
        dx,gamma,beta=batchnorm_backward(dx,layers_MLP_batchnorm[i] ,X_norm_list[i], mu_list[i], var_list[i], gamma, beta)
        #print("dx ki value",dx) 
        for j in range(0,b):
            weights_MLP[i][1:,j]=weights_MLP[i][1:,j]-learning_rate*dx[j]*layers_MLP[i]
            weights_MLP[i][0,i]=weights_MLP[i][0,i]-learning_rate*dx[j]
            
    return weights_MLP,gamma,beta

#weights_MLP,gamma,beta=MLP_backprop_SGD(y1[0,:],no_hidden_layers,size_hidden_layer,weights_MLP,layers_MLP,0.001,gamma,beta,layers_MLP_batchnorm,layers_MLP_nlf,mu_list,var_list,X_norm_list)



#*********************************************************************************************************

def MLP_backprop_momentum(y_tr,no_hidden_layers,size_hidden_layer,weights_MLP,layers_MLP,learning_rate,gamma,beta,layers_MLP_batchnorm,layers_MLP_nlf,mu_list,var_list,X_norm_list,grad):
    x=0
    YY=0.9
    for i in range(no_hidden_layers,-1,-1):
        a=weights_MLP[i].shape[0]
        b=weights_MLP[i].shape[1]
        if x==0:
            dx=np.zeros((size_hidden_layer[i+1]))
            x=1
        else:
            dy=dx
            dx=np.zeros((size_hidden_layer[i+1]))
        
        
        if i==no_hidden_layers:
            dx=(-1)*(y_tr)*(1-(layers_MLP_nlf[i]))
        else:
            for j in range(0,b):
                dx[j]=np.dot(dy,weights_MLP[i+1][j+1,:])*relu_dev(layers_MLP_nlf[i][j])      
        
        dx,gamma,beta=batchnorm_backward(dx,layers_MLP_batchnorm[i] ,X_norm_list[i], mu_list[i], var_list[i], gamma, beta)
        
        grad[i]=(-1)*YY*grad[i] + dx
        for j in range(0,b):
            weights_MLP[i][1:,j]=weights_MLP[i][1:,j]-learning_rate*grad[i][j]*layers_MLP[i]
            weights_MLP[i][0,i]=weights_MLP[i][0,i]-learning_rate*grad[i][j]
            
    return weights_MLP,gamma,beta,grad

#weights_MLP,gamma,beta=MLP_backprop_momentum(y1[0,:],no_hidden_layers,size_hidden_layer,weights_MLP,layers_MLP,0.001,gamma,beta,layers_MLP_batchnorm,layers_MLP_nlf,mu_list,var_list,X_norm_list)



#*********************************************************************************************************

def MLP_backprop_nesterov(y_tr,no_hidden_layers,size_hidden_layer,weights_MLP,layers_MLP,learning_rate,gamma,beta,layers_MLP_batchnorm,layers_MLP_nlf,mu_list,var_list,X_norm_list,grad):
    x=0
    YY=0.9
    for i in range(no_hidden_layers,-1,-1):
        a=weights_MLP[i].shape[0]
        b=weights_MLP[i].shape[1]
        if x==0:
            dx=np.zeros((size_hidden_layer[i+1]))
            x=1
        else:
            dy=dx
            dx=np.zeros((size_hidden_layer[i+1]))
        
        
        if i==no_hidden_layers:
            dx=(-1)*(-1)*(y_tr)*(1-(layers_MLP_nlf[i]))
        else:
            for j in range(0,b):
                dx[j]=np.dot(dy,weights_MLP[i+1][j+1,:])*relu_dev(layers_MLP_nlf[i][j])
             
        
        dx,gamma,beta=batchnorm_backward(dx,layers_MLP_batchnorm[i] ,X_norm_list[i], mu_list[i], var_list[i], gamma, beta)
        grad[i]=(-1)*YY*((-1)*YY*grad[i]+ dx)+ dx
        for j in range(0,b):
            weights_MLP[i][1:,j]=weights_MLP[i][1:,j]-learning_rate*grad[i][j]*layers_MLP[i]
            weights_MLP[i][0,i]=weights_MLP[i][0,i]-learning_rate*grad[i][j]
            
    return weights_MLP,gamma,beta,grad

#weights_MLP,gamma,beta=MLP_backprop_momentum(y1[0,:],no_hidden_layers,size_hidden_layer,weights_MLP,layers_MLP,0.001,gamma,beta,layers_MLP_batchnorm,layers_MLP_nlf,mu_list,var_list,X_norm_list)



#*********************************************************************************************************

def MLP_backprop_adagrad(y_tr,no_hidden_layers,size_hidden_layer,weights_MLP,layers_MLP,learning_rate,gamma,beta,layers_MLP_batchnorm,layers_MLP_nlf,mu_list,var_list,X_norm_list,grad):
    x=0
    YY=0.9
    for i in range(no_hidden_layers,-1,-1):
        a=weights_MLP[i].shape[0]
        b=weights_MLP[i].shape[1]
        if x==0:
            dx=np.zeros((size_hidden_layer[i+1]))
            x=1
        else:
            dy=dx
            dx=np.zeros((size_hidden_layer[i+1]))
        
        
        if i==no_hidden_layers:
            dx=(-1)*(y_tr)*(1-(layers_MLP_nlf[i]))
        else:
            for j in range(0,b):
                dx[j]=np.dot(dy,weights_MLP[i+1][j+1,:])*relu_dev(layers_MLP_nlf[i][j])
                
        dx,gamma,beta=batchnorm_backward(dx,layers_MLP_batchnorm[i] ,X_norm_list[i], mu_list[i], var_list[i], gamma, beta)

        grad[i]=np.sqrt((grad[i])**2 + dx**2)
        for j in range(0,b):
            weights_MLP[i][1:,j]=weights_MLP[i][1:,j]-(learning_rate*dx[j]*layers_MLP[i])/(grad[i][j]+1e-8)
            
            weights_MLP[i][0,i]=weights_MLP[i][0,i]-(learning_rate*dx[j])/(grad[i][j]+1e-8)
            

        
           
    return weights_MLP,gamma,beta,grad


#weights_MLP,gamma,beta,grad=MLP_backprop_adagrad(y1[0,:],no_hidden_layers,size_hidden_layer,weights_MLP,layers_MLP,0.001,gamma,beta,layers_MLP_batchnorm,layers_MLP_nlf,mu_list,var_list,X_norm_list,grad)



#*********************************************************************************************************

def MLP_backprop_RMSprop(y_tr,no_hidden_layers,size_hidden_layer,weights_MLP,layers_MLP,learning_rate,gamma,beta,layers_MLP_batchnorm,layers_MLP_nlf,mu_list,var_list,X_norm_list,grad):
    x=0
    YY=0.9
    for i in range(no_hidden_layers,-1,-1):
        a=weights_MLP[i].shape[0]
        b=weights_MLP[i].shape[1]
        if x==0:
            dx=np.zeros((size_hidden_layer[i+1]))
            x=1
        else:
            dy=dx
            dx=np.zeros((size_hidden_layer[i+1]))
        
        
        if i==no_hidden_layers:
            dx=(-1)*(y_tr)*(1-(layers_MLP_nlf[i]))
        else:
            for j in range(0,b):
                dx[j]=np.dot(dy,weights_MLP[i+1][j+1,:])*relu_dev(layers_MLP_nlf[i][j])
                
        dx,gamma,beta=batchnorm_backward(dx,layers_MLP_batchnorm[i] ,X_norm_list[i], mu_list[i], var_list[i], gamma, beta)

        grad[i]=np.sqrt(0.9*(grad[i])**2 + 0.1*(dx**2))
        for j in range(0,b):
            weights_MLP[i][1:,j]=weights_MLP[i][1:,j]-(learning_rate*dx[j]*layers_MLP[i])/(grad[i][j]+1e-8)
            
            weights_MLP[i][0,i]=weights_MLP[i][0,i]-(learning_rate*dx[j])/(grad[i][j]+1e-8)
            

        
           
    return weights_MLP,gamma,beta,grad



#*********************************************************************************************************

def MLP_backprop_adam(y_tr,no_hidden_layers,size_hidden_layer,weights_MLP,layers_MLP,learning_rate,gamma,beta,layers_MLP_batchnorm,layers_MLP_nlf,mu_list,var_list,X_norm_list,grad,m):
    x=0
    YY=0.9
    prev_weights=weights_MLP
    for i in range(no_hidden_layers,-1,-1):
        a=weights_MLP[i].shape[0]
        b=weights_MLP[i].shape[1]
        if x==0:
            dx=np.zeros((size_hidden_layer[i+1]))
            x=1
        else:
            dy=dx
            dx=np.zeros((size_hidden_layer[i+1]))
        
        
        if i==no_hidden_layers:
            dx=(-1)*(y_tr)*(1-(layers_MLP_nlf[i]))
        else:
            for j in range(0,b):
                dx[j]=np.dot(dy,weights_MLP[i+1][j+1,:])*relu_dev(layers_MLP_nlf[i][j])
                
        dx,gamma,beta=batchnorm_backward(dx,layers_MLP_batchnorm[i] ,X_norm_list[i], mu_list[i], var_list[i], gamma, beta)
        
        grad[i]=(0.9*(grad[i])**2 + 0.1*(dx**2))
        m[i]= 0.7*m[i] + 0.3*dx
        grad_h=grad[i]/(0.1)
        m_h=m[i]/0.3
        for j in range(0,b):
            weights_MLP[i][1:,j]=weights_MLP[i][1:,j]-(learning_rate*m_h[j]*layers_MLP[i])/np.sqrt(grad_h[j]+1e-8)
            
            weights_MLP[i][0,i]=weights_MLP[i][0,i]-(learning_rate*m_h[j])/(grad_h[j]+1e-8)
            

        
           
    return weights_MLP,gamma,beta,grad,m


#*********************************************************************************************************4

def SGD_train(x,y1,gamma,beta,MLP_input_size,no_hidden_layers,size_hidden_layer,non_linear_fn,weights_MLP,layers_MLP,output_size):
    epochs=50
    learning_rate=0.001

    no_of_batches=50
    data_per_batch=int(x.shape[0]/no_of_batches)
    #print(data_per_batch)
    error_epoch=[]
    accuracy_epoch=[]
    epoch_list=[]
    for epoch in range(epochs):
        error_epoch_tr=0
        print('Training epoch: {}'.format(epoch + 1))
        error=0
        acc=0
        for batch in range(no_of_batches):
            print('Training batch: {}'.format(batch + 1))
            error_epoch_tr_batch=0
            acc_batch=0
            error_batch=0
            for i in range(data_per_batch):
                layers_MLP,layers_MLP_batchnorm,layers_MLP_nlf,mu_list,var_list,X_norm_list,gamma,beta=MLP(x[i+batch,:],no_hidden_layers,size_hidden_layer,non_linear_fn,output_size,weights_MLP,layers_MLP,gamma,beta)
                #print(layers_MLP_nlf[no_hidden_layers])
                d=cross_entropy_error(layers_MLP_nlf[no_hidden_layers],y1[i+batch,:])
                error_batch+=d
                error+=d
                #print(y1[i+batch,:].shape)
                #print("chck it",np.argmax(layers_MLP_nlf[no_hidden_layers]),np.argmax(y1[i+batch,:]))
                if (np.argmax(layers_MLP_nlf[no_hidden_layers])==np.argmax(y1[i+batch,:])):
                    #print("yess")
                    acc_batch+=1
                    acc+=1
                #print("weights before update",weights_MLP[2][0,:])
                weights_MLP,gamma,beta=MLP_backprop_SGD(y1[i+batch,:],no_hidden_layers,size_hidden_layer,weights_MLP,layers_MLP,learning_rate,gamma,beta,layers_MLP_batchnorm,layers_MLP_nlf,mu_list,var_list,X_norm_list)
                #print("weights before update",weights_MLP[2][0,:])
            print("error on training data per batch",error_batch/data_per_batch)
            print("acc on training data per batch",acc_batch/data_per_batch)
        error_epoch_tr=error
        error_epoch.append(error_epoch_tr/x.shape[0])
        print("error on training data per epoch",error_epoch_tr/x.shape[0])
        accuracy_epoch.append(acc/x.shape[0])
        print("accuracy of training data per epoch",acc/x.shape[0])
        epoch_list.append(epoch+1)
    f1=plt.plot(epoch_list,error_epoch)
    plt.title("error vs epochs")
    plt.show(f1)

    f2=plt.plot(epoch_list,accuracy_epoch)
    plt.title("accuracy vs epochs")
    plt.show(f2)

#*********************************************************************************************************

def momentum_train(y1,gamma,beta,MLP_input_size,no_hidden_layers,size_hidden_layer,non_linear_fn,weights_MLP,layers_MLP,output_size):

    epochs=50
    learning_rate=0.0001

    no_of_batches=100
    data_per_batch=int(x.shape[0]/no_of_batches)
    #print(data_per_batch)
    error_epoch=[]
    epoch_list=[]
    accuracy_epoch=[]
    for epoch in range(epochs):
        error_epoch_tr=0
        print('Training epoch: {}'.format(epoch + 1))
        error=0
        acc=0
        grad=[]
        for i in range(no_hidden_layers+1):
            grad.append(0)
        for batch in range(no_of_batches):
            print('Training batch: {}'.format(batch + 1))
            error_epoch_tr_batch=0
            acc_batch=0
            error_batch=0
            for i in range(data_per_batch):
                layers_MLP,layers_MLP_batchnorm,layers_MLP_nlf,mu_list,var_list,X_norm_list,gamma,beta=MLP(x[i+batch,:],no_hidden_layers,size_hidden_layer,non_linear_fn,output_size,weights_MLP,layers_MLP,gamma,beta)
                print(layers_MLP_nlf[no_hidden_layers])
                d=cross_entropy_error(layers_MLP_nlf[no_hidden_layers],y1[i+batch,:])
                error_batch+=d
                error+=d
                #print(y1[i+batch,:].shape)
                #print("chck it",np.argmax(layers_MLP_nlf[no_hidden_layers]),np.argmax(y1[i+batch,:]))
                if (np.argmax(layers_MLP_nlf[no_hidden_layers])==np.argmax(y1[i+batch,:])):
                    acc_batch+=1
                    acc+=1
                #print("weights before update",weights_MLP[2][0,:])
                weights_MLP,gamma,beta,grad=MLP_backprop_momentum(y1[i+batch,:],no_hidden_layers,size_hidden_layer,weights_MLP,layers_MLP,learning_rate,gamma,beta,layers_MLP_batchnorm,layers_MLP_nlf,mu_list,var_list,X_norm_list,grad)
                #print("weights before update",weights_MLP[2][0,:])
            print("error on training data per batch",error_batch/data_per_batch)
            print("acc on training data per batch",acc_batch/data_per_batch)
        error_epoch_tr=error
        error_epoch.append(error_epoch_tr/x.shape[0])
        print("error on training data per epoch",error_epoch_tr/x.shape[0])
        accuracy_epoch.append(acc/x.shape[0])
        print("accuracy of training data per epoch",acc/x.shape[0])
        epoch_list.append(epoch+1)
    f1=plt.plot(epoch_list,error_epoch)
    plt.title("error vs epochs")
    plt.show(f1)
    f2=plt.plot(epoch_list,accuracy_epoch)
    plt.title("accuracy vs epochs")
    plt.show(f2)

#*********************************************************************************************************

def nesterov_train(x,y1,gamma,beta,MLP_input_size,no_hidden_layers,size_hidden_layer,non_linear_fn,weights_MLP,layers_MLP,output_size):

    epochs=50
    learning_rate=0.01

    no_of_batches=100
    data_per_batch=int(x.shape[0]/no_of_batches)
    #print(data_per_batch)
    error_epoch=[]
    epoch_list=[]
    accuracy_epoch=[]
    for epoch in range(epochs):
        error_epoch_tr=0
        print('Training epoch: {}'.format(epoch + 1))
        error=0
        acc=0
        grad=[]
        for i in range(no_hidden_layers+1):
            grad.append(0)
        for batch in range(no_of_batches):
            print('Training batch: {}'.format(batch + 1))
            error_epoch_tr_batch=0
            acc_batch=0
            error_batch=0
            for i in range(data_per_batch):
                layers_MLP,layers_MLP_batchnorm,layers_MLP_nlf,mu_list,var_list,X_norm_list,gamma,beta=MLP(x[i+batch,:],no_hidden_layers,size_hidden_layer,non_linear_fn,output_size,weights_MLP,layers_MLP,gamma,beta)
                print(layers_MLP_nlf[no_hidden_layers])
                d=cross_entropy_error(layers_MLP_nlf[no_hidden_layers],y1[i+batch,:])
                error_batch+=d
                error+=d
                #print(y1[i+batch,:].shape)
                #print("chck it",np.argmax(layers_MLP_nlf[no_hidden_layers]),np.argmax(y1[i+batch,:]))
                if (np.argmax(layers_MLP_nlf[no_hidden_layers])==np.argmax(y1[i+batch,:])):
                    acc_batch+=1
                    acc+=1
                #print("weights before update",weights_MLP[2][0,:])
                weights_MLP,gamma,beta,grad=MLP_backprop_nesterov(y1[i+batch,:],no_hidden_layers,size_hidden_layer,weights_MLP,layers_MLP,learning_rate,gamma,beta,layers_MLP_batchnorm,layers_MLP_nlf,mu_list,var_list,X_norm_list,grad)
                #print("weights before update",weights_MLP[2][0,:])
            print("error on training data per batch",error_batch/data_per_batch)
            print("acc on training data per batch",acc_batch/data_per_batch)
        error_epoch_tr=error
        error_epoch.append(error_epoch_tr/x.shape[0])
        print("error on training data per epoch",error_epoch_tr/x.shape[0])
        accuracy_epoch.append(acc/x.shape[0])
        print("accuracy of training data per epoch",acc/x.shape[0])
        epoch_list.append(epoch+1)
    f1=plt.plot(epoch_list,error_epoch)
    plt.title("error vs epochs")
    plt.show(f1)

    f2=plt.plot(epoch_list,accuracy_epoch)
    plt.title("accuracy vs epochs")
    plt.show(f2)

#*********************************************************************************************************

def adagrad_train(x,y1,gamma,beta,MLP_input_size,no_hidden_layers,size_hidden_layer,non_linear_fn,weights_MLP,layers_MLP,output_size):

    epochs=50
    learning_rate=0.01

    no_of_batches=100
    data_per_batch=int(x.shape[0]/no_of_batches)
    #print(data_per_batch)
    error_epoch=[]
    epoch_list=[]
    accuracy_epoch=[]
    for epoch in range(epochs):
        error_epoch_tr=0
        print('Training epoch: {}'.format(epoch + 1))
        error=0
        grad=[]
        for i in range(no_hidden_layers+1):
            grad.append(0)
        acc=0
        for batch in range(no_of_batches):
            print('Training batch: {}'.format(batch + 1))
            error_epoch_tr_batch=0
            acc_batch=0
            error_batch=0
            for i in range(data_per_batch):
                layers_MLP,layers_MLP_batchnorm,layers_MLP_nlf,mu_list,var_list,X_norm_list,gamma,beta=MLP(x[i+batch,:],no_hidden_layers,size_hidden_layer,non_linear_fn,output_size,weights_MLP,layers_MLP,gamma,beta)
                #print(layers_MLP_nlf[no_hidden_layers])
                d=cross_entropy_error(layers_MLP_nlf[no_hidden_layers],y1[i+batch,:])
                error_batch+=d
                error+=d
                #print(y1[i+batch,:].shape)
                #print("chck it",np.argmax(layers_MLP_nlf[no_hidden_layers]),np.argmax(y1[i+batch,:]))
                if (np.argmax(layers_MLP_nlf[no_hidden_layers])==np.argmax(y1[i+batch,:])):
                    acc_batch+=1
                    acc+=1
                #print("weights before update",weights_MLP[2][0,:])
                
                weights_MLP,gamma,beta,grad=MLP_backprop_adagrad(y1[i+batch,:],no_hidden_layers,size_hidden_layer,weights_MLP,layers_MLP,learning_rate,gamma,beta,layers_MLP_batchnorm,layers_MLP_nlf,mu_list,var_list,X_norm_list,grad)
                #print("weights before update",weights_MLP[2][0,:])
            print("error on training data per batch",error_batch/data_per_batch)
            print("acc on training data per batch",acc_batch/data_per_batch)
        error_epoch_tr=error
        error_epoch.append(error_epoch_tr/x.shape[0])
        print("error on training data per epoch",error_epoch_tr/x.shape[0])
        accuracy_epoch.append(acc/x.shape[0])
        print("accuracy of training data per epoch",acc/x.shape[0])
        epoch_list.append(epoch+1)
    f1=plt.plot(epoch_list,error_epoch)
    plt.title("error vs epochs")
    plt.show(f1)

    f2=plt.plot(epoch_list,accuracy_epoch)
    plt.title("accuracy vs epochs")
    plt.show(f2)

#*********************************************************************************************************

def RMSprop_train(x,y1,gamma,beta,MLP_input_size,no_hidden_layers,size_hidden_layer,non_linear_fn,weights_MLP,layers_MLP,output_size):

    epochs=50
    learning_rate=0.01

    no_of_batches=100
    data_per_batch=int(x.shape[0]/no_of_batches)
    #print(data_per_batch)
    error_epoch=[]
    epoch_list=[]
    accuracy_epoch=[]
    for epoch in range(epochs):
        error_epoch_tr=0
        print('Training epoch: {}'.format(epoch + 1))
        error=0
        grad=[]
        for i in range(no_hidden_layers+1):
            grad.append(0)
        acc=0
        for batch in range(no_of_batches):
            print('Training batch: {}'.format(batch + 1))
            error_epoch_tr_batch=0
            acc_batch=0
            error_batch=0
            for i in range(data_per_batch):
                layers_MLP,layers_MLP_batchnorm,layers_MLP_nlf,mu_list,var_list,X_norm_list,gamma,beta=MLP(x[i+batch,:],no_hidden_layers,size_hidden_layer,non_linear_fn,output_size,weights_MLP,layers_MLP,gamma,beta)
                #print(layers_MLP_nlf[no_hidden_layers])
                d=cross_entropy_error(layers_MLP_nlf[no_hidden_layers],y1[i+batch,:])
                error_batch+=d
                error+=d
                #print(y1[i+batch,:].shape)
                #print("chck it",np.argmax(layers_MLP_nlf[no_hidden_layers]),np.argmax(y1[i+batch,:]))
                if (np.argmax(layers_MLP_nlf[no_hidden_layers])==np.argmax(y1[i+batch,:])):
                    acc_batch+=1
                    acc+=1
                #print("weights before update",weights_MLP[2][0,:])
                
                weights_MLP,gamma,beta,grad=MLP_backprop_RMSprop(y1[i+batch,:],no_hidden_layers,size_hidden_layer,weights_MLP,layers_MLP,learning_rate,gamma,beta,layers_MLP_batchnorm,layers_MLP_nlf,mu_list,var_list,X_norm_list,grad)
                #print("weights before update",weights_MLP[2][0,:])
            print("error on training data per batch",error_batch/data_per_batch)
            print("acc on training data per batch",acc_batch/data_per_batch)
        error_epoch_tr=error
        error_epoch.append(error_epoch_tr/x.shape[0])
        print("error on training data per epoch",error_epoch_tr/x.shape[0])
        accuracy_epoch.append(acc/x.shape[0])
        print("accuracy of training data per epoch",acc/x.shape[0])
        epoch_list.append(epoch+1)
    f1=plt.plot(epoch_list,error_epoch)
    plt.title("error vs epochs")
    plt.show(f1)

    f2=plt.plot(epoch_list,accuracy_epoch)
    plt.title("accuracy vs epochs")
    plt.show(f2)

#*********************************************************************************************************

def adam_train(x,y1,gamma,beta,MLP_input_size,no_hidden_layers,size_hidden_layer,non_linear_fn,weights_MLP,layers_MLP,output_size):

    epochs=50
    learning_rate=0.01

    no_of_batches=100
    data_per_batch=int(x.shape[0]/no_of_batches)
    #print(data_per_batch)
    error_epoch=[]
    accuracy_epoch=[]
    epoch_list=[]
    for epoch in range(epochs):
        error_epoch_tr=0
        print('Training epoch: {}'.format(epoch + 1))
        error=0
        grad=[]
        m=[]
        for i in range(no_hidden_layers+1):
            grad.append(0)
        for i in range(no_hidden_layers+1):
            m.append(0)

        for i in range(no_hidden_layers+1):
            grad.append(0)
        acc=0
        for batch in range(no_of_batches):
            print('Training batch: {}'.format(batch + 1))
            error_epoch_tr_batch=0
            acc_batch=0
            error_batch=0
            for i in range(data_per_batch):
                layers_MLP,layers_MLP_batchnorm,layers_MLP_nlf,mu_list,var_list,X_norm_list,gamma,beta=MLP(x[i+batch,:],no_hidden_layers,size_hidden_layer,non_linear_fn,output_size,weights_MLP,layers_MLP,gamma,beta)
                #print(layers_MLP_nlf[no_hidden_layers])
                d=cross_entropy_error(layers_MLP_nlf[no_hidden_layers],y1[i+batch,:])
                error_batch+=d
                error+=d
                #print(y1[i+batch,:].shape)
                #print("chck it",np.argmax(layers_MLP_nlf[no_hidden_layers]),np.argmax(y1[i+batch,:]))
                if (np.argmax(layers_MLP_nlf[no_hidden_layers])==np.argmax(y1[i+batch,:])):
                    acc_batch+=1
                    acc+=1
                #print("weights before update",weights_MLP[2][0,:])
                
                weights_MLP,gamma,beta,grad,m=MLP_backprop_adam(y1[i+batch,:],no_hidden_layers,size_hidden_layer,weights_MLP,layers_MLP,learning_rate,gamma,beta,layers_MLP_batchnorm,layers_MLP_nlf,mu_list,var_list,X_norm_list,grad,m)
                #print("weights before update",weights_MLP[2][0,:])
            print("error on training data per batch",error_batch/data_per_batch)
            print("acc on training data per batch",acc_batch/data_per_batch)
        error_epoch_tr=error
        error_epoch.append(error_epoch_tr/x.shape[0])
        print("error on training data per epoch",error_epoch_tr/x.shape[0])
        accuracy_epoch.append(acc/x.shape[0])
        print("accuracy of training data per epoch",acc/x.shape[0])
        epoch_list.append(epoch+1)
    f1=plt.plot(epoch_list,error_epoch)
    plt.title("error vs epochs")
    plt.show(f1)

    f2=plt.plot(epoch_list,accuracy_epoch)
    plt.title("accuracy vs epochs")
    plt.show(f2)


#*********************************************************************************************************

main()

  





