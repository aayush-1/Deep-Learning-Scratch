from scipy.ndimage.interpolation import rotate
from scipy import misc
import math
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

#***************************************************************************************
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
        return 0

#*****************************************************************************************

def convolution_fn(input_image,filter_kernel,stride,padding,non_linearity):
    #print("input image\n",input_image.shape)
    #print("filter kernel\n", filter_kernel.shape)
    h_inp=input_image.shape[0]
    w_inp=input_image.shape[1]
    d_inp=1
    
    if padding=="same":
        h_out = int(np.ceil(float(h_inp) / float(stride)))
        w_out  = int(np.ceil(float(w_inp) / float(stride)))
        d_out=1
        ph = int(max((h_out - 1) * stride +filter_kernel.shape[0] - h_inp, 0))
        pw = int(max((w_out - 1) * stride +filter_kernel.shape[1] - w_inp, 0))
        pad_top = int(np.floor(ph / 2))
        pad_bottom =int(ph-pad_top)
        pad_left = int(np.floor(pw / 2))
        pad_right = int(pw - pad_left)
        #print(w_inp,pad_top,input_image.shape[2])
        
        h_add_top=np.zeros((pad_top,w_inp,input_image.shape[2]))
        h_add_bottom=np.zeros((pad_bottom,w_inp,input_image.shape[2]))
        w_add_left=np.zeros((h_inp+ph,pad_left,input_image.shape[2]))
        w_add_right=np.zeros((h_inp+ph,pad_right,input_image.shape[2]))
        
        #print(h_add_top.shape)
        #print(input_image.shape)
        input_image=np.vstack((h_add_top,input_image))
        input_image=np.vstack((input_image,h_add_bottom))
        input_image=np.hstack((w_add_left,input_image))
        input_image=np.hstack((input_image,w_add_right))
        
        h_inp=input_image.shape[0]
        w_inp=input_image.shape[1]
        d_inp=1
        
        
        

    else:
        h_out=int(((h_inp-filter_kernel.shape[0])/stride)+1)
        w_out=int(((w_inp-filter_kernel.shape[1])/stride)+1)
        d_out=1

        
    i=0
    j=0
    output_image=np.zeros([h_out,w_out,d_out])
    #print(w_inp,w_out)
    for h in range(0,h_inp,stride):
        if h+filter_kernel.shape[0]<h_inp:
            for w in range(0,w_inp,stride):
                if w+filter_kernel.shape[1]<w_inp:
                    curr_region=input_image[h:h+filter_kernel.shape[0],w:w+filter_kernel.shape[1],:]
                    conv = curr_region*filter_kernel
                    output_image[i,j,0]=np.sum(conv)
                    j=j+1
            i=i+1
            j=0
    
    if non_linearity=="sigmoid":
        for h in range(0,h_out):
            for w in range(0,w_out):
                output_image[h,w,0]=sigmoid(output_image[h,w,0])
                
    if non_linearity=="relu":
        for h in range(0,h_out):
            for w in range(0,w_out):
                output_image[h,w,0]=relu(output_image[h,w,0])
        
            
    if non_linearity=="tanh":
        output_image=np.tanh(output_image)
    #print("\ninput image - convolution_fn: ",input_image.shape)
    #print("\nkernel- convolution_fn: ",filter_kernel.shape)
    #print("\noutput image- convolution_fn: ",output_image.shape) 
    return output_image


#*****************************************************************************************

def pool(input_image,pool_stride,pool_type):
    
    h_inp=input_image.shape[0]
    w_inp=input_image.shape[1]
    d_inp=1
    h_out=int(((h_inp-pool_kernel_size[0])/pool_stride)+1)
    w_out=int(((w_inp-pool_kernel_size[1])/pool_stride)+1)
    d_out=1
    if pool_type=="max":
        i=0
        j=0
        k=0
        output_image=np.zeros([h_out,w_out,d_out])
        for d in range(0,d_inp):
            for h in range(0,h_inp,pool_stride):
                if h+pool_kernel_size[0]<=h_inp:
                    for w in range(0,w_inp,pool_stride):
                        if w+pool_kernel_size[1]<=w_inp:
                            curr_region=input_image[h:h+pool_kernel_size[0],w:w+pool_kernel_size[1],:]
                            output_image[i,j,k]=np.max(curr_region)
                            j=j+1
                    i=i+1
                    j=0
            i=0
            j=0
            k=k+1
    if pool_type== "avg":
        i=0
        j=0
        k=0
        output_image=np.zeros([h_out,w_out,d_out])
        for k in range(0,d_inp):
            for h in range(0,h_inp,pool_stride):
                if h+pool_kernel_size[0]<=h_inp:
                    for w in range(0,w_inp,pool_stride):
                        if w+pool_kernel_size[1]<=w_inp:
                            curr_region=input_image[h:h+pool_kernel_size[0],w:w+pool_kernel_size[1],:]
                            output_image[i,j,k]=np.average(curr_region)
                            j=j+1
                    i=i+1
                    j=0
            i=0
            j=0
            k=k+1
        
        
    return output_image

#*****************************************************************************************

def conv_layer_fn(input_image,filter_set,stride,padding,non_linearity):
    no_of_channels=filter_set.shape[3]
    for i in range(0,no_of_channels):
        if i==0:
            conv1=convolution_fn(input_image,filter_set[:,:,:,i],stride,padding,non_linearity)
        else:
            conv1=np.dstack((conv1,convolution_fn(input_image,filter_set[:,:,:,i],stride,padding,non_linearity)))
    #print("\ninput image -conv_layer_fn: ",input_image.shape)
    #print("\nconvolutional layer-conv_layer_fn: ",conv1.shape)
    return conv1

#*****************************************************************************************

def pooling_fn(input_image,pool_stride,pool_type,pool_kernel_size):
    
    h_inp=input_image.shape[0]
    w_inp=input_image.shape[1]
    d_inp=input_image.shape[2]
    h_out=int(((h_inp-pool_kernel_size[0])/pool_stride)+1)
    w_out=int(((w_inp-pool_kernel_size[1])/pool_stride)+1)
    d_out=d_inp
    output_image=np.zeros([h_out,w_out,d_out])
    i=0
    j=0
    k=0
        
    for d in range(0,d_inp):
        for h in range(0,h_inp,pool_stride):
            if h+pool_kernel_size[0]<=h_inp:
                for w in range(0,w_inp,pool_stride):
                    if w+pool_kernel_size[1]<=w_inp:
                        curr_region=input_image[h:h+pool_kernel_size[0],w:w+pool_kernel_size[1],d]
                        if pool_type== "max":
                            output_image[i,j,k]=np.max(curr_region)
                        elif pool_type=="avg":
                            output_image[i,j,k]=np.average(curr_region)

                        j=j+1
                i=i+1
                j=0
        i=0
        j=0
        k=k+1
    #print("\ninput image -pooling fn: ",input_image.shape)    
    #print("\noutput volume -pooling fn: ",output_image.shape)
    return output_image

#*****************************************************************************************
#composition_conv_layer_fn(input_image,conv_layers,filters,stride,padding,non_linearity,pool_type,pool_stride,pool_kernel_size)
def composition_conv_layer_fn(input_image,conv_layers,filters,stride,padding,non_linearity,pool_type,pool_stride,pool_kernel_size):
    for i in range(0,conv_layers):
        if i==0:
            conv1=conv_layer_fn(input_image,filters[i],stride[i],padding[i],non_linearity[i])
            pool1=pooling_fn(conv1,pool_stride[i],pool_type[i],pool_kernel_size)
        elif i!=0:
            conv2=conv_layer_fn(pool1,filters[i],stride[i],padding[i],non_linearity[i])
            pool2=pooling_fn(conv2,pool_stride[i],pool_type[i],pool_kernel_size)
    #print("\noutput_volume -composition_conv_layer_fn: ",pool1.shape)        
    return conv1,pool1,conv2,pool2
            

#*****************************************************************************************

def unravelling_fn(cnn_output,MLP_input_size,weights):
    unravel_input=cnn_output.flatten()
    mlp_input=np.zeros((MLP_input_size))
    for i in range(0,MLP_input_size):
        mlp_input[i]=np.matmul(unravel_input,np.transpose(weights[1:,i]))+weights[0,i]
    return mlp_input

#*****************************************************************************************

def MLP(mlp_input,no_hidden_layers,size_hidden_layer,non_linear_fn,output_size,weights_MLP,non_linearity):

        
    output_vector=np.zeros((output_size))  
    for i in range(0,output_size):
        output_vector[i]=np.matmul(mlp_input,np.transpose(weights_MLP[0][1:,i]))+weights_MLP[0][0,i]
       
    output_vector=softmax(output_vector)
    #print(output_vector)

    return output_vector


#*****************************************************************************************

def MLP_backprop(y_train,output_vec,weights_MLP,learning_rate,mlp_input):
    a=weights_MLP[0].shape[0]
    b=weights_MLP[0].shape[1]
    for i in range(0,b):
        weights_MLP[0][1:,i]=weights_MLP[0][1:,i]-learning_rate*(y_train[i])*mlp_input*(1-output_vec[i])
        weights_MLP[0][0,i]=weights_MLP[0][0,i]-learning_rate*(y_train[i])*(1-output_vec[i])
    return weights_MLP   
    
#*****************************************************************************************


    

def unravel_backprop(y_train,output_vec,weights_MLP,learning_rate,mlp_input,cnn_output,weights):
    unravel_input=cnn_output.flatten()
    a=weights.shape[1]
    b=weights_MLP[0].shape[1]
    for i in range(0,a):
        mlp_d=np.dot(y_train*(1-output_vec),weights_MLP[0][i+1,:])
        weights[1:,i]=weights[1:,i]-learning_rate*mlp_d*relu_dev(mlp_input[i])*unravel_input
        weights[0,i]=weights[0,i]-learning_rate*mlp_d*relu_dev(mlp_input[i])
    return weights            
    
#*****************************************************************************************]
def conv_backprop(inp_image,y_train,output_vec,weights_MLP,learning_rate,mlp_input,cnn_output,weights,conv1,pool1,conv2,pool_stride,pool_kernel_size,filters,stride,kernel_size):     
    a=np.prod(cnn_output.shape)
    b=weights.shape[1]
    u_d=np.zeros(a)
    mlp_d=np.zeros(b)
    cn_out=cnn_output.flatten()
    #print(cn_out.shape)
    for j in range(a):
        for i in range(b):
            mlp_d[i]=np.dot(y_train*(1-output_vec),weights_MLP[0][i+1,:])
            mlp_input[i]=relu_dev(mlp_input[i])
        u_d[j]=np.dot(weights[j+1,:]*mlp_d,mlp_input)
        u_d[j]=u_d[j]*relu_dev(cn_out[j])
    u_d=u_d.reshape((cnn_output.shape))
    dh=np.zeros((conv2.shape))
    #put zero where non-zero value in 2*2 kernel with stride 2
    h_inp=conv2.shape[0]
    w_inp=conv2.shape[1]
    d_inp=conv2.shape[2]
    i=0
    j=0
    k=0
        
    for d in range(0,d_inp):
        for h in range(0,h_inp,pool_stride[1]):
            if h+pool_kernel_size[0]<=h_inp:
                for w in range(0,w_inp,pool_stride[1]):
                    if w+pool_kernel_size[1]<=w_inp:
                        curr_region=conv2[h:h+pool_kernel_size[0],w:w+pool_kernel_size[1],d]
                        dh[h+(np.unravel_index(np.argmax(curr_region), curr_region.shape))[0],w+(np.unravel_index(np.argmax(curr_region), curr_region.shape))[1]]=u_d[i,j,k]
                        j=j+1
                i=i+1
                j=0
        i=0
        j=0
        k=k+1
    

    
    for i in range(pool1.shape[2]):
        input_image=pool1
        h_inp=input_image.shape[0]
        w_inp=input_image.shape[1]
        d_inp=input_image.shape[2]
        h_out = int(np.ceil(float(h_inp) / float(stride[1])))
        w_out  = int(np.ceil(float(w_inp) / float(stride[1])))
        d_out=d_inp
        ph = int(max((h_out - 1) * stride[1] +filters[1][:,:,:,i].shape[0] - h_inp, 0))
        pw = int(max((w_out - 1) * stride[1] +filters[1][:,:,:,i].shape[1] - w_inp, 0))
        pad_top = int(np.floor(ph / 2))
        pad_bottom =int(ph-pad_top)
        pad_left = int(np.floor(pw / 2))
        pad_right = int(pw - pad_left)
        #print(w_inp,pad_top,input_image.shape[2])

        h_add_top=np.zeros((pad_top,w_inp,input_image.shape[2]))
        h_add_bottom=np.zeros((pad_bottom,w_inp,input_image.shape[2]))
        w_add_left=np.zeros((h_inp+ph,pad_left,input_image.shape[2]))
        w_add_right=np.zeros((h_inp+ph,pad_right,input_image.shape[2]))

        #print(h_add_top.shape)
        #print(input_image.shape)
        input_image=np.vstack((h_add_top,input_image))
        input_image=np.vstack((input_image,h_add_bottom))
        input_image=np.hstack((w_add_left,input_image))
        input_image=np.hstack((input_image,w_add_right))

        h_inp=input_image.shape[0]
        w_inp=input_image.shape[1]
        d_inp=input_image.shape[2]
        h=0
        j=0
        k=0
        for d in range(0,d_inp):
            for h in range(0,h_inp,stride[1]):
                if h+filters[1][:,:,:,i].shape[0]<h_inp:
                    for w in range(0,w_inp,stride[1]):
                        if w+filters[1][:,:,:,i].shape[1]<w_inp:
                            curr_region=input_image[h:h+filters[1][:,:,:,i].shape[0],w:w+filters[1][:,:,:,i].shape[1],:]
                            filters[1][:,:,:,i]=filters[1][:,:,:,i]-learning_rate*dh[h,j,k]*curr_region

                            j=j+1
                    h=h+1
                    j=0
            h=0
            j=0
            k=k+1

    filter180=rotate(filters[1], angle=180, reshape=True,axes=[1,0])
    filter180=np.transpose(filter180, [0, 1, 3, 2])
    dc=conv_layer_fn(dh,filter180,stride[1],padding="same",non_linearity="none")
    
    dc1=np.zeros(conv1.shape)
    
    h_inp=conv1.shape[0]
    w_inp=conv1.shape[1]
    d_inp=conv1.shape[2]
    i=0
    j=0
    k=0
        
    for d in range(0,d_inp):
        for h in range(0,h_inp,pool_stride[0]):
            if h+pool_kernel_size[0]<=h_inp:
                for w in range(0,w_inp,pool_stride[0]):
                    if w+pool_kernel_size[1]<=w_inp:
                        curr_region=conv1[h:h+pool_kernel_size[0],w:w+pool_kernel_size[1],d]
                        dc1[h+(np.unravel_index(np.argmax(curr_region), curr_region.shape))[0],w+(np.unravel_index(np.argmax(curr_region), curr_region.shape))[1]]=dc[i,j,k]
                        j=j+1
                i=i+1
                j=0
        i=0
        j=0
        k=k+1
        
    #print(dc1.shape)
        
    for i in range(inp_image.shape[2]):
        h_inp=inp_image.shape[0]
        w_inp=inp_image.shape[1]
        d_inp=inp_image.shape[2]
        h_out = int(np.ceil(float(h_inp) / float(stride[0])))
        w_out  = int(np.ceil(float(w_inp) / float(stride[0])))
        d_out=d_inp
        ph = int(max((h_out - 1) * stride[0] +filters[0][:,:,:,i].shape[0] - h_inp, 0))
        pw = int(max((w_out - 1) * stride[0] +filters[0][:,:,:,i].shape[1] - w_inp, 0))
        pad_top = int(np.floor(ph / 2))
        pad_bottom =int(ph-pad_top)
        pad_left = int(np.floor(pw / 2))
        pad_right = int(pw - pad_left)
        #print(w_inp,pad_top,input_image.shape[2])

        h_add_top=np.zeros((pad_top,w_inp,inp_image.shape[2]))
        h_add_bottom=np.zeros((pad_bottom,w_inp,inp_image.shape[2]))
        w_add_left=np.zeros((h_inp+ph,pad_left,inp_image.shape[2]))
        w_add_right=np.zeros((h_inp+ph,pad_right,inp_image.shape[2]))

        #print(h_add_top.shape)
        #print(input_image.shape)
        inp_image=np.vstack((h_add_top,inp_image))
        inp_image=np.vstack((inp_image,h_add_bottom))
        inp_image=np.hstack((w_add_left,inp_image))
        inp_image=np.hstack((inp_image,w_add_right))

        h_inp=inp_image.shape[0]
        w_inp=inp_image.shape[1]
        d_inp=inp_image.shape[2]
        h=0
        j=0
        k=0
        for d in range(0,d_inp):    
            for h in range(0,h_inp,stride[0]):
                if h+filters[0][:,:,:,i].shape[0]<h_inp:
                    for w in range(0,w_inp,stride[0]):
                        if w+filters[0][:,:,:,i].shape[1]<w_inp:
                            curr_region=inp_image[h:h+filters[0][:,:,:,i].shape[0],w:w+filters[0][:,:,:,i].shape[1],:]
                            filters[0][:,:,:,i]=filters[0][:,:,:,i]-learning_rate*dc1[h,j,k]*curr_region

                            j=j+1
                    h=h+1
                    j=0
            h=0
            j=0
            k=k+1
            
            
    return filters
    
#*****************************************************************************************
            
def cross_entropy_error(output_vec,y_train):
    e=0
    for i in range(output_vec.shape[0]):
        if output_vec[i]!=0:
            e=e + (-1)*y_train[i]*np.log10(output_vec[i])
    return e
#*****************************************************************************************


def main_func():
    mnist = input_data.read_data_sets("MNIST data/", one_hot=True)
    X_train = np.dstack([img.reshape((28, 28)) for img in mnist.train.images])
    y_train = mnist.train.labels
    X_test = np.dstack([img.reshape((28, 28)) for img in mnist.test.images])
    #print(X_test.shape[2])
    y_test = mnist.test.labels
    
    input_image=X_train[:,:,0]
    input_image=input_image.reshape([28,28,-1])
    input_image_shape=input_image.shape

    #************
    conv_layers =2
    non_linearity=['relu','relu']
    stride=[1,1]
    padding=['same','same']
#*************
    kernel_size =np.array([5,5])

    krnl_size =np.array([kernel_size[0],kernel_size[1],1])
    no_of_filters=[32,64]
    for i in range(0,conv_layers):
        no_of_filters[i]=np.hstack([krnl_size,no_of_filters[i]])
        krnl_size=np.array([kernel_size[0],kernel_size[1],no_of_filters[i][3]])
        
    #print(no_of_filters)

    #**************
    #pool type , pool kernel shape , pool stride
    pool_type=['max','max']
    pool_kernel_size=np.array([2,2])
    pool_stride=[2,2]
#*************
        
    filters=[]
    for i in range(0,conv_layers):
        filters.append(np.random.normal(0,0.01,(no_of_filters[i])))
        
    #print(filters[0].shape)
    #plt.imshow(input_image)

#*******************
    conv1,pool1,conv2,cnn_output=composition_conv_layer_fn(input_image,conv_layers,filters,stride,padding,non_linearity,pool_type,pool_stride,pool_kernel_size)
    #print(cnn_output.shape)
#*******************
    #MLP----no.of hidden layers , size of each hidden layers , non linear fn , size of output layer
    MLP_input_size=1024

    cnn_output_size=np.prod(cnn_output.shape)
    weights=np.random.normal(0,0.021,(cnn_output_size+1,MLP_input_size))

    no_hidden_layers=0
    size_hidden_layer=[]
    non_linear_fn=['tanh']

    output_size=10

    weights_MLP=[]

    weights_MLP.append(np.random.normal(0,0.004,(MLP_input_size+1,output_size)))
    #print(weights.shape)

    
    
    
    epochs=15
    learning_rate=0.001
    no_of_batches=100
    data_per_batch=int(X_train.shape[2]/no_of_batches)
    #print(data_per_batch)
    error_training=[]
    error_test=[]
    accuracy=[]
    error=0
    for epoch in range(epochs):
        error_epoch_tr=0
        error_epoch_ts=0
        print('Training epoch: {}'.format(epoch + 1))
        error=0
        error2=0
        for batch in range(no_of_batches):
            print('Training batch: {}'.format(batch + 1))
            error_epoch_tr_batch=0
            for i in range(data_per_batch):
                input_image=X_train[:,:,i+batch]
                input_image=input_image.reshape([28,28,-1])
                conv1,pool1,conv2,cnn_output=composition_conv_layer_fn(input_image,conv_layers,filters,stride,padding,non_linearity,pool_type,pool_stride,pool_kernel_size)
                mlp_input=unravelling_fn(cnn_output,MLP_input_size,weights)
                output_vec=MLP(mlp_input,no_hidden_layers,size_hidden_layer,non_linear_fn,output_size,weights_MLP,non_linear_fn)
                error+=cross_entropy_error(output_vec,y_train[i+batch,:])
                weights_MLP=MLP_backprop(y_train[i+batch,:],output_vec,weights_MLP,learning_rate,mlp_input)
                weights=unravel_backprop(y_train[i+batch,:],output_vec,weights_MLP,learning_rate,mlp_input,cnn_output,weights)
                filters=conv_backprop(input_image,y_train[i+batch,:],output_vec,weights_MLP,learning_rate,mlp_input,cnn_output,weights,conv1,pool1,conv2,pool_stride,pool_kernel_size,filters,stride,kernel_size)
            error_epoch_tr_batch=error
            print("error on training data per batch",error_epoch_tr_batch/data_per_batch)
        error_epoch_tr=error
        error_training.append(error_epoch_tr/X_train.shape[2])
        print("error on training data per epoch",error_epoch_tr/X_train.shape[2])
        acc=0
        for j in range(X_test.shape[2]):
            input_image_t=X_test[:,:,j]
            input_image_t=input_image_t.reshape([28,28,-1])
            conv1,pool1,conv2,cnn_output_t=composition_conv_layer_fn(input_image_t,conv_layers,filters,stride,padding,non_linearity,pool_type,pool_stride,pool_kernel_size)
            mlp_input_t=unravelling_fn(cnn_output_t,MLP_input_size,weights)
            output_vec_t=MLP(mlp_input_t,no_hidden_layers,size_hidden_layer,non_linear_fn,output_size,weights_MLP,non_linear_fn)
            error2=cross_entropy_error(output_vec,y_test[j,:])
            if np.array_equal(output_vec,y_test[j,:]):
                acc+=1
        error_epoch_ts+=error2
        error_test.append(error_epoch_ts/X_test.shape[2])
        print("error on testing data per epoch",error_epoch_ts/X_test.shape[2])
        
        print("accuracy per epoch: ",float(acc)/float(X_test.shape[2]))
        accuracy.append(float(acc)/float(X_test.shape[2]))
        a=np.random.permutation(X_train.shape[2])
        b=np.random.permutation(X_test.shape[2])
        X_train=X_train[:,:,a]
        y_train=y_train[a,:]
        X_test=X_test[:,:,b]
        y_test=y_test[b,:]


            
    
    
main_func()
