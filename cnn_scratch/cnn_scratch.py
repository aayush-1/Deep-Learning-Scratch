import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import math

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
	
#*****************************************************************************************

def convolution_fn(input_image,filter_kernel,stride,padding,non_linearity):
    #print("input image\n",input_image.shape)
    #print("filter kernel\n", filter_kernel.shape)
    h_inp=input_image.shape[0]
    w_inp=input_image.shape[1]
    d_inp=input_image.shape[2]
    
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
            
    if non_linearity=="tanh":
        output_image=np.tanh(output_image)
    print("\ninput image - convolution_fn: ",input_image.shape)
    print("\nkernel- convolution_fn: ",filter_kernel.shape)
    print("\noutput image- convolution_fn: ",output_image.shape) 
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
    print("\ninput image -conv_layer_fn: ",input_image.shape)
    print("\nconvolutional layer-conv_layer_fn: ",conv1.shape)
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
                        curr_region=input_image[h:h+pool_kernel_size[0],w:w+pool_kernel_size[1],:]
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
    print("\ninput image -pooling fn: ",input_image.shape)    
    print("\noutput volume -pooling fn: ",output_image.shape)
    return output_image

#*****************************************************************************************
#composition_conv_layer_fn(input_image,conv_layers,filters,stride,padding,non_linearity,pool_type,pool_stride,pool_kernel_size)
def composition_conv_layer_fn(input_image,conv_layers,filters,stride,padding,non_linearity,pool_type,pool_stride,pool_kernel_size):
    for i in range(0,conv_layers):
        if i==0:
            conv1=conv_layer_fn(input_image,filters[i],stride[i],padding[i],non_linearity[i])
            pool1=pooling_fn(conv1,pool_stride[i],pool_type[i],pool_kernel_size)
        elif i!=0:
            conv1=conv_layer_fn(pool1,filters[i],stride[i],padding[i],non_linearity[i])
            pool1=pooling_fn(conv1,pool_stride[i],pool_type[i],pool_kernel_size)
    print("\noutput_volume -composition_conv_layer_fn: ",pool1.shape)        
    return pool1
            

#*****************************************************************************************

def unravelling_fn(cnn_output,MLP_input_size,weights):
    unravel_input=cnn_output.flatten()
    mlp_input=np.zeros((MLP_input_size))
    for i in range(0,MLP_input_size):
        mlp_input[i]=np.matmul(unravel_input,np.transpose(weights[1:,i]))+weights[0,i]
    return mlp_input

#*****************************************************************************************

def MLP(mlp_input,no_hidden_layers,size_hidden_layer,non_linear_fn,output_size,weights_MLP,non_linearity):
    o=np.zeros((size_hidden_layer[0]))
    for i in range(0,size_hidden_layer[0]):
        o[i]=np.matmul(mlp_input,np.transpose(weights_MLP[0][1:,i]))+weights_MLP[0][0,i]
    if non_linearity[0]=="sigmoid":
        for h in range(0,size_hidden_layer[0]):
            o[h]=sigmoid(o[h])
            
    if non_linearity[0]=="tanh":
        o=np.tanh(o)
        
    for i in range(1,no_hidden_layers):
        p=np.zeros((size_hidden_layer[i]))
        if i==1:
            for j in range(0,size_hidden_layer[i]):
                p[j]=np.matmul(o,np.transpose(weights_MLP[i][1:,j]))+weights_MLP[i][0,j]
                
        else:
            for j in range(0,size_hidden_layer[i]):
                p[j]=np.matmul(o,np.transpose(weights_MLP[i][1:,j]))+weights_MLP[i][0,j]
        o=p
        if non_linearity[i]=="sigmoid":
            for h in range(0,size_hidden_layer[i]):
                o[h]=sigmoid(o[h])
            
        if non_linearity[i]=="tanh":
            o=np.tanh(o)
        
        
        
        
    output_vector=np.zeros((output_size))  
    for i in range(0,output_size):
        output_vector[i]=np.matmul(o,np.transpose(weights_MLP[no_hidden_layers][1:,i]))+weights_MLP[no_hidden_layers][0,i]
       
    output_vector=softmax(output_vector)

    return output_vector

#*****************************************************************************************

def main_func():
	path=input('input image path :')
	input_image=misc.imread(path)

	input_image_shape=input_image.shape

	#************
	conv_layers =int(input('\nenter no. of convolutional layers: '))
	non_linearity=[]
	stride=[]
	padding=[]
	for i in range(0,conv_layers):
	    print("layer ",i)
	    non_linearity.append(input('\nenter non linearity functions(tanh or sigmoid) for corresponding layer: '))
	    stride.append(int(input('\nenter stride for corresponding layer: ')))
	    padding.append(input('\nenter padding (same or valid) for corresponding layer: '))
	
	#*************
	kernel_size =np.array([int(input('\n filter kernel height: ')),int(input('\nfilter kernel width: '))])

	krnl_size =np.array([kernel_size[0],kernel_size[1],3])
	no_of_filters=[]
	for i in range(0,conv_layers):
	    print('layer',i)
	    no_of_filters.append(int(input('\nenter no. of filters for convolution: ')))
	    no_of_filters[i]=np.hstack([krnl_size,no_of_filters[i]])
	    krnl_size=np.array([kernel_size[0],kernel_size[1],no_of_filters[i][3]])
	    
	#print(no_of_filters)

	#**************
	#pool type , pool kernel shape , pool stride
	pool_type=[]
	pool_kernel_size=np.array([int(input('\n pooling kernel height: ')),int(input('\npooling kernel width: '))])
	pool_stride=[]
	for i in range(0,conv_layers):
	    print("layer ",i)
	    pool_type.append(input('\nenter pool type(max or avg) for each conv layer: '))
	    pool_stride.append(int(input('\nenter pool stride for each conv layer: ' )))


	#**************

	    
	filters=[]
	for i in range(0,conv_layers):
	    filters.append(np.random.normal(0,1,(no_of_filters[i])))
	    
	#print(filters[0].shape)
	plt.imshow(input_image)

#*******************
	cnn_output=composition_conv_layer_fn(input_image,conv_layers,filters,stride,padding,non_linearity,pool_type,pool_stride,pool_kernel_size)
	print(cnn_output.shape)
#*******************

	#MLP----no.of hidden layers , size of each hidden layers , non linear fn , size of output layer
	MLP_input_size=int(input('\nenter input size for MLP: '))

	cnn_output_size=cnn_output.shape[0]*cnn_output.shape[1]*cnn_output.shape[2]
	weights=np.random.normal(0,1,(cnn_output_size+1,MLP_input_size))

	no_hidden_layers=int(input('\nenter hidden layers for MLP: '))
	size_hidden_layer=[]
	non_linear_fn=[]
	for i in range(0,no_hidden_layers):
	    size_hidden_layer.append(int(input('\nsize of corresponding hidden layer: ')))
	    non_linear_fn.append(input('\nnon linearity fn(tanh or sigmoid) for corresponding layer: '))
	    
	output_size=int(input('\nenter output_size for MLP: '))

	weights_MLP=[]
	weights_MLP.append(np.random.normal(0,1,(MLP_input_size+1, size_hidden_layer[0])))
	for i in range(0,no_hidden_layers-1):
	    weights_MLP.append(np.random.normal(0,1,(size_hidden_layer[i]+1,size_hidden_layer[i+1])))
	weights_MLP.append(np.random.normal(0,1,(size_hidden_layer[no_hidden_layers-1]+1,output_size)))
	#print(weights.shape)

#*******************
	mlp_input=unravelling_fn(cnn_output,MLP_input_size,weights)
#*******************
	output_vec=MLP(mlp_input,no_hidden_layers,size_hidden_layer,non_linear_fn,output_size,weights_MLP,non_linear_fn)
	print("\nfinal output vector -MLP_fn",output_vec.shape)
#*****************************************************************************************


main_func()