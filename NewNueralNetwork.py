import numpy as np
import random
import gzip
import time
import json
import tensorflow as tf
from PIL import Image


with open('data.json') as f:
    data=json.load(f)



def NueralNetwork():
    global data
    global x_train
    global y_train
    global x_test
    global y_test
    Network=[None]*4
    def initialize():
        sizes=[(784,128),(128,64),(64,10)]
        for i in range(4):

            if i==0:
                layer={'input':None}
            else:
                col,row=sizes[i-1]
                
                weights=np.matrix(data[str(i-1)][0])

                bias=np.array(data[str(i-1)][1])

                layer={'weights':weights,'bias':bias}
            Network[i]=layer
    
    initialize()

    def sigmoid(x):
        return (1/(1+np.exp(-1*x)))
    
    def activationderivativefunc(pre_activation_array):
        pre_activation_list=pre_activation_array.tolist()

        for x in pre_activation_list:
            x[0]=sigmoid(x[0])*(1-sigmoid(x[0]))
        
        return np.array(pre_activation_list)
    
    def activationfunc(pre_activation_array):
        pre_activation_list=pre_activation_array.tolist()
        
        for x in pre_activation_list:
            x[0]=sigmoid(x[0])
        
        return np.array(pre_activation_list)

    def run_image(image_arr):

        image_arr=image_arr.reshape(784,1)
        Network[0]['output']=np.array(image_arr)

        curr_input=Network[0]['output']
        
        for i in range(1,4):
           
            curr_layer=Network[i]

            curr_weight,curr_bias=curr_layer['weights'],curr_layer['bias']
            pre_activation_array=np.add(curr_weight*curr_input,curr_bias)

            curr_layer['preactivation']=pre_activation_array
            curr_layer['output']=activationfunc(pre_activation_array)

            curr_input=curr_layer['output']
        
        last_layer_output=Network[-1]['output'].tolist()

        return last_layer_output.index(max(last_layer_output))
    def findgradient(label):

        truth=[[0] for x in range(10)]
        truth[label][0]=1
        truth=np.array(truth)

        curr_error=np.multiply((Network[-1]['output']-truth),activationderivativefunc(Network[-1]['preactivation']))
        

        for i in range(3,0,-1):
            curr_layer=Network[i]
            curr_weight=curr_layer['weights']
            
            prev_layer=Network[i-1]

            prev_layer_output=prev_layer['output']
            
            curr_layer['weightsgradient']=np.matrix(np.multiply(curr_error,prev_layer_output.transpose()))
            curr_layer['biasgradient']=curr_error
            
            
            if i!=1:

                curr_error=np.array(curr_weight.transpose()*curr_error)
                
                curr_error=np.multiply(curr_error,activationderivativefunc(prev_layer['preactivation']))
    def gradientdescent(learning_rate):

        for i in range(1,4):
           
            curr_layer=Network[i]

            curr_weight=curr_layer['weights']
            prev_layer=curr_weight
            curr_weight_gradient=curr_layer['weightsgradient']

            curr_bias=curr_layer['bias']
            curr_bias_gradient=curr_layer['biasgradient']

            curr_weight=curr_weight-learning_rate*curr_weight_gradient
            curr_bias=curr_bias-learning_rate*curr_bias_gradient

            Network[i]['weights']=curr_weight 
            Network[i]['bias']=curr_bias
             
    def train(image_arr,label,learning_rate):

        for i in range(8):

            run_image(image_arr)
            findgradient(label)
            gradientdescent(learning_rate)


        

    #Everything down is MNIST data processing


    def byte_to_int(n):

        return int.from_bytes(n,'big')
    def getlabelarr(y_train,max_labels):
        

        _=y_train.read(4)
        total_labels=byte_to_int(y_train.read(4))
            
        labels=[]
        for i in range(max_labels):
            
            labels.append((int.from_bytes(y_train.read(1),'big')))
        
        return labels

    def getimgarr(f,max_img):

        

        _=f.read(4)

        total_images=byte_to_int(f.read(4))
        row=byte_to_int(f.read(4))
        col=byte_to_int(f.read(4))
    

        images=[]
        for i in range(max_img):
            curr_img=[]
            for j in range(784):
            
                single_pixel=(byte_to_int(f.read(1)))/255
                curr_img.append(single_pixel)
            
            images.append(curr_img)
        
        return images
    def turntolist(layer):

        properties=[]

        properties.append(layer['weights'].tolist())
        properties.append(layer['bias'].tolist())
        return properties
    print('Network Created')
    
    (x_train,y_train), (x_test,y_test)=tf.keras.datasets.mnist.load_data()

    x_train=x_train/255
    x_test=x_test/255

    x_train=list(zip(x_train,y_train))
    x_test=list(zip(x_test,y_test))
    # num_epochs=17


    def checkaccuracy(num_images):
        global x_test
        global y_test

        total,correct=0,0
        
        for i in range(num_images):
            
            image,label=x_test[i]
            guess=run_image(image)
            
            if guess==label:
                correct+=1 
            total+=1 

        return correct/total
    print('Images Loaded')
    
    # TRAINN NETWORK
    #---------------------------------------------------------------------------------------
    # for x in range(num_epochs):
    #     s=time.time()
    #     random.shuffle(x_train)
    #     for (image,label) in x_train:
            
            

    #         train(image,label,0.2)
    #     print('Epoch '+ str(x+1)+' Completed. Time taken: '+str(time.time()-s)+ ' Current Accuracy: '+str(checkaccuracy(10000)))
    

    # data={0:turntolist(Network[1]),1:turntolist(Network[2]),2:turntolist(Network[3])}
    # with open('data.json','w') as f:
    #     json.dump(data,f)
    #     f.close()
    #-----------------------------------------------------------------------------------------
    
    

    # TEST WITH OUTSIDE IMAGE 

    #-----------------------------------
    # s=Image.open('5.jpg').convert('L')
    # s=s.resize((28,28))
    # s.save('transformed.jpeg')
    # s=np.array(s)/255
    # ans=run_image(s)
    # print(ans)
    #-----------------------------------

    print('Test Accuracy: '+ str(checkaccuracy(10000)))
NueralNetwork()




            