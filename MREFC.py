# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 15:33:06 2019

@author: yanglei
"""

import numpy as np
import time
from keras.layers import Dense, Input
from keras.models import Model
import sklearn.neighbors as skn
from scipy.spatial.distance import cosine
from scipy.io import loadmat,savemat
time_s=time.time()

def load_data():
    path='./dataset/mnist.npz'
    f=np.load(path)
    x_train,y_train=f['x_train'],f['y_train']
    x_test,y_test=f['x_test'],f['y_test']
    f.close()    
    return (x_train,y_train),(x_test,y_test)

def re_measure(x_train,x_test):
    # create copies of the data
    n,d=x_train.shape
    print(n,d)
    
    remain_idx = list(set(list(range(d))))
        
    input_dim = Input(shape = (d, ))
    encoding_dim = 300
    encoded = Dense(encoding_dim, activation = 'sigmoid')(input_dim)
    decoded = Dense(d, activation = 'sigmoid')(encoded)
    autoencoder = Model(input = input_dim, output = decoded)
    autoencoder.compile(optimizer = 'adadelta', loss = 'mse')
    autoencoder.fit(x_train, x_train, nb_epoch = 20, batch_size = 128, shuffle = True, validation_data = (x_test, x_test),verbose=0)
    auto_acc_list = []                    
    x_train_auto = x_train
    
    #remove feat_idx 特征，记录其cost
    for feat_idx in range(d):
        x_train = x_train_auto
        x_train = x_train.transpose()
        new_x_train = np.append(x_train[:feat_idx], np.full((1, x_train.shape[1]), np.mean(x_train[feat_idx])), axis=0)
        x_train = np.append(new_x_train, x_train[feat_idx+1:], axis=0)
        x_train = x_train.transpose()
        x_train_encoded = autoencoder.predict(x_train,verbose=0)
        auto_cost = x_train_encoded - x_train_auto
        auto_cost = auto_cost ** 2
        auto_cost = sum(sum(auto_cost))
        
        #取出当前feat_idx在 原特征集中的 index
        index=remain_idx[feat_idx]
        auto_acc_list.append((index, auto_cost))
    
    cost_array = [auto_acc[1] for auto_acc in auto_acc_list]
    cost_array = np.array(cost_array)
    cost_array = (cost_array - min(cost_array))/ (max(cost_array) - min(cost_array))
    for auto_index in range(len(auto_acc_list)):
        auto_acc_list[auto_index] = (auto_acc_list[auto_index][0], cost_array[auto_index])
    auto_acc_list.sort(key=lambda x: x[1])
    #np.save('./selected_feature/selected_fea_AE_cost_scaler.npy',auto_acc_list)  

    #auto_acc_list=np.load('./selected_feature/selected_fea_AE_cost_scaler.npy')
    z=0
    auto_acc_list1=np.asarray(auto_acc_list)
    auto_acc_list=list(auto_acc_list1[np.argsort(-auto_acc_list1[:,1])])
    print(auto_acc_list)
    #auto_acc_list.sort(key=lambda x: -x[1])
    cost_list=[i[1] for i in auto_acc_list]
    auto_sum=np.sum(cost_list)
    r=0.9
    for i in range(d):
        #threshold=1-(np.sum(cost_list[:i])/auto_sum)
        threshold=(np.sum(cost_list[:i])/auto_sum)
        if(threshold>r):
            z=i
            print(threshold)
            break
    print('re selected feature : ',z)
    final_index=[int(i[0]) for i in auto_acc_list][:z]
    #np.save('./selected_feature/re_selected_index.npy',final_index)
    print()
    time_end=time.time()
    print('total time:',time_end-time_s)
    return final_index

def cosine_dis(x,y):
    s=(np.linalg.norm(x)*np.linalg.norm(y))
    if(s==0):
        return 0
    else:
        return np.dot(x,y)/s


#non-local distances
def cosine_dis_nonlocal(x,y):
    s=(np.linalg.norm(x)*np.linalg.norm(y))
    if(s==0):
        return 0
    else:
        if(np.dot(x,y)==0):
            return 0
        else:
            return 1/(np.dot(x,y)/s)

def knn_graph_local(X,k):
    d,n=np.shape(X)
    A=skn.kneighbors_graph(X.transpose(),n_neighbors=k,mode='distance',metric=cosine_dis,include_self=None)
    A=A.toarray()
    D=np.zeros([n,n])
    for i in range(n):
        D[i,i]=np.sum(A[i,:])
    L=D-A
    return L
def knn_graph_nonlocal(X,k):
    d,n=np.shape(X)
    A=skn.kneighbors_graph(X.transpose(),n_neighbors=k,mode='distance',metric=cosine_dis_nonlocal,include_self=None)
    A=A.power(-1)
    A=A.toarray()
    D=np.zeros([n,n])
    for i in range(n):
        D[i,i]=np.sum(A[i,:])
    L=D-A
    return L

def xavier_init(fan_in,fan_out,constant=1):
    low=-constant*np.sqrt(6.0/(fan_in+fan_out))
    high=constant*np.sqrt(6.0/(fan_in+fan_out))
    return np.random.uniform(low=low,high=high,size=(fan_in,fan_out))

def y_encode(W1,b1,x,m):
    return 1/(1+np.exp(-(np.dot(W1,x).reshape([m,1])+b1)))

def Xre_decode(W2,b2,y,d):
    return 1/(1+np.exp(-(np.dot(W2,y).reshape([d,1])+b2)))

def sigmid(x):
    return 1/(1+np.exp(-x))

def objective_opt(X,m,gam,lam,bate,k):
    d,n=np.shape(X)              # !!!  row is feature  column is the number of sample 
    print(n,d)
    e=0.0001
    max_iteration=300
    diff=0.00001
    fun_diff=1
    iteration=0
    prior_fun=10000
    
    W1=xavier_init(m,d)
    W2=xavier_init(d,m)
    b1=xavier_init(m,1)
    b2=xavier_init(d,1)
    
    Y=np.zeros([m,n])
    Xre=np.zeros([d,n])
    U=np.eye(d)
    #L=knn_graph(X,k)
    L1=knn_graph_local(X,k)
    Ln=knn_graph_nonlocal(X,k)
    
    score_index=0
    score_result=np.zeros((max_iteration+1,1))
    
    #stop condition:(1) max iteration (2)the difference between two iteration of obecjive_fun less than threshold
    while((iteration<=max_iteration)and(fun_diff>=diff)):
        for i in range(n):
            Y[:,i]=y_encode(W1,b1,X[:,i],m).reshape(m,)
            Xre[:,i]=Xre_decode(W2,b2,Y[:,i],d).reshape(d,)
        
        #objective function
        L_fun=(1/(2*n))*np.power(np.linalg.norm((X-Xre),ord='fro'),2)
        R_fun=lam*np.linalg.norm( np.linalg.norm(W1,axis=0),ord=1 )    #先对列求2范数，再求1范数
        G_fun=gam*np.ndarray.trace( np.dot(np.dot(Y,L1), Y.transpose()) )  /\
                (np.ndarray.trace( np.dot(np.dot(Y,Ln), Y.transpose()) ))
        W_fun=bate*(np.linalg.norm(W1,ord='fro')+np.linalg.norm(W2,ord='fro')+np.linalg.norm(b1,ord='fro')+np.linalg.norm(b2,ord='fro'))
        F_fun=L_fun+R_fun+G_fun+W_fun
        
        fun_diff=abs(prior_fun-F_fun)
        prior_fun=F_fun
        
        
        delta3=np.multiply( np.multiply( (Xre-X),Xre ) , (np.ones([d,n])-Xre) )
        delta2=np.multiply( np.multiply( np.dot(W2.transpose(),delta3) ,Y) ,(np.ones([m,n])-Y) )
        
        #compute U matrix
        for i in range(d):
            nm=np.linalg.norm(W1[:,i])
            if(nm==0):
                U[i,i]=0
            else:
                U[i,i]=1/(nm+e)
                
        #the partial of F_fun 
        part1=np.dot(Y,L1)/ (np.ndarray.trace( np.dot(np.dot(Y,Ln), Y.transpose()) ))
        part2=np.dot(Y,Ln)/ np.power((np.ndarray.trace( np.dot(np.dot(Y,Ln), Y.transpose()) )) ,2)
        part=part1-part2
        W1_partial=(1/n)*np.dot(delta2,X.transpose())+lam*np.dot(W1,U)+\
                    2*gam*np.dot( np.multiply(np.multiply(part,Y),(np.ones([m,n])-Y)) ,X.transpose()) + bate*W1
        W2_partial=(1/n)*np.dot(delta3,Y.transpose())+bate*W2
        b1_partial=(1/n)*np.dot(delta2,np.ones([n,1]))+\
                    2*gam*np.dot( np.multiply(np.multiply(part,Y),(np.ones([m,n])-Y)) ,np.ones([n,1]))+bate*b1
        b2_partial=(1/n)*np.dot(delta3,np.ones([n,1]))+bate*b2
        
        W1=W1-0.1*W1_partial
        W2=W2-0.1*W2_partial
        b1=b1-0.1*b1_partial
        b2=b2-0.1*b2_partial
        
        print(iteration,F_fun,fun_diff)
        score_result[score_index]=F_fun
        score_index+=1
        iteration+=1
    #print(W1)
    score=np.zeros([d,])
    for i in range(d):
        score[i]=np.linalg.norm(W1[:,i])
    index=np.argsort(score)
    #index_fin=(index+1)             #index+1
    savemat('iteration',{'score':score_result})
    return index

def fc_measure(selected_index,x_train):
    x_train=x_train.transpose()[selected_index,:]                  #!!!!
    print(x_train.shape)
    final_feature=objective_opt(x_train,m=300,lam=0.01,gam=0.005,bate=0.01,k=5)
    selected_index=np.asarray(selected_index)
    index=selected_index[final_feature]
    #savepath='./final_index/result_k(5)m(200)_iter(300)_a(0.1).npy'
    #np.save(savepath,index) 

if __name__=='__main__':
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train = x_train.reshape(x_train.shape[0], 784)
    x_test = x_test.reshape(x_test.shape[0], 784)
    # preprocess the data
    x_train = x_train.astype('float32')[:10000,:]
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples') 
    '''
    final_index=np.load('./selected_feature/re_selected_index.npy')
    selected_index=final_index
    '''
    selected_index=re_measure(x_train,x_test)
    fc_measure(selected_index,x_train)