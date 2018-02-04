# hellow-word
just another repository 
import numpy as np
class Perceptron(object):
    '''
    eta  :学习率
    n_iter:权重向量的训练次数
    w_：神经分叉权重向量
    errors_：用于记录神经元判断出错次数
    '''
    def _int_(self,eta = 0.01,n_iter=10):
        self.eta = eta;
        self.n_iter = n_iter
        pass
    def fit(self,x,y):
        '''
        输入训练数据，培训神经元，x输入样本向量，y对应样本分类
        
        X：shape[[1,2,3],[4,5,6]]
        n_samples:2
        n_features:3
        
        y:[1,-1]
        '''
        '''
        初始化权重向量为0
        加一是因为前面算法提到的w0,也就是步调函数阈值
        '''
        self.w_=np.zero(1 + X.shape[1])
        self.errors_ = []
        
        for _ in rang(self.n_iter) :
            errors = 0
            '''
            x:[[1,2,3],[4,5,6]]
            y:[1,-1]
            zip(x,y) = [[1,2,3,1],[4,5,6 -1]]
            '''
            
            for xi,target in zip(x,y):
                '''
                update = n * (y - y')
                '''
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update;
                
                errors += int(update != 0.0)
                self.errors_.append(errors)
                pass
            
            pass
        def net_input(self,x):
            '''
            z = w0*1 +w1*1 +....wn*xn
            '''
            return np.dot(x,self.w_[1:]) + self.w_[0]
            pass
        
        def predict(self,x):
            return np.where(self.net_input(x) >=0.0 , 1,-1)
            pass
        pass


import matplotlib.pyplot as plt
import numpy as np

y = df.loc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

x= df.iloc[0:100,[0, 2]].values
print(x)


x = df.iloc[0:100,[0,2]].values

plt.scatter(x[:50, 0],x[:50, 1],color='red',marker='o',lable='setosa')
plt.scatter(x[50:100, 0],x[50:100, 1],color='blue',marker='x',lable='versicolor')
plt.xlable('花瓣长度')
plt.ylable('花径长度')
plt.legend(loc='upper left')
plt.show()
