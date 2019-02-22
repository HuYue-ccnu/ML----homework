# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import mpl
l_people = list()
l_square = list()
l_money = list()
fr = open("ex1data.txt",'r',encoding='utf-8')

for line in fr.readlines()[1:]:
    line_ = line[line.index(' ')+1:]
#    print(line_)
    people = line_[:line_.index(' ')].strip()
#    print(people)
    two = line_[line_.index(' ')+1:]
    square = two[:two.index(' ')].strip()
    money = two[two.index(' ')+1:].strip()
    l_people.append(float(people))
    l_square.append(float(square))
    l_money.append(float(money))
#    s = line.split(' ')

fr.close()
x_1 = []*len(l_people)
_ = []
a = []
y_1 = []*len(l_people)
for i in range(len(l_people)):
    _.append(l_people[i])
    _.append(l_square[i])
    x_1.append(_)
    a.append(l_money[i])
    y_1.append(a)
#    print(_)
    _ = []
    a = []


#加载数据
def load_data(filename):
    data = []
    with open(filename, 'r',encoding='utf-8') as f:
        for line in f.readlines()[1:]:
            line_ = line[line.index(' ')+1:]
            line_ = line_.split(' ')
            current = [float(item) for item in line_]
            data.append(current)
    return data
data = load_data('ex1data.txt');
data = np.array(data,np.float32)

x = data[:,(0,1)].reshape((-1,2))
y = data[:,2].reshape((-1,1))
m = y.shape[0]


#特征缩放
def featureNormalize(X):
    X_norm = X
    mu = np.zeros((1,X.shape[1]))
    sigma = np.zeros((1,X.shape[1]))
    for i in range(X.shape[1]):
        mu[0,i] = np.mean(X[:,i]) # 均值
        sigma[0,i] = np.std(X[:,i])     # 标准差
        
        X_norm  = (X - mu) / sigma
    return X_norm,mu,sigma

#计算损失
def computeCost(X,y,theta):
    m = y.shape[0]
    C = X.dot(theta) - y 
    J2 = (C.T.dot(C))/(2*m)
    
    return J2

#梯度下降
def gradientDescent(X,y,theta,alpha,num_iters):
    m = y.shape[0]
    # 存储历史误差
    J_history = np.zeros((num_iters, 1))
    for iter in range(num_iters):
        # 对J求导，得到 alpha/m * (WX - Y)*x(i)， (3,m)*(m,1)  X (m,3)*(3,1) = (m,1)
        theta = theta - (alpha/m) * (X.T.dot(X.dot(theta) - y))
        J_history[iter] = computeCost(X, y, theta)
        
    return J_history,theta

iterations = 1000  #迭代次数
alpha = 0.01    #学习率
x = data[:,(0,1)].reshape((-1,2))
y = data[:,2].reshape((-1,1))
m = y.shape[0]
x,mu,sigma = featureNormalize(x)
X = np.hstack([x,np.ones((x.shape[0], 1))])

theta = np.zeros((3, 1))

j = computeCost(X,y,theta)
J_history,theta = gradientDescent(X, y, theta, alpha, iterations)
print('Theta found by gradient descent',theta)

plt.plot(J_history)
plt.ylabel('lost');
plt.xlabel('iter count')
plt.title('convergence graph')

"""完成函数的绘制"""
def draw(data_x,old_y,new_y):
    #创建绘图函数对象
    fig = plt.figure()
    #创建Axes3D对象，让其包含图像3D坐标
    ax = Axes3D(fig)
    ax.scatter(data_x[0], data_x[1], old_y, color='green')
    ax.plot(data_x[0], data_x[1], new_y, color='blue')
    ax.set_xlabel('城市人口')
    ax.set_ylabel('城市面积')
    ax.set_zlabel('饭店利润')
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.title("饭店的利润和城市人口与城市面积的关系")
    plt.show()
newY = []
def predict(data):
    testx = np.array(data)
    testx = ((testx - mu) / sigma)
    testx = np.hstack([testx,np.ones((testx.shape[0], 1))])
    price = testx.dot(theta)
    return float(price)
print(predict([5.7077,60.5252282059873]))
test = []
for i in range(len(l_people)):
    test.append(l_people[i])
    test.append(l_square[i])
    newY.append(predict(test))
    test.clear()

x1 = l_people
x2 = l_square
draw([x1,x2], y_1, newY)
print(theta)