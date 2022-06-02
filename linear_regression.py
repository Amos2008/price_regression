import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from sklearn import linear_model
model_LinearRegression = linear_model.LinearRegression()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('train.csv')


data['MSSubClass']=data['MSSubClass'].astype(str)
MSSubClass = data.loc[:,'MSSubClass']
#提取Id第一行
Id=data.loc[:,'Id']   #ID先提取出来，后面合并表格要用
data=data.drop('Id',axis=1)

x=data.loc[:,data.columns!='SalePrice']
y=data.loc[:,'SalePrice']


mean_cols=x.mean()
print(mean_cols)
x=x.fillna(mean_cols)  #填充缺失值
x_dum=pd.get_dummies(x)    #独热编码
x_train,x_test,y_train,y_test = train_test_split(x_dum,y,test_size = 0.3,random_state = 1)

#再整理出一组标准化的数据，通过对比可以看出模型的效果有没有提高
x_dum=pd.get_dummies(x)
scale_x=StandardScaler()
x1=scale_x.fit_transform(x_dum)
scale_y=StandardScaler()
y=np.array(y).reshape(-1,1)
y1=scale_y.fit_transform(y)
y1=y1.ravel()
x_train1,x_test1,y_train1,y_test1 = train_test_split(x1,y1,test_size = 0.3,random_state = 1)


model_LinearRegression.fit(x_train,y_train)
y_pred = model_LinearRegression.predict(x_train)
plt.figure(figsize=(14,4))
plt.scatter(x_train, y_train, color='g')
plt.plot(x_train, y_pred, color='r')
plt.xlabel('time（0000-2400）')
plt.ylabel('blood glucose value')

