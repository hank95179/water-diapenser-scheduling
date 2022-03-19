import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import csv
index=pd.date_range('20120601','20120601235958',freq='S')
indexmin=pd.date_range('20120601','20120601235958',freq='min')
df1 = pd.read_csv('01/01/2012-06-05.csv')
df2 = pd.read_csv('01/02/2012-06-05.csv')
df4 = pd.read_csv('01/04/2012-06-05.csv')
df5 = pd.read_csv('01/05/2012-06-05.csv')
df6 = pd.read_csv('01/06/2012-06-05.csv')
df7 = pd.read_csv('01/07/2012-06-05.csv')
df1 = np.array(df1,dtype = np.float64)#Fridge XXX
df2 = np.array(df2,dtype = np.float64)#Dryer
df4 = np.array(df4,dtype = np.float64)#Kettle
df5 = np.array(df5,dtype = np.float64)#Washing machine
df6 = np.array(df6,dtype = np.float64)#PC XXX
df7 = np.array(df7,dtype = np.float64)#Freezer XXX
denki1 = pd.DataFrame(df2,index=index)
denki1 = denki1.resample('min').sum()
denki1 = np.array(denki1,dtype = np.float64)
denki2 = pd.DataFrame(df4,index=index)
denki2 = denki2.resample('min').sum()
denki2 = np.array(denki2,dtype = np.float64)
denki3 = pd.DataFrame(df5,index=index)
denki3 = denki3.resample('min').sum()
denki3 = np.array(denki3,dtype = np.float64)
for i in range(1440) :
	if denki1[i] <= 0:
		denki1[i] = 0
	else:
		denki1[i] = 1
	if denki2[i] <= 0:
		denki2[i] = 0
	else:
		denki2[i] = 1
	if denki3[i] <= 0:
		denki3[i] = 0
	else:
		denki3[i] = 1
for x in range(86399):
    df1[x] = df2[x] + df1[x] + df4[x] + df6[x] + df5[x] + df7[x]
denki1 =  pd.DataFrame(denki1,index=indexmin)
denki1 = np.array(denki1,dtype = np.int)
denki2 =  pd.DataFrame(denki2,index=indexmin)
denki2 = np.array(denki2,dtype = np.int)
denki3 =  pd.DataFrame(denki3,index=indexmin)
denki3 = np.array(denki3,dtype = np.int)
data = pd.DataFrame(df1,index=index)
data = data.resample('min').sum()  
data = np.array(data,dtype = np.float64)
x_train, x_test, y_train, y_test = train_test_split(data, denki1, test_size=0.25, random_state=10)
model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
model.fit(x_train,y_train)
test = model.predict(x_test)
model1_test_0 = 0
model1_test_1 = 0
model1_ans_0 = 0
model1_ans_1 = 0
print("預測的分鐘數:",len(test))
for i in range(len(test)):
	if test[i] == 0:
		model1_test_0 += 1
	else:
		model1_test_1 += 1
	if y_test[i] == 0:
		model1_ans_0 += 1
	else:
		model1_ans_1 += 1
ac = accuracy_score(test, y_test)
print("預測精準度:",ac)
print("預測上為0(沒用電的分鐘數):",model1_test_0,"預測上為1(有用電的分鐘數):",model1_test_1)
print("實際上為0(沒用電的分鐘數):",model1_ans_0,"實際上為1(有用電的分鐘數):",model1_ans_1)
# d = []
# d.append(ac)
# fd = pd.Series(d)
# fd.to_csv('AC01.csv',mode='a',index = False)

x_train2, x_test2, y_train2, y_test2 = train_test_split(data, denki2, test_size=0.25, random_state=10)
model2 = make_pipeline(StandardScaler(), SVC(gamma='auto'))
model2.fit(x_train2,y_train2)
test2 = model2.predict(x_test2)
model2_test_0 = 0
model2_test_1 = 0
model2_ans_0 = 0
model2_ans_1 = 0
print("預測的分鐘數:",len(test2))
for i in range(len(test2)):
	if test2[i] == 0:
		model2_test_0 += 1
	else:
		model2_test_1 += 1
	if y_test2[i] == 0:
		model2_ans_0 += 1
	else:
		model2_ans_1 += 1
ac2 = accuracy_score(test2, y_test2)
print("預測精準度:",ac2)
print("預測上為0(沒用電的分鐘數):",model2_test_0,"預測上為1(有用電的分鐘數):",model2_test_1)
print("實際上為0(沒用電的分鐘數):",model2_ans_0,"實際上為1(有用電的分鐘數):",model2_ans_1)
# # print(model2.predict(x_test2))
# ac2 = accuracy_score(model2.predict(x_test2), y_test2)
# print(ac2)
# d2 = []
# d2.append(ac2)
# fd2 = pd.Series(d2)
# fd2.to_csv('AC02.csv',mode='a',index = False)

x_train3, x_test3, y_train3, y_test3 = train_test_split(data, denki3, test_size=0.25, random_state=10)
model3 = make_pipeline(StandardScaler(), SVC(gamma='auto'))
model3.fit(x_train3,y_train3)
test3 = model3.predict(x_test3)
model3_test_0 = 0
model3_test_1 = 0
model3_ans_0 = 0
model3_ans_1 = 0
print("預測的分鐘數:",len(test3))
for i in range(len(test3)):
	if test3[i] == 0:
		model3_test_0 += 1
	else:
		model3_test_1 += 1
	if y_test3[i] == 0:
		model3_ans_0 += 1
	else:
		model3_ans_1 += 1
ac3 = accuracy_score(test3, y_test3)
print("預測精準度:",ac3)
print("預測上為0(沒用電的分鐘數):",model3_test_0,"預測上為1(有用電的分鐘數):",model3_test_1)
print("實際上為0(沒用電的分鐘數):",model3_ans_0,"實際上為1(有用電的分鐘數):",model3_ans_1)
# # print(model3.predict(x_test3))
# ac3 = accuracy_score(model3.predict(x_test3), y_test3)
# print(ac3)
# d3 = []
# d3.append(ac3)
# fd3 = pd.Series(d3)
# fd3.to_csv('AC03.csv',mode='a',index = False)