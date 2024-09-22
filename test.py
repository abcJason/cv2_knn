import cv2
import numpy as np

s='image\\'  #圖像所在路徑
num=100 #共有樣本數量
row=240 #每個數字圖像的行數
col=240 #每個數字圖像的列數
a=np.zeros((num,row,col)) #用來存儲所有樣本的數值

n=0 #用來存儲當前圖像的編號
for i in range(0,10):
    for j in range(1,11):
        a[n,:,:]=cv2.imread(s+str(i)+'\\'+str(i)+'-'+str(j)+'.bmp',0)
        n=n+1

#提取樣本圖像的特徵
feature=np.zeros((num,round(row/5),round(col/5))) #用來存儲所有樣本的特徵值
#print(feature.shape)  #看看feature的shape長什麼樣子
#print(row)            #看看row的值，有多少個特徵（100）個

for ni in range(0,num):
    for nr in range(0,row):
        for nc in range(0,col):
            if a[ni,nr,nc]==255:
                feature[ni,int(nr/5),int(nc/5)]+=1
f=feature   #簡化變量名稱

train = feature[:,:].reshape(-1,round(row/5)*round(col/5)).astype(np.float32) 
print(train.shape)
#貼標籤，需要注意range(0,100)不是range(0,101)
trainLabels = [int(i/10)  for i in range(0,100)]
trainLabels=np.asarray(trainLabels)
print(*trainLabels)   #打印測試看看標籤值
print(type(trainLabels))   #打印測試看看標籤值

#####計算當前待識別圖像的特徵值
o=cv2.imread('image\\test\\5.bmp',0) #讀取待測圖像
##讀取圖像值
of=np.zeros((round(row/5),round(col/5))) #用來存儲測試圖像的特徵值
for nr in range(0,row):
    for nc in range(0,col):
        if o[nr,nc]==255:
            of[int(nr/5),int(nc/5)]+=1

test=of.reshape(-1,round(row/5)*round(col/5)).astype(np.float32) 
#調用函數識別
knn=cv2.ml.KNearest_create()
knn.train(train,cv2.ml.ROW_SAMPLE, trainLabels)
ret,result,neighbours,dist = knn.findNearest(test,k=7)
print("當前隨機數可以判定為類型：", result)
print("距離當前點最近的7個鄰居是：", neighbours)
print("7個最近鄰居的距離 : ", dist)