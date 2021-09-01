import os
import cv2
import numpy as np
import copy
import pandas as pd
import scipy.cluster.vq as vq
from scipy.cluster.vq import whiten
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
#tutor: z5112650@student.unsw.edu.au
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
#from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import roc_curve, auc
from itertools import product, cycle
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
#https://blog.csdn.net/qq_2757   5895/article/details/82628879?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522160354965419195264745600%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=160354965419195264745600&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_v2~rank_v28-1-82628879.pc_first_rank_v2_rank_v28&utm_term=joblib&spm=1018.2118.3001.4187

#as ValueError: Cannot take a larger sample than population when 'replace=False', replace has been changed to be True

def resize(img,size):
    resizedIMG = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    return resizedIMG


def read(img_path):
    imgRead = cv2.imread(img_path)
    imgRGB = cv2.cvtColor(imgRead, cv2.COLOR_BGR2RGB)
    return imgRGB



# for img in glob.glob("path/to/folder/*.png"):
#     cv_img = cv2.imread(img)



def ImgAndLabelList(path):
    ImgList=[]
    LabelList=[]
    for a in os.listdir(path):
        # https://stackoverflow.com/questions/30230592/loading-all-images-using-imread-from-a-given-folder
        # I use the answer of derricw
        if "_rgb.png" in a:
            img_path = os.path.join(path, a)
            imgRGB = read(img_path)
            if imgRGB is not None:
                resizedIMG = resize(imgRGB, 500)
                    # need to resize, otherwiise the image has different size
                ImgList.append(resizedIMG)
                if "ara" in a:
                    LabelList.append(0)
                elif "tobacco" in a:
                    LabelList.append(1)
                        # ara- label0
        # print(LabelList)
        # end
        # os.system("pause")
    return ImgList, LabelList


def WaterShedq(imgGRAY, thresh=95, maxval=255, type=0):
    # set thresh,maxval as 95 255// 160 255
    # plt.imshow(imgthresh,cmap='gray',vmin=0,vmax=255)
    # https://blog.csdn.net/JNingWei/article/details/77747959

    _, imgThresh4 = cv2.threshold(imgGRAY, thresh, maxval, type)
    distance4 = ndimage.distance_transform_edt(imgThresh4)
    # dont know what does footprint=np.ones((3,3)) used for
    local_maxi4 = peak_local_max(distance4, indices=False, footprint=np.ones((3,3)), labels=imgThresh4)

    markers4 = ndimage.label(local_maxi4)[0]
    watershedImg = watershed(-(distance4), markers4, mask=imgThresh4)
    # plt.imshow(ws_labels4,cmap='gray',vmin=0,vmax=255)
    #plt.imshow(ws_labels4)
    return watershedImg



def Surf(ImgList, T=90):
    #https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_surf_intro/py_surf_intro.html?highlight=sift
    ImgDesList=[]
###########################
    # surf= cv2.xfeatures2d.SURF_create(nfeatures)
    # #when t is higher, the amount of kp is smaller
    for img in ImgList:
        ######################plus: watershed
        #img is RGB, print(type(img[0][0][0]))
        #<class 'numpy.uint8'>
        # print(img.shape) (300, 300, 3)
        img1 = copy.deepcopy(img)
        imgGRAY = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        # print(type(imgGRAY[0][0]))
        #<class 'numpy.uint8'>
        imgw=WaterShedq(imgGRAY,75,255)
        #plt.imshow(imgw)
        # print("+++++++")
        # print(type(img1[0][0]))
        #<class 'numpy.int32'>
        #imgw = imgw.astype(np.uint8)
        #print(imgw)#单通道，包包(300,300,1)
        h,w=imgw.shape
        for hi in range(h):
            for wj in range(w):
                if imgw[hi][wj]==0:
                    #the number of label 0 is the largest so, make it to be the background
                    img[hi,wj,0] = 0#
                    img[hi,wj,1] = 0
                    img[hi,wj,2] = 0
        #print(img.shape)  #(300, 300, 3)
######################
    surf = cv2.xfeatures2d.SURF_create(T)
    # when t is higher, the amount of kp is smaller
    #print(ImgList)
    for img in ImgList:
        #########################################################
        keyPoints = surf.detect(img, None)
        # SURFkpsArray = cv2.drawKeypoints(img, keyPoints, None, (255, 0, 0),2)  # if I use 4 in cv2.drawKeypoints(),the circle is too big, So I changed to 2
        # plt.imshow(SURFkpsArray), plt.show()
        # we can see when using surf on image of tobacco, there are many kepypoint on the bowl and other background
        des = surf.detectAndCompute(img, None)[1]
        # print(keyPoints)
        #print(len(des))
        ImgDesList.append(des)
        # print(len(DesList))
        # print("000000000000")
    BunchDesList = []
    for m in ImgDesList:
        for n in m:
            #print(n.shape)
            # print("___")
            # print(type(n))
            # print(i)
            BunchDesList.append(n)
    # # print(len(DesList))
    # # print("++++++")
    return ImgDesList,BunchDesList  # len(ImgDesList) is the number of the whole images in the dataset, it is the high og the final matrix




def cross_val(model, metricsTerm):
    """

    :param metricsTerm:
    :return:
    """
    scores = cross_val_score(model, Array, y_train, cv=10, scoring=metricsTerm)
    return scores


if __name__=="__main__":
    """
    • Arabidopsis: Plant/Ara2013-Canon/*_rgb.png (165 files), Plant/Ara2013-Canon/Metadata.csv.
• Tobacco: Plant/Tobacco/*_rgb.png (62 files), Plant/Tobacco/Metadata.csv."""
    #PlantPath= "D:/UNSW COURSE/COMP9517/pro1_indi/Plant_Phenotyping_Datasets (1)/Plant_Phenotyping_Datasets/Plant"

    AraPath= "D:/UNSW COURSE/COMP9517/pro1_indi/Plant_Phenotyping_Datasets (1)/Plant_Phenotyping_Datasets/Plant/Ara2013-Canon"
    #AraPath="D:/UNSW COURSE/COMP9517/pro1_indi/Plant_Phenotyping_Datasets (1)/Plant_Phenotyping_Datasets/Plant/Ara2013-Canon"
    TobaccoPath= "D:/UNSW COURSE/COMP9517/pro1_indi/Plant_Phenotyping_Datasets (1)/Plant_Phenotyping_Datasets/Plant/Tobacco"
    #TobaccoPath="D:/UNSW COURSE/COMP9517/pro1_indi/Plant_Phenotyping_Datasets (1)/Plant_Phenotyping_Datasets/Plant/Tobacco"
    #if use this code in different computer, please change this path!

    AraImgList=ImgAndLabelList(AraPath)[0]
    #print(AraImgList)#x
    AraLabelList = ImgAndLabelList(AraPath)[1]
    #print(AraLabelList)#y


    TobaccoImgList=ImgAndLabelList(TobaccoPath)[0]
    TobaccoLabelList = ImgAndLabelList(TobaccoPath)[1]
    # print(TobaccoImgList)
    # print(TobaccoLabelList)
    # print(len(TobaccoLabelList))
    dataset=[]
    dataset.extend(AraImgList)
    dataset.extend(TobaccoImgList)#x
    #print(dataset)
    # data_df = pd.DataFrame(dataset)
    # data_df.to_csv('araplustobac.csv')

    labelset= []
    labelset.extend(AraLabelList)
    labelset.extend(TobaccoLabelList) #y
    # data_dfL = pd.DataFrame(labelset)
    # data_dfL.to_csv('Laraplustobac.csv')


    # x_train,x_test,y_train,y_test = train_test_split(dataset,labelset,test_size=0.2)# if use train test split packege
    # print(x_train)
    # print(y_train)
    # print("+++++++++++++++++++++++++++++++")


    # file_name = 'araplustobac.csv'
    # train_data = pd.read_csv(file_name, sep='\t', header=None, nrows=4000)
    # train_data.to_csv('training.tsv', sep='\t', header=None, index=False)
    #
    # test_data = pd.read_csv(file_name, sep='\t', header=None, skiprows=4000)
    # test_data.to_csv('test.tsv', sep='\t', header=None, index=False)

    #https: // blog.csdn.net / m0_38061927 / article / details / 76180541
    X = np.array(dataset)
    y = np.array(labelset)
    ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8, random_state=0)
    for train_index, test_index in ss.split(X, y):
        #print("TRAIN:", train_index, "TEST:", test_index)  #
        X_train, X_test = X[train_index], X[test_index]  # X is train setis for y_train
        y_train, y_test = y[train_index], y[test_index]  # X_test is for y_test , use the trained clf to predict y-train, check whether the y_pred is equal to y_test

    #0.8 is for train and validation,10 fold
    #|---- train ------------------|- validation ----|         |--- test ---|
    #|---- train ---|- validation -|-----train ------|         |--- test ---|
    #.......X 10......
    #|- validation--|----------- train --------------|         |--- test ---|
    #0.3 is test

        #print(type(X_train))
        #<class 'numpy.ndarray'>
    #######################################################################################
    #     train_data = pd.DataFrame(train_set)

    #     train_data.info()

    #     print(train_data[784].value_counts() / len(train_data))
        #8 get features
    set=399
    #the threshold is higher, the keypoint detected is smaller.
    # print(AllimgDes)
    # print("++++++++++++++++++++++++++")
    AllimgDes = Surf(X_train, set)[0]
    BunchTrainDesList = Surf(X_train, set)[1]
    #print(BunchTrainDesList )
    obsDes = whiten(np.array(BunchTrainDesList))
    #130
    kg=130
    codebook = vq.kmeans(obsDes, k_or_guess=130, iter=1, thresh=1e-05, check_finite=True)[0]
    #codebook, distortion = vq.kmeans(obs=BunchTrainDesList, k_or_guess=130, thresh=1e-05, check_finite=True)
    #codebook, distortion = vq.kmeans(obs=BunchTrainDesList, k_or_guess=130, thresh=1e-05, check_finite=False)
    #scipy.cluster.vq.kmeans(obs, k_or_guess, iter=20, thresh=1e-05, check_finite=True)
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans.html
    #https://zhuanlan.zhihu.com/p/67862407
    #print(codebook)
    #print(len(codebook))
    # plt.scatter(obsDes[:, 0], obsDes[:, 1])
    # plt.scatter(codebook[:, 0], codebook[:, 1], c='r')
    # plt.show()
    """
    obsDes：ndarray, 
    Each row of the M by N array is an observation vector. 
    The columns are the features seen during each observation.
     The features must be whitened first with the whiten function.

：M×N whitened

k_or_guess: int or ndarray, 
The number of centroids to generate. 
A code is assigned to each centroid,
 which is also the row index of the centroid in the code_book matrix generated.
  The initial k centroids are chosen by randomly selecting observations from the observation matrix. Alternatively, 
  passing a k by N array specifies the initial k centroids.


iter: int, optional, The number of times to run k-means, returning the codebook with the lowest distortion. This argument is ignored if initial centroids are specified with an array for the k_or_guess parameter. This parameter does not represent the number of iterations of the k-means algorithm.

iter:int，
thresh:float, optional, Terminates the k-means algorithm 
if the change in distortion since the last k-means iteration
 is less than or equal to thresh.前后两次之间rss的变化率差的阈值 ，如果自上次k means迭代以来的变化小于或等于阈值，则终止k均值算法。

check_finite: bool , optional, 
Whether to check that the input matrices 
contain only finite numbers. Disabling may give a 
performance gain, but may result in problems (crashes, non-termination) 
if the inputs do contain infinities or NaNs. Default: True

（3）Returns

codebook  ndarray，A k by N array of k centroids. The i’th centroid codebook[i] is represented with the code i. 
The centroids and codes generated represent the lowest distortion seen, not necessarily the globally minimal distortion.


distortion   float, The mean (non-squared) Euclidean distance between the observations 
passed and the centroids generated. Note the difference to the standard definition of 
distortion in the context of the K-means algorithm, which is the sum of the squared distances.
distortion  float，
"""
    #centroids, labels = kmeans2(X, 3)
    #https://blog.csdn.net/mimiduck/article/details/107842187

    #130 is the k or guess(The number of centroids to generate)=len(codebook)=width
    #(len(AllimgDes))=number of images in the dataset=len(train dataset)
    high=len(AllimgDes)#=227
    width=len(codebook)# = histogram's column=130=k
    FeatureMatrix = np.zeros((high, width), dtype="float32")
    """ vq.vq(obs, codebook): 
    obs : ndarray
Each row of the NxM array is an observation. 
The columns are the “features” seen during each observation. 
The features must be whitened first using the whiten function or something equivalent.

code_book : ndarray. it is usually generated using the k-means algorithm.
 Each row of the array holds a different code, and the columns are the features of the code.

return : code : ndarray ,A length N array holding the code book index for each observation(227个).N即质心的数量（N就是130）

dist : ndarray,The distortion (distance) between the observation and its nearest code."""
    #lis = []
    for i in range(high):#227
        #obs=AllimgDes[i]
        obs=whiten(AllimgDes[i])#high
        #print(len(obs))#每个AllimgDes[i]，

        #lis.append(obs)
        code_book=codebook#width,len is 130 is the column of the histogram
        code = vq.vq(obs, code_book)[0]
        # print(code)
        # print("+++++++")
        #print(len(code))#每个AllimgDes[i]，i.e.
        for WidthIndex in code:#code
            #print(WidthIndex)
            #print(FeatureMatrix.shape)#(227, 130)
            FeatureMatrix[i, WidthIndex] = FeatureMatrix[i, WidthIndex] + 1
    #print(len(lis))
    Array = np.array(FeatureMatrix)
    #Array=FeatureMatrix
    #https: // blog.csdn.net / weixin_41947092 / article / details / 80182276
    #create model,train
    classes=["Arabidopsis","Tobacco"]



    #model
    clf=LinearSVC()
    clf.fit(Array,y_train)
    #clf modle has been trained with the train data


    #creat test dataset array, so _clf can be changed to be _rf or _lr
    #ImgDesList=Surf(X_test,set)[0]
    #开始test！
    BunchTrainDesList_clf = Surf(X_test, set)[1]

    # print(BunchTrainDesList )
    obsDes_clf = whiten(np.array(BunchTrainDesList_clf))
    # 130个类,julei

    AllimgDes_clf = Surf(X_test, set)[0]
    high=len(AllimgDes_clf)#=227
    width=len(codebook)# = histogram's column=130=k
    FeatureMatrix_clf = np.zeros((high, width), dtype="float32")
    for i in range(high):  # 227
        # obs=AllimgDes[i]
        obs_clf = whiten(AllimgDes_clf[i])  # high
        # print(len(obs))#每个AllimgDes[i]，

        # lis.append(obs)
        code_book_clf = codebook   # width,len is 130 is the column of the histogram
        code_clf  = vq.vq(obs_clf, code_book_clf )[0]
        # print(code)
        # print("+++++++")
        # print(len(code))#每个AllimgDes[i]，
        for WidthIndex in code_clf :  # code [ 39  52  97  52  52  97  78  24 100   9  52   9  52 129   9  52  52  54
            # print(WidthIndex)
            # print(FeatureMatrix.shape)#(227, 130)
            FeatureMatrix_clf[i, WidthIndex] = FeatureMatrix_clf[i, WidthIndex] + 1
    # print(len(lis))
    Array_clf = np.array(FeatureMatrix_clf)


    y_pred_clf  = clf.predict(Array_clf)
##########################




    precmean=round(cross_val(clf,'precision').mean(),4)
    recalmean=round(cross_val(clf,'recall').mean(),4)
    aucmean=round(cross_val(clf,'roc_auc').mean(),4)
    F1mean=round(cross_val(clf,'f1').mean(),4)
    #PREscores = cross_val_score(clf, FeatureMatrix, y_train, cv=10,scoring='precision')

    #print(f" AUC is {auc},\n precision is {pre}, \n recall is {rec}.")
    #print(f"acc= {acc}")
    print(f"when using svm: in the validation dataset, mean AUC is {aucmean},\n mean precision is {precmean}, \n mean recall is {recalmean},\n mean f1 is {F1mean}.")
    print(classification_report(y_test, y_pred_clf, target_names=["Arabidopsis","Tobacco"], digits=3))
    # fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_clf, pos_label=2)
    # print(f"when using svm: in the test dataset, the AUC is {metrics.auc(fpr, tpr)}")
    # dataset_size = len(X_test)
    # TwoDim_dataset = X_test.reshape(dataset_size, -1)
    # metrics.plot_roc_curve(clf, TwoDim_dataset, y_test)  # doctest: +SKIP
    # plt.show()
    from sklearn.metrics import average_precision_score
    from sklearn.metrics import plot_roc_curve
    # nsamples, nx, ny =Array_clf.shape
    # d2_train_dataset = Array_clf.reshape((nsamples, nx * ny))

    y_score =clf.decision_function(Array_clf)




    classes = ["Arabidopsis", "Tobacco"]


    def CMclass(yTest=y_test, YPredict=y_pred_clf):
        cm = confusion_matrix(y_test, y_pred_clf)
        #1 is tobbaco
        # 0 is ara
        #here is not about tOrF
        #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
        return cm

    #cm= CMclass()
    def PlotCMatrix(cm, classes= ["Arabidopsis", "Tobacco"], normalize=False, title='Confusion matrix', cmap = "Greens"):
        plt.figure()
        plt.imshow(cm,cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, classes, rotation=0)
        plt.yticks(tick_marks, classes)
        thresh = cm.max() / 3.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j],
                         horizontalalignment='center',
                         color='white' if cm[i, j] > thresh else 'black')
        plt.tight_layout()
        plt.ylabel('True result')
        plt.xlabel('Predicted result')
        plt.show()
        ###########
    #PlotCMatrix(cm)
#################

    model_rgb_rf = RandomForestClassifier(n_estimators=15, max_depth=3, random_state=12)  # 1234随机初始化的种子
    model_rgb_rf.fit(Array, y_train)  #
    y_pred_for =model_rgb_rf.predict(Array_clf)



    #LR
    model_rgb_lr = LogisticRegression(penalty='l2', C=1, random_state=12,max_iter=1000)
    model_rgb_lr.fit(Array, y_train)
    y_pred_lr =model_rgb_lr.predict(Array_clf)



    precmeanf = round(cross_val(model_rgb_rf , 'precision').mean(), 4)
    recalmeanf = round(cross_val(model_rgb_rf , 'recall').mean(), 4)
    aucmeanf = round(cross_val(model_rgb_rf , 'roc_auc').mean(), 4)
    F1mean = round(cross_val(model_rgb_rf, 'f1').mean(), 4)
    # PREscores = cross_val_score(clf, FeatureMatrix, y_train, cv=10,scoring='precision')

    # print(f" AUC is {auc},\n precision is {pre}, \n recall is {rec}.")
    # print(f"acc= {acc}")
    print(f"if use RandomForestClassifier,in the validation dataset, threshold of SURF is {set}, mean AUC is {aucmeanf},\n mean precision is {precmeanf}, \n mean recall is {recalmeanf},\n mean f1 is {F1mean}.")
    print("the classification report of RandomForestClassifier is listed below")
    print(classification_report(y_test, y_pred_for, target_names=["Arabidopsis","Tobacco"], digits=3))


    precmean_lr = round(cross_val(model_rgb_lr, 'precision').mean(), 4)
    recalmean_lr = round(cross_val(model_rgb_lr , 'recall').mean(), 4)
    aucmean_lr = round(cross_val(model_rgb_lr , 'roc_auc').mean(), 4)
    F1mean = round(cross_val(model_rgb_lr, 'f1').mean(), 4)
    # PREscores = cross_val_score(clf, FeatureMatrix, y_train, cv=10,scoring='precision')

    # print(f" AUC is {auc},\n precision is {pre}, \n recall is {rec}.")
    # print(f"acc= {acc}")
    print(f"if use LogisticRegression,in the validation dataset, mean AUC is {aucmean_lr},\n mean precision is {precmean_lr}, \n mean recall is {recalmean_lr},\n mean f1 is {F1mean}.")
    print("the classification report of LR model is listed below")
    print(classification_report(y_test, y_pred_lr, target_names=["Arabidopsis","Tobacco"], digits=3))
    # print(classification_report(y_true, y_pred, target_names=target_names))




    k_range=range(1,25)
    cv_scores=[]#用来放结果
    for n in k_range:
        knn=KNeighborsClassifier(n)#knn
        scores=cross_val_score(knn,Array, y_train,cv=10,scoring='accuracy')
        cv_scores.append(scores.mean())

    plt.plot(k_range,cv_scores)
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.show()


    knnOne=KNeighborsClassifier(1)
    precmean_knn = round(cross_val(knnOne, 'precision').mean(), 4)
    recalmean_knn = round(cross_val(knnOne, 'recall').mean(), 4)
    aucmean_knn = round(cross_val(knnOne, 'roc_auc').mean(), 4)
    F1mean = round(cross_val(knnOne, 'f1').mean(), 4)
    # PREscores = cross_val_score(clf, FeatureMatrix, y_train, cv=10,scoring='precision')

    # print(f" AUC is {auc},\n precision is {pre}, \n recall is {rec}.")
    # print(f"acc= {acc}")
    print(f"As when n_neoghouber is chosen as 1, the score is the best, the n_neighbour is set as 1.\n if use KNeighbors classifier,in the validation dataset, mean AUC is {aucmean_knn},\n mean precision is {precmean_knn}, \n mean recall is {recalmean_knn},\n mean f1 is {F1mean}.")
    print("the classification report of KNeighbors Classifier is listed below")
    knnOne.fit(Array, y_train)
    y_pred_knn = knnOne.predict(Array_clf)
    print(classification_report(y_test, y_pred_knn, target_names=["Arabidopsis", "Tobacco"], digits=3))

    #plot the curve of each models,uncomment this part to plot
    ax = plt.gca()
    rfc_disp_rf = plot_roc_curve(model_rgb_rf,Array_clf, y_test, ax=ax)
    rfc_disp = plot_roc_curve(clf, Array_clf, y_test, ax=ax)
    rfc_disp_lr=plot_roc_curve(model_rgb_lr, Array_clf, y_test, ax=ax)
    rfc_disp_knn=plot_roc_curve(knnOne, Array_clf, y_test, ax=ax)
    plt.show()


