# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 11:48:49 2018

@author: ndoannguyen
"""

from Constants import *
from SingerClassification import *
from itertools import chain
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_auc_score

#TEST 1
prepareImageFolder(RAW_IMAGE_FOLDER, IMAGE_FOLDER)
z = os.listdir(IMAGE_FOLDER)

"""
#TEST 2
face = Face(IMAGE_FOLDER + "/BaoThy_0.png")
print(face.gray_image)
"""

"""
#TEST 3
face = Face(IMAGE_FOLDER + "/BaoThy_0.png")
face.draw()
"""

#TEST 4
"""
for i in range(10):
    face = Face(IMAGE_FOLDER + "/BaoThy_%d.png" % i)
    face.detectFaces()
    print(face.face_positions[0])
    face.draw(mode = "face_marked")
"""

"""
#TEST 5
face = Face(IMAGE_FOLDER + "/BaoThy_25.png")
face.draw(mode = "full_gray", index = 0)
"""

"""
#TEST 6
face = Face(IMAGE_FOLDER + "/BaoThy_10.png")
face.detectEyes()
face.draw(mode = "normalized_eye")
"""

"""
#TEST 8
transformImagesToText(IMAGE_FOLDER, TEXT_DATA_FOLDER, BAOTHY_DAMVINHHUNG_TRAIN, chain(SINGER_IMAGE_RANGE["BaoThy"][:100], SINGER_IMAGE_RANGE["DamVinhHung"][:100]))
"""

#TEST 9
"""
singers = ["DamVinhHung", "DanTruong", "LamTruong", "NooPhuocThinh"]
transformImagesToFacesTable(IMAGE_FOLDER, TEXT_DATA_FOLDER, FACES_DATA, chain.from_iterable([SINGER_IMAGE_RANGE[s] for s in singers]))
"""

"""
singers = ["DamVinhHung", "DanTruong", "LamTruong", "NooPhuocThinh"]
transformImagesToEyesTable(IMAGE_FOLDER, TEXT_DATA_FOLDER, EYES_DATA, chain.from_iterable([SINGER_IMAGE_RANGE[s] for s in singers]))
"""

"""
data_set = DataSet(FACES_DATA)
X = data_set.X
y = data_set.y
names = data_set.names
data_set.trainTestSplit()
model = SVC(kernel = 'linear', decision_function_shape='ovr')
score = data_set.score(model)
confusion_matrix = data_set.getConfusionMatrix(model)
#prediction = data_set.predict(model)
coefs = model.coef_
significant_features = data_set.getSignificantFeatures(model, seuil = 0.0005)
"""


data_set = DataSet(FACES_DATA, selected_columns = significant_features)
X = data_set.X
y = data_set.y
names = data_set.names
data_set.trainTestSplit()
model = SVC(kernel = 'linear', decision_function_shape='ovr', C=1)
score = data_set.score(model)
confusion_matrix = data_set.getConfusionMatrix(model)


"""
SPLIT = 60
singers = ["BaoThy", "DanTruong"]
transformImagesToEyesTable(IMAGE_FOLDER, TEXT_DATA_FOLDER, DATA_TRAIN, chain.from_iterable([SINGER_IMAGE_RANGE[s][:SPLIT] for s in singers])) 
transformImagesToEyesTable(IMAGE_FOLDER, TEXT_DATA_FOLDER, DATA_TEST, chain.from_iterable([SINGER_IMAGE_RANGE[s][SPLIT:] for s in singers]))
"""

"""
trainDataSet = EyesDataSet(DATA_TRAIN)
X_train = trainDataSet.X
y_train = trainDataSet.y
names_train = trainDataSet.names

testDataSet = EyesDataSet(DATA_TEST)
X_test = testDataSet.X
y_test = testDataSet.y
names_test = testDataSet.names

clf = SVC(kernel = 'linear', decision_function_shape = 'ovr')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
score = clf.score(X_test, y_test)
print(confusion_matrix(y_test, y_pred))
print(roc_auc_score(y_test, y_pred))
"""

"""
o = clf.coef_
SEUIL = 0.0005
indices = [i for i in range(len(o[0])) if abs(o[0,i]) > SEUIL] #or abs(o[1, i]) > SEUIL or abs(o[2, i]) > SEUIL]

trainDataSet = EyesDataSet(DATA_TRAIN, selected_columns = indices)
X_train = trainDataSet.X
y_train = trainDataSet.y
names_train = trainDataSet.names

testDataSet = EyesDataSet(BAOTHY_DAMVINHHUNG_TEST, selected_columns = indices)
X_test = testDataSet.X
y_test = testDataSet.y
names_test = testDataSet.names

clf = SVC(kernel = 'linear', C = 0.0001)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
score = clf.score(X_test, y_test)
print(confusion_matrix(y_test, y_pred))
"""