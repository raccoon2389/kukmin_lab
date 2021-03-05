import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

train_x = np.load('log detection/dataset/req2log.npy')
test_x = np.load('log detection/dataset/req2logTEST.npy')
anomal_x = np.load('log detection/dataset/req2logANOMAL.npy')

train_y = np.zeros((train_x.shape[0]))
anomal_y = np.zeros((anomal_x.shape[0])) + 1
train_data = np.concatenate([train_x,anomal_x],axis=0)

train_label = np.concatenate([train_y,anomal_y]).reshape(-1,1)

rnd_clf = RandomForestClassifier(n_estimators=400,max_leaf_nodes=100,max_depth=1000,random_state=0,n_jobs=-1)
rnd_clf.fit(train_data,train_label)
y_pred_rf = rnd_clf.predict(test_x)

a=0
for i in y_pred_rf:
    a+=i
print(a)

y_test = np.zeros(y_pred_rf.shape[0])
print(confusion_matrix(y_test,y_pred_rf))
print(classification_report(y_test,y_pred_rf))
print(accuracy_score(y_test, y_pred_rf))