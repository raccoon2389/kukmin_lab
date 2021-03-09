import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

train_x = np.load('log detection/dataset/req2log2.npy')
test_x = np.load('log detection/dataset/req2logTEST2.npy')
anomal_x = np.load('log detection/dataset/req2logANOMAL2.npy')

train_y = np.zeros((train_x.shape[0]))
anomal_y = np.zeros((anomal_x.shape[0])) + 1
train_data = np.concatenate([train_x,anomal_x],axis=0)

train_label = np.concatenate([train_y,anomal_y]).reshape(-1,1)

rnd_clf = RandomForestClassifier(n_estimators=1000,max_leaf_nodes=200,max_depth=12,random_state=666,n_jobs=-1)
rnd_clf.fit(train_data,train_label)
y_pred_rf = rnd_clf.predict(train_data)

a=0
for i in y_pred_rf:
    a+=i
print(a)



y_test = np.zeros(y_pred_rf.shape[0])
print(confusion_matrix(train_label,y_pred_rf))
print(classification_report(train_label,y_pred_rf))
print(accuracy_score(train_label, y_pred_rf))