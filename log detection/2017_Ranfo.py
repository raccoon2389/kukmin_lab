import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

train_x = np.load('log detection/dataset/req2log2.npy')
test_x = np.load('log detection/dataset/req2logTEST2.npy')
anomal_x = np.load('log detection/dataset/req2logANOMAL2.npy')

train_y = np.zeros((train_x.shape[0]))
anomal_y = np.zeros((anomal_x.shape[0])) + 1
train_data = np.concatenate([train_x,anomal_x],axis=0)

train_label = np.concatenate([train_y,anomal_y]).reshape(-1,1)

train_x,test_x,train_y , test_y = train_test_split(train_data,train_label,test_size=0.4,shuffle=True,random_state=666)

rnd_clf = RandomForestClassifier(n_estimators=1000,max_leaf_nodes=400,max_depth=50,random_state=666,n_jobs=-1)
rnd_clf.fit(train_x,train_y)
y_pred_rf = rnd_clf.predict(test_x)

a=0
for i in y_pred_rf:
    a+=i
# print(a)



y_test = np.zeros(y_pred_rf.shape[0])
print(confusion_matrix(test_y,y_pred_rf))
print(classification_report(test_y,y_pred_rf))
print(accuracy_score(test_y, y_pred_rf))