import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix , classification_report
df=pd.read_csv(r"C:\Users\pc\Desktop\creditcard.csv")
print(df.head())
print(df.info())
print("nbH : ",df.shape[0])
print("nbV : ",df.shape[1])
print("nombre operations normales : ")
print(df['Class'].value_counts()[0])
print("nombre operations fraudes : ")
print(df['Class'].value_counts()[1])
fraude_prc = (df['Class'].value_counts()[1]/df.shape[0])*100
print("prc du fraude : ",fraude_prc ,"%")
x=df.drop('Class',axis=1)
y=df['Class']
test_size=0.3
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
scaler = StandardScaler()
x_train['Amount']=scaler.fit_transform(x_train[['Amount']])
x_test['Amount']=scaler.transform(x_test[['Amount']])
print("on est bien decouper les infos")
print("volume des infos (X_train):",x_train.shape)
model=LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
y_pred=model.predict(x_test)
print("Rapport de IA sur fraudes")
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test, y_pred))
