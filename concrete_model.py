import pandas as pd
df=pd.read_excel("Concrete_Data.xls")
print(df.head())
df.columns=["cement","slag","fly_ash","water","superplasticizer","coarse_aggregate","fine _aggregate","age","strength"]
print(df.head())
x=df.drop("strength",axis=1)
y=df["strength"]
print(x.shape)
print(y.shape)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(x_train.shape)
print(x_test.shape)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
print(x_train_scaled[0])


