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
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train_scaled, y_train)
y_pred=model.predict(x_test_scaled)
print(y_pred[:5])
print(y_test.iloc[:5].values)
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
r2=r2_score(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
rmse=mean_squared_error(y_test,y_pred)**0.5
print("R2:", r2)
print("MAE:", mae)
print("RMSE:", rmse)
import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred)
plt.plot([0,80],[0,80])
plt.plot([0,80],[0,80], color='red', linestyle='--') # add a dashed line for reference
plt.xlabel("Actual Strength(MPa)")
plt.ylabel("Predicted Strength(MPa)")
plt.title("Actual vs Predicted Concrete Strength")
plt.show()