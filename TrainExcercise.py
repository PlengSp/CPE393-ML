import pandas as pd                         
import pickle                               
from sklearn.ensemble import RandomForestRegressor  
from sklearn.model_selection import train_test_split  

df = pd.read_csv("mldeployment-cpe393-main\Housing.csv")


df["mainroad"] = df["mainroad"].map({"yes": 1, "no": 0})
df["guestroom"] = df["guestroom"].map({"yes": 1, "no": 0})
df["basement"] = df["basement"].map({"yes": 1, "no": 0})
df["hotwaterheating"] = df["hotwaterheating"].map({"yes": 1, "no": 0})
df["airconditioning"] = df["airconditioning"].map({"yes": 1, "no": 0})
df["prefarea"] = df["prefarea"].map({"yes": 1, "no": 0})


df = pd.get_dummies(df, columns=["furnishingstatus"], drop_first=True)


X = df.drop("price", axis=1)


y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()


model.fit(X_train, y_train)

with open("mldeployment-cpe393-main/app/model2.pkl", "wb") as f:
    pickle.dump(model, f)

# Confirmation message
print("Model trained and saved to app/model2.pkl")