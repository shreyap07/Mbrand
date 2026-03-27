
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,roc_curve
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(layout="wide")
st.title("💄 Customized Makeup Analytics Dashboard")

df = pd.read_csv("data.csv")

# Encode categorical
df_enc = df.copy()
le = LabelEncoder()
for col in df_enc.select_dtypes(include='object').columns:
    df_enc[col] = le.fit_transform(df_enc[col])

# Tabs
tab1,tab2,tab3,tab4,tab5 = st.tabs(["Overview","Diagnostics","Prediction","Clustering","Association"])

# TAB 1
with tab1:
    st.subheader("Dataset Overview")
    st.write(df.head())
    fig=px.histogram(df,x="Spending")
    st.plotly_chart(fig)

# TAB 2
with tab2:
    st.subheader("Correlation Matrix")
    st.write(df_enc.corr())

# TAB 3
with tab3:
    st.subheader("Classification Model")
    X=df_enc.drop("Purchase",axis=1)
    y=df_enc["Purchase"]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

    model=RandomForestClassifier()
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)

    st.write("Accuracy:",accuracy_score(y_test,y_pred))
    st.write("Precision:",precision_score(y_test,y_pred))
    st.write("Recall:",recall_score(y_test,y_pred))
    st.write("F1:",f1_score(y_test,y_pred))

    y_prob=model.predict_proba(X_test)[:,1]
    fpr,tpr,_=roc_curve(y_test,y_prob)
    st.line_chart({"FPR":fpr,"TPR":tpr})

    st.subheader("Feature Importance")
    imp=pd.DataFrame({"feature":X.columns,"importance":model.feature_importances_})
    st.bar_chart(imp.set_index("feature"))

    st.subheader("Regression (Spending)")
    reg=LinearRegression()
    reg.fit(X_train,y_train)
    st.write("Regression model trained")

    st.subheader("Predict New Customer")
    age=st.slider("Age",18,50)
    income=st.selectbox("Income",[20000,40000,80000,120000])
    usage=st.selectbox("Usage",[0,1,2])
    pain=st.selectbox("Pain",[0,1,2])
    product=st.selectbox("Product",[0,1,2,3,4])

    input_data=np.array([[age,income,usage,pain,product,1000]])
    pred=model.predict(input_data)
    st.write("Prediction:",pred)

# TAB 4
with tab4:
    st.subheader("Customer Segmentation")
    kmeans=KMeans(n_clusters=3)
    clusters=kmeans.fit_predict(X)
    df["Cluster"]=clusters
    st.write(df.head())

# TAB 5
with tab5:
    st.subheader("Association Rules")
    basket=pd.get_dummies(df["Product"])
    freq=apriori(basket,min_support=0.1,use_colnames=True)
    rules=association_rules(freq,metric="confidence",min_threshold=0.5)
    st.write(rules)
