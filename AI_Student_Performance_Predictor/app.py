import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Student Performance Predictor", page_icon="🎓", layout="wide")
df = pd.read_csv("data/student_performance.csv")
features=["attendance","previous_marks","internal_marks","assignment_score","study_hours"]
X=df[features].astype(float).to_numpy(); y=df.performance_level.to_numpy()
mn=X.min(0); mx=X.max(0); Xn=(X-mn)/(mx-mn+1e-9)

def predict(v,k=11):
    vn=(np.array(v,float)-mn)/(mx-mn+1e-9)
    d=np.sqrt(((Xn-vn)**2).sum(1)); idx=np.argsort(d)[:k]
    counts=pd.Series(y[idx]).value_counts()
    return counts.index[0], counts.iloc[0]/k*100

def advice(level):
    return {"Excellent":"Excellent performance. Maintain consistency and keep challenging yourself.",
            "Good":"Good progress. Focus on weak subjects and maintain regular study habits.",
            "Average":"Create a fixed study routine, improve assignments and revise before internals.",
            "Needs Improvement":"Priority improvement needed: attendance, daily study time and assignments."}[level]

st.title("🎓 AI-Powered Student Performance Predictor")
st.caption("Machine Learning project using a custom K-Nearest Neighbors (KNN) algorithm")
st.info("Prototype uses a synthetic dataset for demonstration only.")

c1,c2=st.columns(2)
with c1:
    st.subheader("Student Details")
    a=st.slider("Attendance (%)",40,100,75)
    p=st.slider("Previous Marks (%)",0,100,65)
    m=st.slider("Internal Marks (%)",0,100,65)
    ass=st.slider("Assignment Score (%)",0,100,70)
    h=st.slider("Study Hours / Day",0.0,10.0,3.0,0.5)
    go=st.button("🔮 Predict Performance",use_container_width=True)
with c2:
    st.subheader("Prediction")
    if go:
        level,conf=predict([a,p,m,ass,h])
        st.success(f"Predicted Level: {level}")
        st.metric("Model Confidence",f"{conf:.1f}%")
        st.write(advice(level))
        st.dataframe(pd.DataFrame({"Metric":["Attendance","Previous Marks","Internal Marks","Assignments","Study Hours"],
                                   "Value":[f"{a}%",f"{p}%",f"{m}%",f"{ass}%",f"{h} hrs/day"]}),hide_index=True,use_container_width=True)
    else: st.write("Enter details and click Predict Performance.")

st.divider()
st.subheader("📊 Dataset Overview")
q1,q2,q3,q4=st.columns(4)
q1.metric("Students",len(df)); q2.metric("Avg Attendance",f"{df.attendance.mean():.1f}%")
q3.metric("Avg Marks",f"{df.previous_marks.mean():.1f}%"); q4.metric("Avg Study Time",f"{df.study_hours.mean():.1f} hrs")
fig,ax=plt.subplots()
df.performance_level.value_counts().reindex(["Needs Improvement","Average","Good","Excellent"],fill_value=0).plot(kind="bar",ax=ax)
ax.set_xlabel("Performance Level"); ax.set_ylabel("Students"); ax.set_title("Performance Distribution")
st.pyplot(fig)
st.subheader("How the AI works")
st.markdown("1. Collect student features.  2. Normalize values.  3. Find the most similar students using KNN.  4. Predict the most common performance level.  5. Generate an improvement recommendation.")
