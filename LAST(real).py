# 라이브러리 추가
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ptitprince as pt  # RainCloud 플롯을 위한 라이브러리
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings('ignore')  # 경고 메시지 무시

# CSV 파일을 읽어오기
data = pd.read_csv('/Users/joon/Documents/GitHub/Smoke_Detection/data/1.smoke_detection_iot.csv')

# 데이터셋 크기와 간단한 정보 출력
print(f"Shape Of The Dataset : {data.shape}")
print(f"\nGlimpse Of The Dataset :")
data.head().style.set_properties(**{"background-color": "#5f0d11","color":"#ddab46","border": "1.5px #5f0d11"})
print(f"Informations Of The Dataset :\n")
print(data.info())

# 데이터 요약
print(f"Summary Of The Dataset :")
data.describe().style.set_properties(**{"background-color": "#5f0d11","color":"#ddab46","border": "1.5px #5f0d11"})

# 중복 데이터 확인 및 제거
dup = data[data.duplicated()].shape[0]
print(f"There is {dup} duplicate entry among {data.shape[0]} entries in this dataset.")
data.drop_duplicates(keep='first', inplace=True)
print(f"After removing duplicate entries there are {data.shape[0]} entries in this dataset.")

# 결측값 확인
print(f"Null values of the Dataset :")
data.isna().sum().to_frame().T.style.set_properties(**{"background-color": "#5f0d11","color":"#ddab46","border": "1.5px #5f0d11"})

# 데이터 전처리
data["Fire Alarm"].replace({0:"No", 1:"Yes"}, inplace = True)  # 0과 1을 Yes와 No로 변환
data.rename(columns={"Temperature[C]": "Temperature","Humidity[%]": "Humidity","TVOC[ppb]":"TVOC","eCO2[ppm]":"eCO2","Pressure[hPa]":"Pressure"}, inplace = True)  # 열 이름 변경
# 사용하지 않는 열 삭제
data = data[["Temperature","Humidity","TVOC","eCO2","Raw H2","Raw Ethanol","Pressure","PM1.0","PM2.5","NC0.5","NC1.0","NC2.5","Fire Alarm"]]

# 전처리 후 데이터셋 확인
print("After preprocessing, let's have a glimpse of the final dataset :")
data.head().style.set_properties(**{"background-color": "#5f0d11","color":"#ddab46","border": "1.5px #5f0d11"})
print(f"After preprocessing, let's have a look on the summary of the dataset :")
data.describe().T.style.set_properties(**{"background-color": "#5f0d11","color":"#ddab46","border": "1.5px #5f0d11"})

# 시각화를 위한 사용자 지정 색상 팔레트 설정
sns.set_style("white")
sns.set(rc={"axes.facecolor":"#e9bb93", "figure.facecolor":"#e9bb93",
            "axes.grid":True, "grid.color":"white", "axes.edgecolor":"black",
            "grid.linestyle": u"-", "axes.labelcolor": "black", "font.family": [u"DejaVu Sans"],
            "text.color": "black", "xtick.color": "black", "ytick.color": "black",
            "legend.facecolor":"#e9bb93", "legend.frameon": True, "legend.edgecolor":"black"})
sns.set_context("poster",font_scale = .7)

# 사용자 정의 팔레트와 컬러맵 정의
palette = ["#272716","#6c2411","#ae0d13","#cc3f18","#db6400"]
palette_cmap = ["#272716","#6c2411","#cc3f18","#db6400","#ae0d13"]

# 온도 분포 시각화
print(f"Let's have a look on the distribution of temperature :")
_, axs = plt.subplot_mosaic([["00", "00"], ["10", "11"]], figsize=(20,12), sharey=True)
plt.tight_layout(pad=5.0)

# 전체 온도 분포
sns.histplot(data=data, x="Temperature", hue="Fire Alarm", hue_order=data["Fire Alarm"].value_counts(ascending=True).index,
             multiple="stack", ax=axs["00"], palette=palette[0:3:2], kde=True, bins=45, alpha=1, fill=True, edgecolor="black", linewidth=2)
axs["00"].lines[0].set_color("orange")
axs["00"].lines[1].set_color("orange")
axs["00"].set_title("\nOverall Temperature", fontsize=25)
axs["00"].set_ylabel("Count", fontsize=20)
axs["00"].set_xlabel("Temperature", fontsize=20)
axs["00"].set_yscale("linear")

# 화재 경보 발생 시 온도 분포
sns.histplot(data=data[data["Fire Alarm"]=="Yes"], x="Temperature", ax=axs["10"], color=palette[2], kde=True, bins=30,
             alpha=1, fill=True, edgecolor="black", linewidth=2)
axs["10"].lines[0].set_color("orange")
axs["10"].set_title("\nFor Triggering Fire Alarm", fontsize=25)
axs["10"].set_ylabel("Count", fontsize=20)
axs["10"].set_xlabel("Temperature", fontsize=20)
axs["10"].set_yscale("linear")

# 화재 경보가 발생하지 않을 때 온도 분포
sns.histplot(data=data[data["Fire Alarm"]=="No"], x="Temperature", ax=axs["11"], color=palette[0], kde=True, bins=30,
             alpha=1, fill=True, edgecolor="black", linewidth=2)
axs["11"].lines[0].set_color("orange")
axs["11"].set_title("\nFor Not Triggering Fire Alarm", fontsize=25)
axs["11"].set_ylabel("Count", fontsize=20)
axs["11"].set_xlabel("Temperature", fontsize=20)
axs["11"].set_yscale("linear")

plt.show()

# 화재 경보에 대한 온도 효과 분석
print("Let's have a look on the distribution of effectiveness-wise temperature analysis :")
plt.subplots(figsize=(20, 8))

p=pt.RainCloud(data=data, x=data["Fire Alarm"], y=data["Temperature"],
               order=data["Fire Alarm"].value_counts(ascending=True).index, pointplot=True, linecolor="orange",
               point_size=2, palette=palette[0:3:2], saturation=1, linewidth=3, edgecolor="black")
p.axes.set_title("\nEffectiveness Towards Fire Alarm\n", fontsize=25)
p.axes.set_xlabel("Effectiveness", fontsize=20)
p.axes.set_ylabel("Temperature", fontsize=20)

plt.show()

# 습도 분포 시각화
print(f"Let's have a look on the distribution of humidity :")
_, axs = plt.subplot_mosaic([["00", "00"], ["10", "11"]], figsize=(20,12), sharey=False)
plt.tight_layout(pad=5.0)

# 전체 습도 분포
sns.histplot(data=data, x="Humidity", hue="Fire Alarm", hue_order=data["Fire Alarm"].value_counts(ascending=True).index,
             multiple="stack", ax=axs["00"], palette=palette[0:3:2], kde=True, bins=45, alpha=1, fill=True, edgecolor="black", linewidth=2)
axs["00"].lines[0].set_color("orange")
axs["00"].lines[1].set_color("orange")
axs["00"].set_title("\nOverall Humidity", fontsize=25)
axs["00"].set_ylabel("Count", fontsize=20)
axs["00"].set_xlabel("Humidity", fontsize=20)
axs["00"].set_yscale("linear")

# 화재 경보 발생 시 습도 분포
sns.histplot(data=data[data["Fire Alarm"]=="Yes"], x="Humidity", ax=axs["10"], color=palette[2], kde=True, bins=30,
             alpha=1, fill=True, edgecolor="black", linewidth=2)
axs["10"].lines[0].set_color("orange")
axs["10"].set_title("\nFor Triggering Fire Alarm", fontsize=25)
axs["10"].set_ylabel("Count", fontsize=20)
axs["10"].set_xlabel("Humidity", fontsize=20)
axs["10"].set_yscale("linear")

# 화재 경보가 발생하지 않을 때 습도 분포
sns.histplot(data=data[data["Fire Alarm"]=="No"], x="Humidity", ax=axs["11"], color=palette[0], kde=True, bins=30,
             alpha=1, fill=True, edgecolor="black", linewidth=2)
axs["11"].lines[0].set_color("orange")
axs["11"].set_title("\nFor Not Triggering Fire Alarm", fontsize=25)
axs["11"].set_ylabel("Count", fontsize=20)
axs["11"].set_xlabel("Humidity", fontsize=20)
axs["11"].set_yscale("linear")

plt.show()

# 화재 경보에 대한 습도 효과 분석
print("Let's have a look on the distribution of effectiveness-wise humidity analysis :")
plt.subplots(figsize=(20, 8))

p=pt.RainCloud(data=data, x=data["Fire Alarm"], y=data["Humidity"],
               order=data["Fire Alarm"].value_counts(ascending=True).index, pointplot=True, linecolor="orange",
               point_size=2, palette=palette[0:3:2], saturation=1, linewidth=3, edgecolor="black")
p.axes.set_title("\nEffectiveness Towards Fire Alarm\n", fontsize=25)
p.axes.set_xlabel("Effectiveness", fontsize=20)
p.axes.set_ylabel("Humidity", fontsize=20)

plt.show()

# TVOC 분포 시각화
print(f"Let's have a look on the distribution of total volatile organic compounds :")
_, axs = plt.subplot_mosaic([["00", "00"], ["10", "11"]], figsize=(20,12), sharey=False)
plt.tight_layout(pad=5.0)

# 전체 TVOC 분포
sns.histplot(data=data, x="TVOC", hue="Fire Alarm", hue_order=data["Fire Alarm"].value_counts(ascending=True).index,
             multiple="stack", ax=axs["00"], palette=palette[0:3:2], kde=False, bins=45, alpha=1, fill=True, edgecolor="black", linewidth=2)
axs["00"].set_title("\nOverall Total Volatile Organic Compounds", fontsize=25)
axs["00"].set_ylabel("Count", fontsize=20)
axs["00"].set_xlabel("TVOC", fontsize=20)
axs["00"].set_yscale("log")

# 화재 경보 발생 시 TVOC 분포
sns.histplot(data=data[data["Fire Alarm"]=="Yes"], x="TVOC", ax=axs["10"], color=palette[2], kde=False, bins=30,
             alpha=1, fill=True, edgecolor="black", linewidth=2)
axs["10"].set_title("\nFor Triggering Fire Alarm", fontsize=25)
axs["10"].set_ylabel("Count", fontsize=20)
axs["10"].set_xlabel("TVOC", fontsize=20)
axs["10"].set_yscale("log")

# 화재 경보가 발생하지 않을 때 TVOC 분포
sns.histplot(data=data[data["Fire Alarm"]=="No"], x="TVOC", ax=axs["11"], color=palette[0], kde=False, bins=30,
             alpha=1, fill=True, edgecolor="black", linewidth=2)
axs["11"].set_title("\nFor Not Triggering Fire Alarm", fontsize=25)
axs["11"].set_ylabel("Count", fontsize=20)
axs["11"].set_xlabel("TVOC", fontsize=20)
axs["11"].set_yscale("log")

plt.show()

# 화재 경보에 대한 TVOC 효과 분석
print("Let's have a look on the distribution of effectiveness-wise total volatile organic compounds analysis :")
plt.subplots(figsize=(20, 8))

p=pt.RainCloud(data=data, x=data["Fire Alarm"], y=data["TVOC"],
               order=data["Fire Alarm"].value_counts(ascending=True).index, pointplot=True, linecolor="orange",
               point_size=2, palette=palette[0:3:2], saturation=1, linewidth=3, edgecolor="black")
p.axes.set_title("\nEffectiveness Towards Fire Alarm\n", fontsize=25)
p.axes.set_xlabel("Effectiveness", fontsize=20)
p.axes.set_ylabel("Total Volatile Organic Compounds", fontsize=20)
p.axes.set_yscale("log")

plt.show()

# CO2 등가 농도 분포 시각화
print(f"Let's have a look on the distribution of CO2 equivalent concentration :")
_, axs = plt.subplot_mosaic([["00", "00"], ["10", "11"]], figsize=(20,12), sharey=False)
plt.tight_layout(pad=5.0)

# 전체 CO2 등가 농도 분포
sns.histplot(data=data, x="eCO2", hue="Fire Alarm", hue_order=data["Fire Alarm"].value_counts(ascending=True).index,
             multiple="stack", ax=axs["00"], palette=palette[0:3:2], kde=False, bins=45, alpha=1, fill=True, edgecolor="black", linewidth=2)
axs["00"].set_title("\nOverall CO2 Equivalent Concentration", fontsize=25)
axs["00"].set_ylabel("Count", fontsize=20)
axs["00"].set_xlabel("CO2 Equivalent Concentration", fontsize=20)
axs["00"].set_yscale("log")

# 화재 경보 발생 시 CO2 등가 농도 분포
sns.histplot(data=data[data["Fire Alarm"]=="Yes"], x="eCO2", ax=axs["10"], color=palette[2], kde=False, bins=30,
             alpha=1, fill=True, edgecolor="black", linewidth=2)
axs["10"].set_title("\nFor Triggering Fire Alarm", fontsize=25)
axs["10"].set_ylabel("Count", fontsize=20)
axs["10"].set_xlabel("CO2 Equivalent Concentration", fontsize=20)
axs["10"].set_yscale("log")

# 화재 경보가 발생하지 않을 때 CO2 등가 농도 분포
sns.histplot(data=data[data["Fire Alarm"]=="No"], x="eCO2", ax=axs["11"], color=palette[0], kde=False, bins=30,
             alpha=1, fill=True, edgecolor="black", linewidth=2)
axs["11"].set_title("\nFor Not Triggering Fire Alarm", fontsize=25)
axs["11"].set_ylabel("Count", fontsize=20)
axs["11"].set_xlabel("CO2 Equivalent Concentration", fontsize=20)
axs["11"].set_yscale("log")

plt.show()

# 화재 경보에 대한 CO2 등가 농도 효과 분석
print("Let's have a look on the distribution of effectiveness-wise CO2 equivalent concentration analysis :")
plt.subplots(figsize=(20, 8))

p=pt.RainCloud(data=data, x=data["Fire Alarm"], y=data["eCO2"],
               order=data["Fire Alarm"].value_counts(ascending=True).index, pointplot=True, linecolor="orange",
               point_size=2, palette=palette[0:3:2], saturation=1, linewidth=3, edgecolor="black")
p.axes.set_title("\nEffectiveness Towards Fire Alarm\n", fontsize=25)
p.axes.set_xlabel("Effectiveness", fontsize=20)
p.axes.set_ylabel("CO2 Equivalent Concentration", fontsize=20)
p.axes.set_yscale("log")

plt.show()

# 원시 수소 분포 시각화
print(f"Let's have a look on the distribution of raw hydrogen :")
_, axs = plt.subplot_mosaic([["00", "00"], ["10", "11"]], figsize=(20,12), sharey=True)
plt.tight_layout(pad=5.0)

# 전체 원시 수소 분포
sns.histplot(data=data, x="Raw H2", hue="Fire Alarm", hue_order=data["Fire Alarm"].value_counts(ascending=True).index,
             multiple="stack", ax=axs["00"], palette=palette[0:3:2], kde=False, bins=45, alpha=1, fill=True, edgecolor="black", linewidth=2)
axs["00"].set_title("\nOverall Raw Hydrogen Existance", fontsize=25)
axs["00"].set_ylabel("Count", fontsize=20)
axs["00"].set_xlabel("Raw Hydrogen", fontsize=20)
axs["00"].set_yscale("log")

# 화재 경보 발생 시 원시 수소 분포
sns.histplot(data=data[data["Fire Alarm"]=="Yes"], x="Raw H2", ax=axs["10"], color=palette[2], kde=False, bins=30,
             alpha=1, fill=True, edgecolor="black", linewidth=2)
axs["10"].set_title("\nFor Triggering Fire Alarm", fontsize=25)
axs["10"].set_ylabel("Count", fontsize=20)
axs["10"].set_xlabel("Raw Hydrogen", fontsize=20)
axs["10"].set_yscale("log")

# 화재 경보가 발생하지 않을 때 원시 수소 분포
sns.histplot(data=data[data["Fire Alarm"]=="No"], x="Raw H2", ax=axs["11"], color=palette[0], kde=False, bins=30,
             alpha=1, fill=True, edgecolor="black", linewidth=2)
axs["11"].set_title("\nFor Not Triggering Fire Alarm", fontsize=25)
axs["11"].set_ylabel("Count", fontsize=20)
axs["11"].set_xlabel("Raw Hydrogen", fontsize=20)
axs["11"].set_yscale("log")

plt.show()

# 화재 경보에 대한 원시 수소 효과 분석
print("Let's have a look on the distribution of effectiveness-wise raw hydrogen existance analysis :")
plt.subplots(figsize=(20, 8))

p=pt.RainCloud(data=data, x=data["Fire Alarm"], y=data["Raw H2"],
               order=data["Fire Alarm"].value_counts(ascending=True).index, pointplot=True, linecolor="orange",
               point_size=2, palette=palette[0:3:2], saturation=1, linewidth=3, edgecolor="black")
p.axes.set_title("\nEffectiveness Towards Fire Alarm\n", fontsize=25)
p.axes.set_xlabel("Effectiveness", fontsize=20)
p.axes.set_ylabel("Raw Hydrogen", fontsize=20)
p.axes.set_yscale("linear")

plt.show()

# 원시 에탄올 분포 시각화
print(f"Let's have a look on the distribution of raw ethanol :")
_, axs = plt.subplot_mosaic([["00", "00"], ["10", "11"]], figsize=(20,12), sharey=True)
plt.tight_layout(pad=5.0)

# 전체 원시 에탄올 분포
sns.histplot(data=data, x="Raw Ethanol", hue="Fire Alarm", hue_order=data["Fire Alarm"].value_counts(ascending=True).index,
             multiple="stack", ax=axs["00"], palette=palette[0:3:2], kde=False, bins=45, alpha=1, fill=True, edgecolor="black", linewidth=2)
axs["00"].set_title("\nOverall Raw Ethanol Existance", fontsize=25)
axs["00"].set_ylabel("Count", fontsize=20)
axs["00"].set_xlabel("Raw Ethanol", fontsize=20)
axs["00"].set_yscale("log")

# 화재 경보 발생 시 원시 에탄올 분포
sns.histplot(data=data[data["Fire Alarm"]=="Yes"], x="Raw Ethanol", ax=axs["10"], color=palette[2], kde=False, bins=30,
             alpha=1, fill=True, edgecolor="black", linewidth=2)
axs["10"].set_title("\nFor Triggering Fire Alarm", fontsize=25)
axs["10"].set_ylabel("Count", fontsize=20)
axs["10"].set_xlabel("Raw Ethanol", fontsize=20)
axs["10"].set_yscale("log")

# 화재 경보가 발생하지 않을 때 원시 에탄올 분포
sns.histplot(data=data[data["Fire Alarm"]=="No"], x="Raw Ethanol", ax=axs["11"], color=palette[0], kde=False, bins=30,
             alpha=1, fill=True, edgecolor="black", linewidth=2)
axs["11"].set_title("\nFor Not Triggering Fire Alarm", fontsize=25)
axs["11"].set_ylabel("Count", fontsize=20)
axs["11"].set_xlabel("Raw Ethanol", fontsize=20)
axs["11"].set_yscale("log")

plt.show()

# 화재 경보에 대한 원시 에탄올 효과 분석
print("Let's have a look on the distribution of effectiveness-wise raw ethanol existance analysis :")
plt.subplots(figsize=(20, 8))

p=pt.RainCloud(data=data, x=data["Fire Alarm"], y=data["Raw Ethanol"],
               order=data["Fire Alarm"].value_counts(ascending=True).index, pointplot=True, linecolor="orange",
               point_size=2, palette=palette[0:3:2], saturation=1, linewidth=3, edgecolor="black")
p.axes.set_title("\nEffectiveness Towards Fire Alarm\n", fontsize=25)
p.axes.set_xlabel("Effectiveness", fontsize=20)
p.axes.set_ylabel("Raw Ethanol", fontsize=20)
p.axes.set_yscale("linear")

plt.show()

# 공기압 분포 시각화
print(f"Let's have a look on the distribution of air pressure :")
_, axs = plt.subplot_mosaic([["00", "00"], ["10", "11"]], figsize=(20,12), sharey=True)
plt.tight_layout(pad=5.0)

# 전체 공기압 분포
sns.histplot(data=data, x="Pressure", hue="Fire Alarm", hue_order=data["Fire Alarm"].value_counts(ascending=True).index,
             multiple="stack", ax=axs["00"], palette=palette[0:3:2], kde=False, bins=45, alpha=1, fill=True, edgecolor="black", linewidth=2)
axs["00"].set_title("\nOverall Air Pressure", fontsize=25)
axs["00"].set_ylabel("Count", fontsize=20)
axs["00"].set_xlabel("Pressure", fontsize=20)
axs["00"].set_yscale("linear")

# 화재 경보 발생 시 공기압 분포
sns.histplot(data=data[data["Fire Alarm"]=="Yes"], x="Pressure", ax=axs["10"], color=palette[2], kde=False, bins=30,
             alpha=1, fill=True, edgecolor="black", linewidth=2)
axs["10"].set_title("\nFor Triggering Fire Alarm", fontsize=25)
axs["10"].set_ylabel("Count", fontsize=20)
axs["10"].set_xlabel("Pressure", fontsize=20)
axs["10"].set_yscale("linear")

# 화재 경보가 발생하지 않을 때 공기압 분포
sns.histplot(data=data[data["Fire Alarm"]=="No"], x="Pressure", ax=axs["11"], color=palette[0], kde=False, bins=30,
             alpha=1, fill=True, edgecolor="black", linewidth=2)
axs["11"].set_title("\nFor Not Triggering Fire Alarm", fontsize=25)
axs["11"].set_ylabel("Count", fontsize=20)
axs["11"].set_xlabel("Pressure", fontsize=20)
axs["11"].set_yscale("log")

plt.show()

# 화재 경보에 대한 공기압 효과 분석
print("Let's have a look on the distribution of effectiveness-wise air pressure :")
plt.subplots(figsize=(20, 8))

p=pt.RainCloud(data=data, x=data["Fire Alarm"], y=data["Pressure"],
               order=data["Fire Alarm"].value_counts(ascending=True).index, pointplot=True, linecolor="orange",
               point_size=2, palette=palette[0:3:2], saturation=1, linewidth=3, edgecolor="black")
p.axes.set_title("\nEffectiveness Towards Fire Alarm\n", fontsize=25)
p.axes.set_xlabel("Effectiveness", fontsize=20)
p.axes.set_ylabel("Pressure", fontsize=20)
p.axes.set_yscale("linear")

plt.show()

# 미세먼지(PM1.0, PM2.5) 분포 시각화
print(f"Let's have a look on the distribution of particulate matter values :")
_, axs = plt.subplots(2, 1, figsize=(20,12), sharey=True)
plt.tight_layout(pad=5.0)

# PM1.0 분포
sns.histplot(data=data, x="PM1.0", hue="Fire Alarm", hue_order=data["Fire Alarm"].value_counts(ascending=True).index,
             multiple="stack", ax=axs[0], palette=palette[0:3:2], kde=False, bins=45, alpha=1, fill=True, edgecolor="black", linewidth=2)
axs[0].set_title("\nOverall PM1.0 Values", fontsize=25)
axs[0].set_ylabel("Count", fontsize=20)
axs[0].set_xlabel("PM1.0 Values", fontsize=20)
axs[0].set_yscale("linear")

# PM2.5 분포
sns.histplot(data=data, x="PM2.5", hue="Fire Alarm", hue_order=data["Fire Alarm"].value_counts(ascending=True).index,
             multiple="stack", ax=axs[1], palette=palette[0:3:2], kde=False, bins=45, alpha=1, fill=True, edgecolor="black", linewidth=2)
axs[1].set_title("\nOverall PM2.5 Values", fontsize=25)
axs[1].set_ylabel("Count", fontsize=20)
axs[1].set_xlabel("PM2.5 Values", fontsize=20)
axs[1].set_yscale("log")

plt.show()

# 미세먼지 농도 분포 시각화
print(f"Let's have a look on the distribution of the concentration of particulate matter values :")
_, axs = plt.subplots(3, 1, figsize=(20,16), sharey=True)
plt.tight_layout(pad=5.0)

# NC0.5 분포
sns.histplot(data=data, x="NC0.5", hue="Fire Alarm", hue_order=data["Fire Alarm"].value_counts(ascending=True).index,
             multiple="stack", ax=axs[0], palette=palette[0:3:2], kde=False, bins=45, alpha=1, fill=True, edgecolor="black", linewidth=2)
axs[0].set_title("\nOverall NC0.5 Values", fontsize=25)
axs[0].set_ylabel("Count", fontsize=20)
axs[0].set_xlabel("NC0.5 Values", fontsize=20)
axs[0].set_yscale("linear")

# NC1.0 분포
sns.histplot(data=data, x="NC1.0", hue="Fire Alarm", hue_order=data["Fire Alarm"].value_counts(ascending=True).index,
             multiple="stack", ax=axs[1], palette=palette[0:3:2], kde=False, bins=45, alpha=1, fill=True, edgecolor="black", linewidth=2)
axs[1].set_title("\nOverall NC1.0 Values", fontsize=25)
axs[1].set_ylabel("Count", fontsize=20)
axs[1].set_xlabel("NC1.0 Values", fontsize=20)
axs[1].set_yscale("log")

# NC2.5 분포
sns.histplot(data=data, x="NC2.5", hue="Fire Alarm", hue_order=data["Fire Alarm"].value_counts(ascending=True).index,
             multiple="stack", ax=axs[2], palette=palette[0:3:2], kde=False, bins=45, alpha=1, fill=True, edgecolor="black", linewidth=2)
axs[2].set_title("\nOverall NC2.5 Values", fontsize=25)
axs[2].set_ylabel("Count", fontsize=20)
axs[2].set_xlabel("NC2.5 Values", fontsize=20)
axs[2].set_yscale("log")

plt.show()

# 상관 관계 맵 생성
catcol = [col for col in data.columns if data[col].dtype == "object"]  # 객체 데이터 유형의 열 선택
le = LabelEncoder()  # 레이블 인코더 초기화
for col in catcol:
    data[col] = le.fit_transform(data[col])  # 객체 데이터를 숫자로 변환

plt.subplots(figsize =(12, 12))
sns.heatmap(data.corr(), cmap = palette_cmap, square=True, cbar_kws=dict(shrink =.82),
            annot=True, vmin=-1, vmax=1, linewidths=3,linecolor='#e0b583',annot_kws=dict(fontsize =12))
plt.title("Pearson Correlation Of Features\n", fontsize=25)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# 모델 생성 및 성능 평가

x = data.drop(["Fire Alarm"], axis=1)  # 독립 변수 X 설정
y = data["Fire Alarm"]  # 종속 변수 Y 설정

sc = StandardScaler()  # 스케일러 초기화
x = sc.fit_transform(x)  # 데이터 스케일링
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)  # 데이터 분할

print(f"Shape of training data : {x_train.shape}, {y_train.shape}")
print(f"Shape of testing data : {x_test.shape}, {y_test.shape}")

# 로지스틱 회귀 모델
lr = LogisticRegression()
lr.fit(x_train, y_train)  # 모델 학습
lr_pred = lr.predict(x_test)  # 테스트 데이터 예측
lr_conf = confusion_matrix(y_test, lr_pred)  # 혼동 행렬
lr_report = classification_report(y_test, lr_pred)  # 분류 보고서
lr_acc = round(accuracy_score(y_test, lr_pred)*100, ndigits=2)  # 정확도 계산
print(f"Confusion Matrix : \n\n{lr_conf}")
print(f"\nClassification Report : \n\n{lr_report}")
print(f"\nThe Accuracy of Logistic Regression is {lr_acc} %")

# 가우시안 나이브 베이즈
gnb = GaussianNB()
gnb.fit(x_train, y_train)  # 모델 학습
gnb_pred = gnb.predict(x_test)  # 테스트 데이터 예측
gnb_conf = confusion_matrix(y_test, gnb_pred)  # 혼동 행렬
gnb_report = classification_report(y_test, gnb_pred)  # 분류 보고서
gnb_acc = round(accuracy_score(y_test, gnb_pred)*100, ndigits=2)  # 정확도 계산
print(f"Confusion Matrix : \n\n{gnb_conf}")
print(f"\nClassification Report : \n\n{gnb_report}")
print(f"\nThe Accuracy of Gaussian Naive Bayes is {gnb_acc} %")

# 버누리 나이브 베이즈
bnb = BernoulliNB()
bnb.fit(x_train, y_train)  # 모델 학습
bnb_pred = bnb.predict(x_test)  # 테스트 데이터 예측
bnb_conf = confusion_matrix(y_test, bnb_pred)  # 혼동 행렬
bnb_report = classification_report(y_test, bnb_pred)  # 분류 보고서
bnb_acc = round(accuracy_score(y_test, bnb_pred)*100, ndigits=2)  # 정확도 계산
print(f"Confusion Matrix : \n\n{bnb_conf}")
print(f"\nClassification Report : \n\n{bnb_report}")
print(f"\nThe Accuracy of Bernoulli Naive Bayes is {bnb_acc} %")

# 서포트 벡터 머신
svm = SVC(C=100, gamma=0.002)
svm.fit(x_train, y_train)  # 모델 학습
svm_pred = svm.predict(x_test)  # 테스트 데이터 예측
svm_conf = confusion_matrix(y_test, svm_pred)  # 혼동 행렬
svm_report = classification_report(y_test, svm_pred)  # 분류 보고서
svm_acc = round(accuracy_score(y_test, svm_pred)*100, ndigits=2)  # 정확도 계산
print(f"Confusion Matrix : \n\n{svm_conf}")
print(f"\nClassification Report : \n\n{svm_report}")
print(f"\nThe Accuracy of Support Vector Machine is {svm_acc} %")

# 랜덤 포레스트
rfg = RandomForestClassifier(n_estimators=100, random_state=42)
rfg.fit(x_train, y_train)  # 모델 학습
rfg_pred = rfg.predict(x_test)  # 테스트 데이터 예측
rfg_conf = confusion_matrix(y_test, rfg_pred)  # 혼동 행렬
rfg_report = classification_report(y_test, rfg_pred)  # 분류 보고서
rfg_acc = round(accuracy_score(y_test, rfg_pred)*100, ndigits=2)  # 정확도 계산
print(f"Confusion Matrix : \n\n{rfg_conf}")
print(f"\nClassification Report : \n\n{rfg_report}")
print(f"\nThe Accuracy of Random Forest Classifier is {rfg_acc} %")

# K 최근접 이웃
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train, y_train)  # 모델 학습
knn_pred = knn.predict(x_test)  # 테스트 데이터 예측
knn_conf = confusion_matrix(y_test, knn_pred)  # 혼동 행렬
knn_report = classification_report(y_test, knn_pred)  # 분류 보고서
knn_acc = round(accuracy_score(y_test, knn_pred)*100, ndigits=2)  # 정확도 계산
print(f"Confusion Matrix : \n\n{knn_conf}")
print(f"\nClassification Report : \n\n{knn_report}")
print(f"\nThe Accuracy of K Nearest Neighbors Classifier is {knn_acc} %")

# 극단적인 그라디언트 부스팅
xgb = XGBClassifier(use_label_encoder=False)
xgb.fit(x_train, y_train)  # 모델 학습
xgb_pred = xgb.predict(x_test)  # 테스트 데이터 예측
xgb_conf = confusion_matrix(y_test, xgb_pred)  # 혼동 행렬
xgb_report = classification_report(y_test, xgb_pred)  # 분류 보고서
xgb_acc = round(accuracy_score(y_test, xgb_pred)*100, ndigits=2)  # 정확도 계산
print(f"Confusion Matrix : \n\n{xgb_conf}")
print(f"\nClassification Report : \n\n{xgb_report}")
print(f"\nThe Accuracy of Extreme Gradient Boosting Classifier is {xgb_acc} %")

# 신경망 아키텍처
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam

regularization_parameter = 0.003  # 정규화 파라미터

# 신경망 모델 구성
neural_model = Sequential([
    tf.keras.layers.Dense(units=32, input_dim=(x_train.shape[-1]), activation="relu", kernel_regularizer=regularizers.l1(regularization_parameter)),
    tf.keras.layers.Dense(units=64, activation="relu", kernel_regularizer=regularizers.l1(regularization_parameter)),
    tf.keras.layers.Dense(units=128, activation="relu", kernel_regularizer=regularizers.l1(regularization_parameter)),
    tf.keras.layers.Dropout(0.3),  # 드롭아웃으로 과적합 방지
    tf.keras.layers.Dense(units=16, activation="relu", kernel_regularizer=regularizers.l1(regularization_parameter)),
    tf.keras.layers.Dense(units=1, activation="sigmoid")  # 이진 분류 출력을 위한 시그모이드 활성화 함수
])

print(neural_model.summary())  # 신경망 구조 요약 출력

# 정확도가 100%에 도달하면 학습을 중지하는 콜백 클래스
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get("accuracy") == 1.0):
            print("\nAccuracy is 100% so canceling training!")
            self.model.stop_training = True

callbacks = myCallback()  # 콜백 인스턴스 생성

# 모델 컴파일
neural_model.compile(optimizer=Adam(learning_rate=0.001),  # 최적화 알고리즘: Adam
                     loss="binary_crossentropy",  # 손실 함수: 이진 교차 엔트로피
                     metrics=["accuracy"])  # 평가지표: 정확도

# 모델 학습
history = neural_model.fit(x_train, y_train,
                           epochs=150,  # 에포크 수
                           verbose=1,  # 학습 과정 출력
                           batch_size=64,  # 배치 크기
                           validation_data=(x_test, y_test),  # 검증 데이터
                           callbacks=[callbacks])  # 콜백 설정

# 학습 성능 데이터프레임으로 정리
performance = pd.DataFrame(history.history)
performance["Epoch"] = range(1, len(history.history["accuracy"]) + 1)  # 에포크 번호 추가
performance.rename(columns={"loss": "Training Loss", "accuracy": "Training Accuracy", "val_loss": "Validation Loss", "val_accuracy": "Validation Accuracy"}, inplace=True)

# 신경망 아키텍처 성능 시각화
print(f"Let's have a look on the performance of neural network architecture :")
_, axs = plt.subplots(2, 1, figsize=(20,12), sharex=True)
plt.tight_layout(pad=5.0)

# 학습 및 검증 정확도 그래프
sns.lineplot(data=performance, x="Epoch", y="Training Accuracy", ax=axs[0], color=palette[0], alpha=1, linewidth=4)
sns.lineplot(data=performance, x="Epoch", y="Validation Accuracy", ax=axs[0], color=palette[2], alpha=1, linewidth=4)
axs[0].set_title("\nTraining And Validation Accuracy", fontsize=25)
axs[0].set_ylabel("Accuracy", fontsize=20)
axs[0].set_xlabel("Epoch", fontsize=20)
axs[0].legend(["Training Accuracy", "Validation Accuracy"], title="Accuracy")

# 학습 및 검증 손실 그래프
sns.lineplot(data=performance, x="Epoch", y="Training Loss", ax=axs[1], color=palette[0], alpha=1, linewidth=4)
sns.lineplot(data=performance, x="Epoch", y="Validation Loss", ax=axs[1], color=palette[2], alpha=1, linewidth=4)
axs[1].set_title("\nTraining And Validation Loss", fontsize=25)
axs[1].set_ylabel("Loss", fontsize=20)
axs[1].set_xlabel("Epoch", fontsize=20)
axs[1].legend(["Training Loss", "Validation Loss"], title="Loss")

plt.show()
