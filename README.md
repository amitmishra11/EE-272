---
jupyter:
  colab:
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.10.6
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .code execution_count="26" id="ZfPAyGgkL00o"}
``` python
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
```
:::

::: {.cell .code execution_count="27" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":73}" id="lCfN1jjjU54b" outputId="818d4b5f-5b60-4af4-821f-8a900d7ea1d9"}
``` python
# from google.colab import files
# uploaded=files.upload()
# import csv
de=pd.read_csv('class_data.csv')
```
:::

::: {.cell .code execution_count="28" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":73}" id="MK3KT5MnPEl4" outputId="da49025c-4f52-4fe1-f415-f12aab5c7359"}
``` python
# from google.colab import files
# uploaded=files.upload()
# import csv
df=pd.read_csv('detect_data.csv')
```
:::

::: {.cell .code execution_count="29" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":206}" id="NjhxS7jkVeK1" outputId="f0f05da5-584a-498f-e845-b80880490519"}
``` python
de.head()
```

::: {.output .execute_result execution_count="29"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>G</th>
      <th>Va</th>
      <th>Vb</th>
      <th>Vc</th>
      <th>Ia</th>
      <th>Ib</th>
      <th>Ic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1.154673</td>
      <td>6.408686</td>
      <td>-5.254013</td>
      <td>60.499877</td>
      <td>-21.265321</td>
      <td>-39.234556</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-3.740934</td>
      <td>6.821089</td>
      <td>-3.080155</td>
      <td>51.362040</td>
      <td>3.428051</td>
      <td>-54.790092</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-5.578969</td>
      <td>6.204178</td>
      <td>-0.625208</td>
      <td>35.426322</td>
      <td>25.698334</td>
      <td>-61.124656</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-6.531995</td>
      <td>4.998723</td>
      <td>1.533272</td>
      <td>17.975166</td>
      <td>41.839681</td>
      <td>-59.814847</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-6.416192</td>
      <td>1.177615</td>
      <td>5.238577</td>
      <td>-21.070529</td>
      <td>60.459488</td>
      <td>-39.388959</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .code execution_count="30" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":518}" id="vU0PMkWFVihy" outputId="ec5e68d5-e443-441e-ee81-c3485f852f4e"}
``` python
print(df['Output (S)'].value_counts(),"\n")
sns.countplot(x=df['Output (S)'])
plt.show()
```

::: {.output .stream .stdout}
    Output (S)
    0    1448
    1     589
    Name: count, dtype: int64 
:::

::: {.output .display_data}
![](vertopal_8066f867aa804212a930e5838110e1d4/023f688cda0b0a18c918f2216e924440e4d9878b.png)
:::
:::

::: {.cell .code execution_count="31" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":206}" id="v9b2FCk69XTY" outputId="8c8ed3bd-b692-4d7d-b3f8-d6c937248f97"}
``` python
de['fault_types'] = de['G'].astype('str') + de['C'].astype('str') + de['B'].astype('str') +de['A'].astype('str')
de.head()
```

::: {.output .execute_result execution_count="31"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>G</th>
      <th>Va</th>
      <th>Vb</th>
      <th>Vc</th>
      <th>Ia</th>
      <th>Ib</th>
      <th>Ic</th>
      <th>fault_types</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1.154673</td>
      <td>6.408686</td>
      <td>-5.254013</td>
      <td>60.499877</td>
      <td>-21.265321</td>
      <td>-39.234556</td>
      <td>0000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-3.740934</td>
      <td>6.821089</td>
      <td>-3.080155</td>
      <td>51.362040</td>
      <td>3.428051</td>
      <td>-54.790092</td>
      <td>0000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-5.578969</td>
      <td>6.204178</td>
      <td>-0.625208</td>
      <td>35.426322</td>
      <td>25.698334</td>
      <td>-61.124656</td>
      <td>0000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-6.531995</td>
      <td>4.998723</td>
      <td>1.533272</td>
      <td>17.975166</td>
      <td>41.839681</td>
      <td>-59.814847</td>
      <td>0000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-6.416192</td>
      <td>1.177615</td>
      <td>5.238577</td>
      <td>-21.070529</td>
      <td>60.459488</td>
      <td>-39.388959</td>
      <td>0000</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .code execution_count="32" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":206}" id="kwvRur6HPr2x" outputId="68065835-c187-493a-8afe-e5def769eb68"}
``` python
df.head()
```

::: {.output .execute_result execution_count="32"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Output (S)</th>
      <th>Va</th>
      <th>Vb</th>
      <th>Vc</th>
      <th>Ia</th>
      <th>Ib</th>
      <th>Ic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>-1.154673</td>
      <td>6.408686</td>
      <td>-5.254013</td>
      <td>60.499877</td>
      <td>-21.265321</td>
      <td>-39.234556</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>-3.740934</td>
      <td>6.821089</td>
      <td>-3.080155</td>
      <td>51.362040</td>
      <td>3.428051</td>
      <td>-54.790092</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>-5.578969</td>
      <td>6.204178</td>
      <td>-0.625208</td>
      <td>35.426322</td>
      <td>25.698334</td>
      <td>-61.124656</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>-6.531995</td>
      <td>4.998723</td>
      <td>1.533272</td>
      <td>17.975166</td>
      <td>41.839681</td>
      <td>-59.814847</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>-6.416192</td>
      <td>1.177615</td>
      <td>5.238577</td>
      <td>-21.070529</td>
      <td>60.459488</td>
      <td>-39.388959</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .code execution_count="33" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":583}" id="uu3ZCoV0WPf6" outputId="e55c2635-f3e4-421a-e0de-13513e68185f"}
``` python
print("[G C B A]\n[0 0 0 0] -> No fault \n[1 0 0 1] -> LG fault\n[0 1 1 0] -> LL fault\n[1 0 1 1] -> LLG Fault\n[0 1 1 1] -> LLL Fault\n[1 1 1 1] -> LLLG fault\n")
plt.figure(figsize=(8,5))
de.fault_types.value_counts().plot.pie()
plt.title("Type of Faults")
plt.ylabel("")
plt.show()
```

::: {.output .stream .stdout}
    [G C B A]
    [0 0 0 0] -> No fault 
    [1 0 0 1] -> LG fault
    [0 1 1 0] -> LL fault
    [1 0 1 1] -> LLG Fault
    [0 1 1 1] -> LLL Fault
    [1 1 1 1] -> LLLG fault
:::

::: {.output .display_data}
![](vertopal_8066f867aa804212a930e5838110e1d4/c16f98b605771c5059cde01a1d01e7c227bb5173.png)
:::
:::

::: {.cell .code execution_count="34" id="3kHJ7JtgW_Ek"}
``` python
NF = de[de['fault_types']=='0000']
LG = de[de['fault_types']=='1001'] 
LL = de[de['fault_types']=='0110'] 
LLG = de[de['fault_types']=='1011'] 
LLL = de[de['fault_types']=='0111'] 
LLLG = de[de['fault_types']=='1111']
```
:::

::: {.cell .code execution_count="35" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":483}" id="pPvJjm2lW2WJ" outputId="e6dc475e-e4bf-49e0-acce-e4c70398e0a6"}
``` python
print("For No Fault")
plt.subplots(1,3,figsize=(14,5))
plt.subplot(131)
sns.scatterplot(x=NF['Ia'],y=NF['Va'],color='darkred')

plt.subplot(132)
sns.scatterplot(x=NF['Ib'],y=NF['Vb'],color='green')

plt.subplot(133)
sns.scatterplot(x=NF['Ic'],y=NF['Vc'])

# plt.tight_layout()
plt.show()
```

::: {.output .stream .stdout}
    For No Fault
:::

::: {.output .display_data}
![](vertopal_8066f867aa804212a930e5838110e1d4/56aa196e03ab4d7a523e21fc235470c5c9cc8f01.png)
:::
:::

::: {.cell .code execution_count="36" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":416}" id="2gk4pzOtXiwo" outputId="7a0cb602-a05b-4acf-9545-4e33a19f9067"}
``` python
print("For Line to Line Fault")
plt.subplots(1,3,figsize=(15,5))

plt.subplot(131)
sns.scatterplot(x=LL['Ia'],y=LL['Va'],color='darkred')

plt.subplot(132)
sns.scatterplot(x=LL['Ib'],y=LL['Vb'],color='green')

plt.subplot(133)
sns.scatterplot(x=LL['Ic'],y=LL['Vc'])

plt.tight_layout()
plt.show()
```

::: {.output .stream .stdout}
    For Line to Line Fault
:::

::: {.output .display_data}
![](vertopal_8066f867aa804212a930e5838110e1d4/0b351d77dd3979fe53897da73dc25913cf52eeb0.png)
:::
:::

::: {.cell .code execution_count="37" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":416}" id="EkvFmTRUXlAg" outputId="850ecacc-2f8a-4245-9aa1-5875f68c102f"}
``` python
print("For Line Line Ground Fault")
plt.subplots(1,3,figsize=(15,5))

plt.subplot(131)
sns.scatterplot(x=LLG['Ia'],y=LLG['Va'],color='darkred')

plt.subplot(132)
sns.scatterplot(x=LLG['Ib'],y=LLG['Vb'],color='green')

plt.subplot(133)
sns.scatterplot(x=LLG['Ic'],y=LLG['Vc'])

plt.tight_layout()
plt.show()
```

::: {.output .stream .stdout}
    For Line Line Ground Fault
:::

::: {.output .display_data}
![](vertopal_8066f867aa804212a930e5838110e1d4/10a58ab786ecd3b9263fe8a13223616f47c2045f.png)
:::
:::

::: {.cell .code execution_count="38" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":416}" id="28zZ2uFeXqKa" outputId="21067025-6743-40c5-f8a3-176f8e666467"}
``` python
print("For Line Line Line Fault")
plt.subplots(1,3,figsize=(15,5))

plt.subplot(131)
sns.scatterplot(x=LLL['Ia'],y=LLL['Va'],color='darkred')

plt.subplot(132)
sns.scatterplot(x=LLL['Ib'],y=LLL['Vb'],color='green')

plt.subplot(133)
sns.scatterplot(x=LLL['Ic'],y=LLL['Vc'])

plt.tight_layout()
plt.show()
```

::: {.output .stream .stdout}
    For Line Line Line Fault
:::

::: {.output .display_data}
![](vertopal_8066f867aa804212a930e5838110e1d4/03a5e070a10362cb6d61723cdc3ff85f7fa5c92f.png)
:::
:::

::: {.cell .code execution_count="39" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":415}" id="-nB6jv-ZXy1Q" outputId="4bc31710-884e-43a6-8054-100aa9669682"}
``` python
print("For Line Line Line Ground Fault")
plt.subplots(1,3,figsize=(15,5))

plt.subplot(131)
sns.scatterplot(x=LLLG['Ia'],y=LLLG['Va'],color='darkred')

plt.subplot(132)
sns.scatterplot(x=LLLG['Ib'],y=LLLG['Vb'],color='green')

plt.subplot(133)
sns.scatterplot(x=LLLG['Ic'],y=LLLG['Vc'])

plt.tight_layout()
plt.show()
```

::: {.output .stream .stdout}
    For Line Line Line Ground Fault
:::

::: {.output .display_data}
![](vertopal_8066f867aa804212a930e5838110e1d4/7f7f155d4b10b4939efa2972f85edca75a6b7b76.png)
:::
:::

::: {.cell .code execution_count="40" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":1000}" id="XYeXKtv2X5oe" outputId="91136063-115b-4d01-89e9-dd4d020ac671"}
``` python
sns.pairplot(df,hue='Output (S)')
plt.show()
```

::: {.output .display_data}
![](vertopal_8066f867aa804212a930e5838110e1d4/16d3b1c176b5009eedd535710f865b39abe0e417.png)
:::
:::

::: {.cell .code execution_count="41" id="aozuGWQ6Ptwa"}
``` python
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
```
:::

::: {.cell .code execution_count="42" id="74HZbVveP814"}
``` python
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn import metrics
```
:::

::: {.cell .code execution_count="43" id="rwsuUbUiVxTa"}
``` python
from sklearn.metrics import ConfusionMatrixDisplay
```
:::

::: {.cell .code execution_count="44" id="_xQb2teLSObk"}
``` python
dip = df.drop(columns=['Output (S)'],axis=1)
dipc = dip.columns

mms = MinMaxScaler()
df_dip = mms.fit_transform(dip)

dip = pd.DataFrame(df_dip, columns=dipc)
```
:::

::: {.cell .code execution_count="45" id="SHzcYOiOSPVn"}
``` python
dop = df.iloc[:,0]
```
:::

::: {.cell .code execution_count="46" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":300}" id="-tjXTwe_SduF" outputId="d808f0e6-b700-4322-d894-41fc155d766b"}
``` python
dip.describe()
```

::: {.output .execute_result execution_count="46"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Va</th>
      <th>Vb</th>
      <th>Vc</th>
      <th>Ia</th>
      <th>Ib</th>
      <th>Ic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2037.000000</td>
      <td>2037.000000</td>
      <td>2037.000000</td>
      <td>2037.000000</td>
      <td>2037.000000</td>
      <td>2037.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.635744</td>
      <td>0.360359</td>
      <td>0.476131</td>
      <td>0.489066</td>
      <td>0.512243</td>
      <td>0.500563</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.179725</td>
      <td>0.200825</td>
      <td>0.306045</td>
      <td>0.369783</td>
      <td>0.363161</td>
      <td>0.279159</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.512994</td>
      <td>0.198243</td>
      <td>0.192724</td>
      <td>0.108082</td>
      <td>0.115263</td>
      <td>0.315755</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.660131</td>
      <td>0.363915</td>
      <td>0.433694</td>
      <td>0.438231</td>
      <td>0.542135</td>
      <td>0.496382</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.783571</td>
      <td>0.528717</td>
      <td>0.817711</td>
      <td>0.910837</td>
      <td>0.896058</td>
      <td>0.697991</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {.cell .code execution_count="47" id="J1GofUVHSgkn"}
``` python
X_train, X_test, y_train, y_test = train_test_split(dip, dop, test_size=0.25, random_state=67)
```
:::

::: {.cell .code execution_count="48" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="Dm1sarsySlxS" outputId="3294354b-fd8b-4946-d773-bf169028ff9f"}
``` python
sv = SVC(C=1000)

sv.fit(X_train,y_train)
scores = cross_val_score(sv, X_test, y_test, cv=10)
print("Score:", np.mean(scores))
# metrics.confusion_matrix(X_test,y_test)
# plt.show()
```

::: {.output .stream .stdout}
    Score: 0.9607843137254901
:::
:::

::: {.cell .code execution_count="49" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="9arNNE2ZTCBU" outputId="f3cd510c-d024-4036-e749-e89e9dec402f"}
``` python
dtc = DecisionTreeClassifier(criterion='gini',ccp_alpha=0.0012)

dtc.fit(X_train,y_train)
scores = cross_val_score(dtc, X_test, y_test, cv=10)
print("Score:", np.mean(scores),"\n")
# metrics.plot_confusion_matrix(dtc,X_test,y_test)
# plt.show()
```

::: {.output .stream .stdout}
    Score: 0.915686274509804 
:::
:::

::: {.cell .code execution_count="50" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":565}" id="gsGuNH6jZGy_" outputId="fefb6dfc-df79-4730-d37d-f41ca0582197"}
``` python
from sklearn.tree import plot_tree
plt.figure(figsize=(15,7))
plot_tree(dtc,filled=True,feature_names=['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc'])
plt.show()
```

::: {.output .display_data}
![](vertopal_8066f867aa804212a930e5838110e1d4/da85ddafa199b8d427d1ba5892702fb187b6ba99.png)
:::
:::
