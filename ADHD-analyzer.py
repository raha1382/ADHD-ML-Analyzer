import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


df=pd.read_excel('ADHD.xlsx')

df.drop("sex",axis=1,inplace=True)
df.drop("specify",axis=1,inplace=True)
df.drop("age",axis=1,inplace=True)
df.drop("nbt_completed",axis=1,inplace=True)
df.drop("nbt_year",axis=1,inplace=True)
df.drop("aas_change",axis=1,inplace=True)
df.drop("if_you_have_been_diagnosed_with_a_mental_illness_at_what_age_was_this",axis=1,inplace=True)
df.drop("home_language",axis=1,inplace=True)
df.drop("was_this_diagnosis_made_before_or_after_you_left_high_school",axis=1,inplace=True)
df.drop("if_you_have_ever_experienced_difficulties_and_or_symptoms_of_a_mental_illness_how_old_were_you_when_this_started",axis=1,inplace=True)
# ...
for i, row in df.iterrows():
    if(pd.isna(df.at[i,"asrs1_total.y"])):
        df.at[i,"asrs1_total.y"]=df.at[i,"bai1_total"]
# ...
for j in df.columns[df.columns.get_loc("aas1_item_1"):df.columns.get_loc("aas1_item_9")+1]:
    counter=[0 for i in range(6)]
    index=0
    for i, row in df.iterrows():
        for k in range(6):
            if(df.at[i,j]==k):
                counter[k]+=1

    for i in range(6):
        if(counter[i]==max(counter)):
            index=i
    for i, row in df.iterrows():
        if(pd.isna(df.at[i,j])):
            df.at[i,j]=index
# ...
for i, row in df.iterrows():
    if(df.at[i,"nbt_did_math"]=="yes"):
        df.at[i,"nbt_did_math"]=1
    if(df.at[i,"nbt_did_math"]=="no"):
        df.at[i,"nbt_did_math"]=0
    if(pd.isna(df.at[i,"nbt_did_math"])):
        if(df.at[i,"nbt_math"]!=0):
            df.at[i,"nbt_did_math"]=1
        elif(df.at[i,"nbt_math"]==0):
            df.at[i,"nbt_did_math"]=0
# ...
a="have_you_ever_used_prescribed_psychiatric_medication_for_a_mental_illness_or_symptoms_of_one"
b="are_you_currently_in_therapy_or_counselling_for_a_mental_illness_or_symptoms_of_one"
for j in df.columns[df.columns.get_loc(a):df.columns.get_loc(b)+1]:
    for i, row in df.iterrows():
        if(df.at[i,j]=="yes"):
            df.at[i,j]=1
        elif(df.at[i,j]=="no" or df.at[i,j]=="not applicable"):
            df.at[i,j]=0
# ...
a="have_you_ever_experienced_any_mental_health_difficulties_or_symptoms_before_starting_university_e_g_in_primary_or_high_school"
for i, row in df.iterrows():
    if(df.at[i,a]=="yes"):
        df.at[i,a]=1
    if(df.at[i,a]=="no"):
        df.at[i,a]=0
# ...
a="have_you_ever_been_diagnosed_with_a_mental_illness"
for i, row in df.iterrows():
    if(df.at[i,a][0]=="y"):
        df.at[i,a]=1
    else:
        df.at[i,a]=0
# ...
diagnoses = [
    "breakdowns",
    "anxiety",
    "panic",
    "depression",
    "adhd",
    "bipolar",
    "trauma",
    "suicidal",
    "insomnia",
    "ptsd",
    "ocd"
]

for diagnose in diagnoses:
    df[diagnose + "with out doctor"] = 0
    df[diagnose] = 0

for index, row in df.iterrows():
    for diagnose in diagnoses:
        a="if_yes_please_list_these_difficulties_and_or_symptoms"
        if type(row[a]) == str and (diagnose in row[a]):
            df.at[index, diagnose + "with out doctor"] = 1

df.drop("if_yes_please_list_these_difficulties_and_or_symptoms",axis=1,inplace=True)

for index, row in df.iterrows():
    for diagnose in diagnoses:
        a="if_you_have_been_diagnosed_formally_or_informally_please_list_the_diagnosis_diagnoses"
        if type(row[a]) == str and (diagnose in row[a]):
            df.at[index, diagnose] = 1

df.drop("if_you_have_been_diagnosed_formally_or_informally_please_list_the_diagnosis_diagnoses",axis=1,inplace=True)



# # ...
df["being ill"] = 0
for diagnose in diagnoses:
    for i ,row in df.iterrows():
        if df.at[i, diagnose]!=0 or  df.at[i, diagnose+"with out doctor"]!=0 :
            df.at[i,"being ill"] = 1
        
# ...

df_result=df
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_result)
# ...
clusters=2
kmeans = KMeans(n_clusters=clusters, random_state=42)

plt.figure(figsize=(8, 6))
colors = ['red', 'blue','orange','green']
df_pca = pd.DataFrame(df_pca, columns=['feature1', 'feature2'])
df_pca['cluster'] = kmeans.fit_predict(df_pca)
for i in range(clusters):
    clustered_data = df_pca[df_pca['cluster'] == i]
    plt.scatter(clustered_data['feature1'], clustered_data['feature2'], 
                color=colors[i], label=f'Cluster {i}')
    
score = silhouette_score(df_pca[['feature1', 'feature2']], df_pca['cluster'])
plt.title(f'Silhouette Score: {score}')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
centroids = kmeans.cluster_centers_

# centers
print("centers:")
print(centroids)
plt.show()

#...
a="are_you_currently_using_prescribed_psychiatric_medication_for_a_mental_illness_or_symptoms_of_one"
b="are_you_currently_in_therapy_or_counselling_for_a_mental_illness_or_symptoms_of_one"
c="have_you_ever_been_to_therapy_or_counselling_for_a_mental_illness_or_symptoms_of_one"
d="have_you_ever_used_prescribed_psychiatric_medication_for_a_mental_illness_or_symptoms_of_one"
e="have_you_ever_experienced_any_mental_health_difficulties_or_symptoms_before_starting_university_e_g_in_primary_or_high_school"
g="have_you_ever_been_diagnosed_with_a_mental_illness"
test="being ill"

# Combining all diagnoses and their corresponding "with out doctor" labels in the target variable
all_labels = diagnoses + [i + "with out doctor" for i in diagnoses] + [test]+[a,b,c,d,e,g]
y = df[test]
X = df.drop(all_labels, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
print(models)
# ...

# Perform cross-validation with knn because of the better result
print("\nusing KNN:")
X = X.to_numpy()
y = y.to_numpy()
X = np.ascontiguousarray(X)
y = np.ascontiguousarray(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))