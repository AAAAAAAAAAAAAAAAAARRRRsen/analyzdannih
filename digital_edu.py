import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('train.csv')
male = 0
female = 0

def male_female_langs(row):
    global male, female
    if row['sex'] == 2 and (row['langs'].find('English') != -1 or row['langs'].find('Russian') != -1):
        male += 1
    if row['sex'] == 1 and (row['langs'].find('English') != -1 or row['langs'].find('Russian') != -1):
        female += 1
    return False

df.apply(male_female_langs, axis = 1)
s = pd.Series(data = [female, male],
index = ['Девушки', 'Мужчины'])
s.plot(kind = 'barh')
plt.show()


df.drop(['bdate', 'has_photo', 'has_mobile', 'followers_count', 'graduation', 'relation', 'last_seen', 'life_main', 'people_main', 'id', 'city', 'occupation_name', 'relation', 'career_start', 'career_end'], axis = 1, inplace = True)




def sex_apply(sex):
    if sex ==2:
        return 1
    return 0
df['sex'] = df['sex'].apply(sex_apply)


df['education_form'].fillna('Full-time', inplace = True)
df[list(pd.get_dummies(df['education_form']).columns)] = pd.get_dummies(df['education_form'])
df.drop(['education_form'], axis = 1, inplace = True)

def status_apply(status):
    if status == 'Undergraduate applicant':
        return 0
    elif status == "Student (Bachelor's)" or status == "Student (Master's)" or status == "Student (Specialist)":
        return 1
    elif status == "Alumns (Bachelor's)" or status == "Alumns (Master's)" or status == "Alumns (Specialist)":
        return 2
    else:
        return 3
df['education_status'] = df['education_status'].apply(sex_apply)

def langs_apply(lang):
    if lang.find('English') != -1 and lang.find('Russian') != -1:
        return 1
    else:
        return 0
df['langs'] = df['langs'].apply(sex_apply)
print(df['langs'].value_counts())

df['occupation_type'].fillna('university', inplace = True)
def occupation_type_apply(occupation_type):
    if occupation_type == 'university':
        return 1
    else:
        return 0
df['occupation_type'] = df['occupation_type'].apply(occupation_type_apply)
df.info()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

x = df.drop('result', axis = 1)
y = df['result']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print('Процент правильно предсказанных исходов:', round(accuracy_score(y_test, y_pred), 2) * 100)
print('Confusion matrix')
print(confusion_matrix(y_test, y_pred))







    










