import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier



""" Ucitavanje dataseta """
data = pd.read_csv('mushrooms.csv')

""" Sredjivanje izlaza - klasa """
data['class'].unique()
data['class'] = data['class'].apply(lambda label: int(label == 'p'))

""" Izbacivanje atributa koji ima istu vrednost """
data = data.drop('veil-type', axis=1)

""" Provera da li ima duplikata unutar dataseta """
tot = len(set(data.index))
last = data.shape[0]- tot 
print('Ima {} duplikata.\n'.format(last))

""" Provera da li ima null vrednosti unutar dataseta """
null_count = 0
for val in data.isnull().sum():  
    null_count += val
print('Ima {} null vrednosti.\n'.format(null_count))

""" Informacije u vezi podataka """
data.info()  

""" Prikaz broja vrednosti unutar odredjenog atributa """
def show_features(data):
    col_count, col_var = [], []
    for col in data:
        col_count.append(len(data[col].unique()))
        col_var.append(data[col].unique().sum())
    data_dict = {'Broj': col_count, 'Promenljive': col_var}
    data_table = pd.DataFrame(data_dict, index=data.columns)
    print(data_table)
     
show_features(data) 

""" Graficki prikaz podataka """
plt.figure(1)
sns.set(style="ticks", color_codes=True)
plt.title("Odnos broja primeraka po pripadnosti",fontsize=14)
ax1 = sns.countplot(x = data["class"], data = data)
plt.ylabel('Broj primeraka')
plt.xlabel('Pripadnost klasama')
for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.0f}'.format(height),
            ha="center") 

plt.figure(2)
sns.set(style="ticks", color_codes=True)
plt.title('Atribut - Cap Shape')
sns.countplot(x='cap-shape', hue='class', data=data)
plt.xticks(np.arange(10),('Convex', 'Bell', 'Sunken', 'Flat', 'Knobbed', 'Conical'), rotation='vertical')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')

plt.figure(3)
sns.set(style="ticks", color_codes=True)
plt.title('Atribut - Cap surface')
sns.countplot(x='cap-surface', hue='class', data=data)
plt.xticks(np.arange(10),('Smooth', 'Scaly', 'Fibrous', 'Grooves'), rotation='vertical')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')

plt.figure(4)
sns.set(style="ticks", color_codes=True)
sns.countplot(x='cap-color', hue='class', data=data)
plt.title('Atribut - Cap Color')
plt.xticks(np.arange(11),('Brown', 'Yellow', 'White', 'Gray', 'Red', 'Pink', 'Buff', 'Purple', 'Cinnamon', 'Green'), rotation='vertical')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')

plt.figure(5)
sns.set(style="ticks", color_codes=True)
sns.countplot(x='bruises', hue='class', data=data)
plt.title('Atribut - Bruises')
plt.xticks(np.arange(2),('Bruise', 'No Bruise'), rotation='vertical')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')

plt.figure(6)
sns.set(style="ticks", color_codes=True)
plt.title('Atribut - Odor')
sns.countplot(x='odor', hue='class', data=data)
plt.xticks(np.arange(10),('Pungent', 'Almond', 'Anise', 'None', 'Foul', 'Creosote', 'Fish', 'Spicy', 'Musty'), rotation='vertical')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')

plt.figure(7)
sns.set(style="ticks", color_codes=True)
plt.title('Atribut - Gill attachment')
sns.countplot(x='gill-attachment', hue='class', data=data)
plt.xticks(np.arange(10),('Free', 'Attached', 'Descending', 'Notched'), rotation='vertical')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')

plt.figure(8)
sns.set(style="ticks", color_codes=True)
plt.title('Atribut - Gill spacing')
sns.countplot(x='gill-spacing', hue='class', data=data)
plt.xticks(np.arange(10),('Close', 'Crowded', 'Distant'), rotation='vertical')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')

plt.figure(9)
sns.set(style="ticks", color_codes=True)
plt.title('Atribut - Gill size')
sns.countplot(x='gill-size', hue='class', data=data)
plt.xticks(np.arange(10),('Narrow', 'Broad'), rotation='vertical')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')

plt.figure(10)
sns.set(style="ticks", color_codes=True)
plt.title('Atribut - Gill color')
sns.countplot(x='gill-color', hue='class', data=data)
plt.xticks(np.arange(10),('Black', 'Brown', 'Gray', 'Pink', 'White', 'Chocolate', 'Purple', 'Red', 'Buff', 'Green', 'Yellow', 'Orange'), rotation='vertical')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')

plt.figure(11)
sns.set(style="ticks", color_codes=True)
plt.title('Atribut - Stalk shape')
sns.countplot(x='stalk-shape', hue='class', data=data)
plt.xticks(np.arange(10),('Enlarging', 'Tapering'), rotation='vertical')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')

plt.figure(12)
sns.set(style="ticks", color_codes=True)
plt.title('Atribut - Stalk root')
sns.countplot(x='stalk-root', hue='class', data=data)
plt.xticks(np.arange(10),('Equal', 'Club', 'Bulbous', 'Rooted', 'Unknown', 'Cup', 'Rhizomorphs'), rotation='vertical')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')

plt.figure(13)
sns.set(style="ticks", color_codes=True)
plt.title('Atribut - Stalk surface above ring')
sns.countplot(x='stalk-surface-above-ring', hue='class', data=data)
plt.xticks(np.arange(10),('Smooth', 'Fibrous', 'Silky', 'Scaly'), rotation='vertical')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')

plt.figure(14)
sns.set(style="ticks", color_codes=True)
plt.title('Atribut - Stalk surface below ring')
sns.countplot(x='stalk-surface-below-ring', hue='class', data=data)
plt.xticks(np.arange(10),('Smooth', 'Fibrous', 'Scaly', 'Silky'), rotation='vertical')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')

plt.figure(15)
sns.set(style="ticks", color_codes=True)
plt.title('Atribut - Stalk color above ring')
sns.countplot(x='stalk-color-above-ring', hue='class', data=data)
plt.xticks(np.arange(10),('White', 'Gray', 'Pink', 'Brown', 'Buff', 'Red', 'Orange', 'Cinnamon', 'Yellow'), rotation='vertical')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')

plt.figure(16)
sns.set(style="ticks", color_codes=True)
plt.title('Atribut - Stalk color below ring')
sns.countplot(x='stalk-color-below-ring', hue='class', data=data)
plt.xticks(np.arange(10),('White', 'Pink', 'Gray', 'Buff', 'Brown', 'Red', 'Yellow', 'Orange', 'Cinnamon'), rotation='vertical')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')

plt.figure(17)
sns.set(style="ticks", color_codes=True)
plt.title('Atribut - Veil color')
sns.countplot(x='veil-color', hue='class', data=data)
plt.xticks(np.arange(10),('White', 'Brown', 'Orange', 'Yellow'), rotation='vertical')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')

plt.figure(18)
sns.set(style="ticks", color_codes=True)
plt.title('Atribut - Ring number')
sns.countplot(x='ring-number', hue='class', data=data)
plt.xticks(np.arange(10),('One', 'Two', 'None'), rotation='vertical')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')

plt.figure(19)
sns.set(style="ticks", color_codes=True)
plt.title('Atribut - Ring type')
sns.countplot(x='ring-type', hue='class', data=data)
plt.xticks(np.arange(10),('Pendant', 'Evanescent', 'Large', 'Flaring', 'None', 'Cobwebby', 'Sheathing', 'Zone'), rotation='vertical')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')

plt.figure(20)
sns.set(style="ticks", color_codes=True)
plt.title('Atribut - Spore print color')
sns.countplot(x='spore-print-color', hue='class', data=data)
plt.xticks(np.arange(10),('Black', 'Brown', 'Purple', 'Chocolate', 'White', 'Green', 'Orange', 'Yellow', 'Buff'), rotation='vertical')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')

plt.figure(21)
sns.set(style="ticks", color_codes=True)
sns.countplot(x='population', hue='class', data=data)
plt.title('Atribut - Population')
plt.xticks(np.arange(7),('Scattered', 'Numerous', 'Abundant', 'Several', 'Solitary', 'Clustered'), rotation='vertical')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')

plt.figure(22)
sns.countplot(x='habitat', hue='class', data=data)
plt.title('Atribut - Habitat')
plt.xticks(np.arange(8),('Urban', 'Grasses', 'Meadows', 'Woods', 'Paths', 'Waste', 'Leaves'), rotation='vertical')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
plt.legend(loc='upper right')

plt.tight_layout()
sns.despine()
plt.show()

""" Odredjivanje ulaznih i izlaznih podataka - Atributi:Klase """
X = data.drop('class',axis=1) 
y = data['class']             

""" Label enkodovanje """
Encoder_X = LabelEncoder() 
for col in X.columns:
    X[col] = Encoder_X.fit_transform(X[col])
Encoder_y = LabelEncoder()
y = Encoder_y.fit_transform(y)

""" Konvertovanje 'kategorickih' atributa u indikatorske promenljive """
X = pd.get_dummies(X,columns=X.columns,drop_first=True)


""" Podela dataseta na podatke za trening - test """
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=33)

""" Algoritmi """
r1 = LogisticRegression()
r1.fit(X_train, y_train)
r1.predict(X_test)
resenje1 = r1.score(X_test,y_test)
print('Logisticka regresija: ', resenje1)


r2 = DecisionTreeClassifier()
r2.fit(X_train, y_train)
r2.predict(X_test)
resenje2 = r2.score(X_test, y_test)
print('Stablo odlucivanja: ', resenje2)


r3 = RandomForestRegressor(n_estimators=1, random_state=0)
r3.fit(X_train, y_train)
r3.predict(X_test)
resenje3 = r3.score(X_test, y_test)
print('Slucajna suma: ', resenje3)


r4 = GaussianNB()
r4.fit(X_train, y_train)
r4.predict(X_test)
resenjen4 = r4.score(X_test, y_test)
print('Naivni Bajes: ', resenjen4)



r5 = SVC()
r5.fit(X_train, y_train)
r5.predict(X_test)
resenje5 = r5.score(X_test,y_test)
print('Potporni vektori: ', resenje5)


r6 = KNeighborsClassifier()
r6.fit(X_train, y_train)
r6.predict(X_test)
resenje6 = r6.score(X_test, y_test)
print('K najblizih suseda:', resenje6)

