# import liblaries
#liblaries for requests
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# %matplotlib inline

#libraries for ML
from scipy import stats
from scipy.stats import kruskal
import sklearn as sk
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.svm import SVR
import warnings

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor



warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)



pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)


def download_file(file_name, file_url):
    if os.path.exists(file_name):
        pass
    else:
        response = requests.get(file_url)
        if response.status_code == 200:
            with open(file_name, 'wb') as f:
                f.write(response.content)
            print(f'{file_name} was downloaded')
        else:
            print(f'Failed to download the document. Status code: {response.status_code}')


file_url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-11-03/ikea.csv"
file_name = 'ikea.csv'

download_file(file_name, file_url)

df = pd.read_csv(file_name, encoding='utf-8')
# print(df.head(1))
'''
item_id - unique number for item. Type: int64
name - name of item. Type: int64
category - category name. Type: int64
price - price of item. Type: float64
old_price - have or not old_price. Type: object
sellable_online - item in online market. Type: bool
link - link on item in market. Type: object
other_colors - available or not different color of item. Type: object
short_description - short info about item . Type: object
designer - designer name. Type: object
depth - depth of item. Type: float64
height - height of item. Type: float64
width - width of item. Type: float64

'''
# print('\nFirst elements')
# print(df.head(3))
# print('\nLast elements')
# print(df.tail(3))
# print('\nRandom elements')
# print(df.sample(3))
# print('\nTotal duplicated')
# print(df.duplicated().sum())
# print('\nUnique values')
# print(df.nunique())
# print('\nColumns info')
# print(df.info())
# print('\nColumns nulls')
# print(df.isnull().sum())
# print('\nMain stat info')
# print(df.describe())

### Check duplicates
# print(df[df.duplicated(subset='item_id')])
# print(df.query('item_id == 10443629'))

### Drop duplicates
df = df.drop_duplicates(subset='item_id').reset_index(drop=True)
# print(df.info())
# print(df.nunique())

### Drop useless columns
df = df.drop(columns=['item_id', 'Unnamed: 0'])
# print(df.columns)
# print(df.shape)

###COLORS
### Creating a function to extract colors from a link
def extract_colors(link):
    parts = link.split('/')
    colors = set()
    for part in parts:
        sub_parts = part.split('-')
        for sub_part in sub_parts:
            if 'black' in sub_part.lower():
                colors.add('black')
            elif 'white' in sub_part.lower():
                colors.add('white')
            elif 'gray' in sub_part.lower() or 'grey' in sub_part.lower():
                colors.add('gray')
            elif 'blue' in sub_part.lower():
                colors.add('blue')
            elif 'red' in sub_part.lower():
                colors.add('red')
            elif 'green' in sub_part.lower():
                colors.add('green')
            elif 'yellow' in sub_part.lower():
                colors.add('yellow')
            elif 'brown' in sub_part.lower():
                colors.add('brown')
            elif 'pink' in sub_part.lower():
                colors.add('pink')
            elif 'purple' in sub_part.lower():
                colors.add('purple')
            elif 'orange' in sub_part.lower():
                colors.add('orange')
            elif 'beige' in sub_part.lower():
                colors.add('beige')
            elif 'gold' in sub_part.lower():
                colors.add('gold')
            elif 'silver' in sub_part.lower():
                colors.add('silver')
    if not colors:
        return 'unknown_color'
    return ', '.join(colors)

###Creating a new column 'all_colors'
df['all_colors'] = df['link'].apply(extract_colors)
# print('\nInfo about all_colors')
# print(df['all_colors'].describe())
# print('\nUnique all_colors')
# print(df['all_colors'].unique())
# print(df['all_colors'].value_counts().sort_values(ascending=False))
# print(df[['all_colors']].head(5))
# print(df[['all_colors']].tail(5))

###Counting the number of each color and creating df with the most popular colors and vice versa
all_colors_count=df['all_colors'].value_counts(normalize=True).sort_values(ascending=False)
# print(all_colors_count)

cumulative_percentage = all_colors_count.cumsum()
# print('---')
# print(cumulative_percentage)

popular_colors=all_colors_count[cumulative_percentage < 0.5][::1]
# print(f'\nPopular colors - more than half of all items')
# print(popular_colors)

### Updating df popular_colors_df and less_popular_colors_df
popular_colors_df=df[df['all_colors'].isin(popular_colors.index)]
less_popular_colors_df=df[~df['all_colors'].isin(popular_colors.index)]
# print('\nShapes of pop and less_pop colors df')
# print(popular_colors_df.shape)
# print(less_popular_colors_df.shape)


### Show all_colors(10 most used colors)
color_counts = df['all_colors'].value_counts().head(10)
# print(color_counts)
# colors = ['#FFFFFF',  # white
#           '#C8A2C8',  # purple
#           '#000000',  # black
#           '#808080',  # gray
#           '#8B4513',  # brown
#           '#654321',  # brown, black
#           '#F5F5DC',  # beige
#           '#D3D3D3',  # white, gray
#           '#D2B48C',  # gray, brown
#           '#FF0000']  # red
# explode = [0.1 if i == 0 else 0 for i in range(len(color_counts))]
# #
# plt.figure(figsize=(10, 6))
# color_counts.plot(kind='pie', autopct='%1.1f%%',
#                   colors=colors, explode=explode,
#                   shadow=True, wedgeprops={'edgecolor': 'gray'},
#                   textprops={'color': 'blue'})
# plt.title('10 most used colors in IKEA DataFrame')
# plt.ylabel('')
# plt.tight_layout()
# plt.show()


###Show Other_colors_online
# print('\nOther_colors distribution')
# print(df['other_colors'].value_counts(normalize=True).sort_values(ascending=False))
# ax = sns.countplot(x = df['other_colors'], hue=df['other_colors'])
# for i in ax.containers:
#     ax.bar_label(i,)
# plt.show()

###CATEGORY
df.category.unique()
df.category.nunique()
df['category'].value_counts()
top10_count_category=df['category'].value_counts()[0:10]
# print(top10_count_category)
# df['short_category'] = df['category'].apply(lambda x: x[:10])
# plt.figure(figsize=(12,8))
# sns.countplot(x=df.short_category, data=df)
# plt.title('division for category')
# plt.xlabel('category')
# plt.ylabel('frequency')
# plt.xticks(rotation=45)
# plt.show()

###Show distribution of categories by count of items
# print('\ncategory distribution')
category_dist = df['category'].value_counts(normalize=True).sort_values(ascending=False)
# print(category_dist.head(10))
#
# ax = sns.countplot(x = df['category'], hue=df['category'], order=category_dist.index)
# for i in ax.containers:
#     ax.bar_label(i,)
# ax.set_xticklabels([s[:10] for s in category_dist.index], rotation=45, fontsize=7)
# plt.show()

# print(df.sort_values('price',ascending=False).max())
df_price_category=df.groupby('category')['price'].median()
# print(df_price_category)

df_price_max=df.groupby('category')['price'].max()
# print(df_price_max)

df_price_category_top5 = df.groupby('category')['price'].median().sort_values().tail(5).index
df_top5 = df[df['category'].isin(df_price_category_top5)]

# plt.figure(figsize=(10, 8))
#
# sns.histplot(x='price', hue='category', data=df_top5,
#              palette='inferno', fill=True,
#              kde=True, common_norm=False)
#
# plt.xlabel('Price')
# plt.ylabel('Count')
# plt.title('Distribution of Prices for Top 5 Categories')
# plt.show()
#
median_d = df.groupby(['category'])['depth'].median()
median_h = df.groupby(['category'])['height'].median()
median_w = df.groupby(['category'])['width'].median()
#
#
df = df.set_index(['category'])
df['depth_1'] = df['depth'].fillna(median_d)
df['height_1'] = df['height'].fillna(median_h)
df['width_1'] = df['width'].fillna(median_w)
# print('With category index')
# print(df.head(10))

###Show values of category columns
# print('\nCategory columns values')
for c in df.columns:
    s = set(df[c])
    if len(s)<20:
        print(c, ':', s)


# ###PRICE
# print('\nPrice chaking ')
# print(df['price'].describe())
# df.sort_values('price',ascending=False).max()
# sns.histplot(df['price'], kde = True)
# plt.show()
#
# ###show price distibution
# sns.boxplot(df['price'])
# plt.show()
# print(df[df['price']>9000])
# #
# ###show 10 items with MIN price
# print('\n 10 Most cheapest items')
# print(df.sort_values(by='price').head(10))
#
# ###show 10 items with MAX price
# print('\n 10 Most expensive items')
# print(df.sort_values(by='price',ascending=False).head(10))
#
# ###show 3 items with MIN/MAX mean price by category
df_price_category=df.groupby('category')['price'].mean()
# print(df_price_category.sort_values().head(3))
# print(df_price_category.sort_values(ascending=False).head(3))

#
# ###Correlation matrix for numbers
# pearson_cor_matrix = df[['price', 'depth', 'height', 'width']].corr()
# sns.heatmap(pearson_cor_matrix, xticklabels= pearson_cor_matrix.columns,
#             yticklabels= pearson_cor_matrix.columns, annot= True)
# plt.show()
#
#
# spearman_cor_matrix = df[['price', 'depth', 'height', 'width']].corr(method='spearman')
# sns.heatmap(spearman_cor_matrix, xticklabels= spearman_cor_matrix.columns,
#             yticklabels= spearman_cor_matrix.columns, annot= True)
# plt.show()


###DESIGNERS
# print('\nInfo about designers')
# print(df['designer'].describe())
# print('\nUnique designers')
# print(df['designer'].unique())
# print(df['designer'].value_counts().sort_values(ascending=False))

##Creating a length of designer`s name
df['designer_len'] = [len(x) for x in df['designer']]
# print(df['designer_len'].value_counts().sort_values())
# print(df.head())
# print(df.sort_values(by='designer_len', ascending=False).head(50))

###Creating a function to clear the 'designer' column
def cleanDesigners(value, removeIKEA=False, emptyValue=np.nan):
    if not isinstance(value, str):
        return value

    if len(value) > 0 and value[0].isdigit():
        return emptyValue

    designers = value.split("/")

    if removeIKEA:
        try:
            designers.remove("IKEA of Sweden")
        except:
            pass
    if len(designers) > 0:
        return '/'.join(sorted(designers))
    else:
        return emptyValue

df['designer_clean'] = df['designer'].apply(cleanDesigners, args= (False, "IKEA of Sweden"))
df['designer_encoded'] = df['designer_clean'].factorize()[0]


'''
Method: Kruskal-Wallis Test
Hypothesis - The category of furniture significantly affects its size.
H0 - There is no statistical dependence between the sizes of furniture and their categories.
H1 - There is a statistically significant dependence between the sizes of furniture and their categories.
'''

categories = df.index.unique()
# Preparing data for the test
depth_groups = [df['depth_1'][df.index == category] for category in categories]
height_groups = [df['height_1'][df.index == category] for category in categories]
width_groups = [df['width_1'][df.index == category] for category in categories]

# # Test Kruskal-Wallis for depth
# stat_depth, p_depth = kruskal(*depth_groups)
# print(f'Kruskal-Wallis test for depth: H-statistic = {stat_depth}, p-value = {p_depth}')
#
# # Test Kruskal-Wallis for height
# stat_height, p_height = kruskal(*height_groups)
# print(f'Kruskal-Wallis test for height: H-statistic = {stat_height}, p-value = {p_height}')
#
# # Test Kruskal-Wallis for width
# stat_width, p_width = kruskal(*width_groups)
# print(f'Kruskal-Wallis test for width: H-statistic = {stat_width}, p-value = {p_width}')
#
# alpha = 0.05
# print("\nResults:")
# if p_depth < alpha:
#     print("There is a significant difference in depth between categories (reject H0)")
# else:
#     print("There is no significant difference in depth between categories (fail to reject H0)")
#
# if p_height < alpha:
#     print("There is a significant difference in height between categories (reject H0)")
# else:
#     print("There is no significant difference in height between categories (fail to reject H0)")
#
# if p_width < alpha:
#     print("There is a significant difference in width between categories (reject H0)")
# else:
#     print("There is no significant difference in width between categories (fail to reject H0)")

'''
Method: Chi-Square Test
Hypothesis - The color of the product directly depends on the category of goods.
H0 - Colors are independent of the category of goods.
H1 - Colors depend on the category of goods.
'''
# df = df.reset_index()
# cross_tab = pd.crosstab(df['category'], df['all_colors'])
# chi2_stat, p_val, _, _ = stats.chi2_contingency(cross_tab)
# print("Chi-squared statistic:", chi2_stat)
# print("P-value:", p_val)

# ###MODEL PIPELINE
df = df.reset_index()
X = df[['depth', 'width', 'height', 'category','designer_clean', 'other_colors']]
Y = df['price']
#
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#
# numeric_transf = Pipeline(steps=[
#     ('scaler', StandardScaler()),
#     ('impute', SimpleImputer(strategy='median'))
# ])
#
# categorical_transf = Pipeline(steps=[
#     ('impute', SimpleImputer(strategy='most_frequent')),
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])
#
# col_prepr = ColumnTransformer(transformers=[
#     ('numeric', numeric_transf, ['depth', 'width', 'height']),
#     ('categorical', categorical_transf, ['category', 'designer_clean', 'other_colors'])
# ])
#
# dtr = Pipeline(steps=[
#     ('col_prep', col_prepr),
#     ('dtr', DecisionTreeRegressor(max_depth=10, random_state=42))
# ])
# dtr.fit(X_train, Y_train)
# dtr_predict = dtr.predict(X_test)
#
# print('\nResults of DecisionTreeRegressor')
# print('R^2 : {:.5f}'.format(r2_score(dtr_predict, Y_test)))
# print('MAE : {:.5f}'.format(mean_absolute_error(dtr_predict, Y_test)))
# print('MSE : {:.5f}'.format(np.sqrt(mean_squared_error(dtr_predict, Y_test))))
# print('Feature importance:')
# print('--------------------------------')
# for feat, importance in zip(X_train.columns, dtr.steps[1][1].feature_importances_):
#     print('{:.5f}    {f}'.format(importance, f=feat))
# print('\n')
# #
# #
# df['category_median_price'] = df.groupby('category')['price'].transform('median')
# df['designer_median_price'] = df.groupby('designer')['price'].transform('median')
# def getBestRegressor(X, Y):
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#     models = [
#         LinearRegression(),
#         LassoCV(),
#         RidgeCV(),
#         SVR(kernel='linear'),
#         KNeighborsRegressor(n_neighbors=16),
#         DecisionTreeRegressor(max_depth=10, random_state=42),
#         RandomForestRegressor(random_state=42),
#         GradientBoostingRegressor()
#     ]
#
#     TestModels = pd.DataFrame()
#     tmp = {}
#     for model in models:
#         m = str(model)
#         print(m)
#         tmp['Model'] = m[:m.index('(')]
#         model.fit(X_train, Y_train)
#         Y_predict = model.predict(X_test)
#         tmp['R^2'] = '{:.5f}'.format(r2_score(Y_test, Y_predict))
#         tmp['MAE'] = '{:.5f}'.format(mean_absolute_error(Y_test, Y_predict))
#         tmp['RMSE'] = '{:.5f}'.format(np.sqrt(mean_squared_error(Y_test, Y_predict)))
#
#         TestModels = pd.concat([TestModels, pd.DataFrame([tmp])])
#
#     TestModels.set_index('Model', inplace=True)
#     res = TestModels
#
#     return res
#
X1 = df[['depth_1', 'width_1', 'height_1', 'category_median_price', 'designer_median_price']]
Y1 = df['price']
#
# test1 = getBestRegressor(X1, Y1)
# print(test1.sort_values(by='R^2', ascending=False))
#
# # ###GridSearchCV for RandomForestRegressor
X_train, X_test, Y_train, Y_test = train_test_split(X1, Y1, test_size=0.2, random_state=42)
params = {'n_estimators': [10, 50, 100],
            'max_depth': [10, 50, 100],
          }
forest_grid = GridSearchCV(RandomForestRegressor(), param_grid=params, cv=5)
# forest_grid.fit(X_train, Y_train)
#
# print('Best Estimator :', forest_grid.best_estimator_)
# print('Best Score     :', forest_grid.best_score_)
# print('')
# print('R^2            : {:.5f}'.format(r2_score(Y_test, forest_grid.predict(X_test))))
# print('MAE            : {:.5f}'.format(mean_absolute_error(forest_grid.predict(X_test), Y_test)))
# print('RMSE           : {:.5f}'.format(np.sqrt(mean_squared_error(forest_grid.predict(X_test), Y_test))))
# print('')
# print('Feature importance:')
# print('--------------------------------')
#
# for feat, importance in zip(X_train.columns, forest_grid.best_estimator_.feature_importances_):
#     print('{:.5f}    {f}'.format(importance, f=feat))
#
# sns.set_style('whitegrid')
# plt.figure(figsize=(10, 6))
# sns.barplot(y=X_train.columns, x=forest_grid.best_estimator_.feature_importances_,palette='viridis')
# plt.title('Feature Importance')
# plt.xlabel('Importance')
# plt.ylabel('Feature')
# plt.tight_layout()
# plt.show()
#
#
# # ###GridSearchCV for DecisionTreeRegressor
# X_train, X_test, Y_train, Y_test = sk.model_selection.train_test_split(X1, Y1, test_size=0.2, random_state=42)
# decision_grid = GridSearchCV(DecisionTreeRegressor(), {'max_depth': [10, 30, 50, 70, 100, 150, 180, 200, None]})
# decision_grid.fit(X_train, Y_train)
#
# print('Best Estimator :', decision_grid.best_estimator_)
# print('Best Score     :', decision_grid.best_score_)
# print('')
# print('R^2            : {:.5f}'.format(sk.metrics.r2_score(Y_test, decision_grid.predict(X_test))))
# print('MAE            : {:.5f}'.format(sk.metrics.mean_absolute_error(decision_grid.predict(X_test), Y_test)))
# print('RMSE           : {:.5f}'.format(np.sqrt(sk.metrics.mean_squared_error(decision_grid.predict(X_test), Y_test))))
# print('')
# print('Feature importance:')
# print('--------------------------------')
#
# for feat, importance in zip(X_train.columns, decision_grid.best_estimator_.feature_importances_):
#     print('{:.5f}    {f}'.format(importance, f=feat))
#
# sns.set_style('whitegrid')
# plt.figure(figsize=(10, 6))
# sns.barplot(y=X_train.columns, x=decision_grid.best_estimator_.feature_importances_,palette='viridis')
# plt.title('Feature Importance')
# plt.xlabel('Importance')
# plt.ylabel('Feature')
# plt.tight_layout()
# plt.show()

