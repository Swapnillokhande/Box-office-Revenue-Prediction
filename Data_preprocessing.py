#predict world wide box-office revenue for the movies



import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import json
import ast
import calendar

train_data=pd.read_csv("D:/Mactores/Boxoffice Revenue/tmdb-box-office-prediction/train.csv",na_values='.')
test_data=pd.read_csv("D:/Mactores/Boxoffice Revenue/tmdb-box-office-prediction/test.csv")

#Data exploration

#combine the data sets (train and test) to check the properties

train_data['source']='train_data'
test_data['source']='test_data'
comb_data = pd.concat([train_data, test_data],ignore_index=True)
print ("number of rows and columns in Train dataset ",train_data.shape)
print("number of rows and columns in Test dataset ",test_data.shape)
print("number of rows and columns in combined dataset ",comb_data.shape)

#find the data type of each variable
print("Types columns : \n" + str(comb_data.dtypes))

#find the unique entries
print("\nSummary for unique values\n",comb_data.apply(lambda x: len(x.unique())))

#check whether the combined dataset contains any missing values in any column

print("Summary for missing values\n",comb_data.isnull().sum())

#Check the percentage of null values per variable
print(comb_data.isnull().sum()/comb_data.shape[0]*100) #show values in percentage

#drop unwanted columns (parameters) or those columns having more than 50% missing values

comb_data.drop(['id','belongs_to_collection','homepage','imdb_id','original_title','overview','popularity','status','tagline','title'],axis=1,inplace=True)



#define functions
def parse_json(data):
    
    for j in data:
        ret = data[j]
    return ret

#comb_data['production_companies']=comb_data['production_companies'].apply(parse_json)
#print(comb_data['production_companies'].head(10))

# Helper function to parse text and convert given strings to lists
def text_to_list(x):
 if pd.isna(x):
     return ''
 else:
     return ast.literal_eval(x)

for col in ['genres', 'production_companies', 'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']:
   comb_data[col] = comb_data[col].apply(text_to_list)




#create a feature - Number of genres for each movie
comb_data['genre_number']=comb_data['genres'].apply(lambda x:len(x))
print(comb_data['genre_number'].head(10))


#create a function to make three new columns for genres if movie has more than 1 genre
def parse_genre(x):
    if type(x) == str:
        return pd.Series([str('Unknown'),'',''], index=['genres1', 'genres2', 'genres3'] )
    if len(x) == 1:
        return pd.Series([str(x[0]['name']),'NA','NA'], index=['genres1', 'genres2', 'genres3'] )
    if len(x) == 2:
        return pd.Series([str(x[0]['name']),str(x[1]['name']),'NA'], index=['genres1', 'genres2', 'genres3'] )
    if len(x) > 2:
        return pd.Series([str(x[0]['name']),str(x[1]['name']),str(x[2]['name'])], index=['genres1', 'genres2', 'genres3'])

comb_data[['genres1', 'genres2', 'genres3']]=comb_data['genres'].apply(parse_genre)
print(comb_data['genres1'].head(10))
comb_data.drop(columns='genres', inplace=True)

#create a feature - Number of production companies
comb_data['production_company_number'] = comb_data['production_companies'].apply(lambda x: len(x))
print(comb_data['production_company_number'].head(10))

#create a function to make three new columns for production companies if movie has more than 1 production company
def parse_production_companies(x):
    if type(x) == str:
        return pd.Series(['Unknown','',''], index=['prod1', 'prod2', 'prod3'])
    if len(x) == 1:
        return pd.Series([str(x[0]['name']),'NA','NA'], index=['prod1', 'prod2', 'prod3'])
    if len(x) == 2:
        return pd.Series([x[0]['name'],x[1]['name'],'NA'], index=['prod1', 'prod2', 'prod3'])
    if len(x) > 2:
        return pd.Series([x[0]['name'],x[1]['name'],x[2]['name']], index=['prod1', 'prod2', 'prod3'])

comb_data[['prod1','prod2','prod3']]=comb_data['production_companies'].apply(parse_production_companies)
print(comb_data['prod1'].head(10))
comb_data.drop(columns='production_companies', inplace=True)

#create a feature - Number of production countries for each movie
comb_data['production_countries_number']=comb_data['production_countries'].apply(lambda x: len(x))
print(comb_data['production_countries_number'].head(10))

#create a function to make three new columns for production countries if movie has more production company from more than 1 country

def parse_production_countries(x):
    if type(x) == str:
        return pd.Series(['Unknown','',''], index=['country1', 'country2', 'country3'])
    if len(x) == 1:
        return pd.Series([x[0]['name'],'NA','NA'], index=['country1', 'country2', 'country3'])
    if len(x) == 2:
        return pd.Series([x[0]['name'],x[1]['name'],'NA'], index=['country1', 'country2', 'country3'])
    if len(x) > 2:
        return pd.Series([x[0]['name'],x[1]['name'],x[2]['name']], index=['country1', 'country2', 'country3'])

comb_data[['country1', 'country2', 'country3']]=comb_data['production_countries'].apply(parse_production_countries)
print(comb_data['country1'].head(10))
comb_data.drop(columns=['production_countries'],inplace=True)

# Parse and break-down the date column (‘release_date’ column)
comb_data['release_date'] = pd.to_datetime(comb_data['release_date'])#, format='%m/%d/%y',errors='ignore')
print(comb_data['release_date'].head(10))

# Parse ‘weekday’
comb_data['weekday'] = comb_data['release_date'].dt.weekday_name
# fill Nan in ‘weekday’ column with the most common weekday value — 4 (Friday)
comb_data['weekday'].fillna('Friday', inplace=True)
print(comb_data['weekday'].head(10))

# Parse ‘month’
comb_data['month'] = comb_data['release_date'].dt.month_name()
# fill Nan in ‘month’ with the most common month value 
comb_data['month'].fillna(comb_data['month'].value_counts().idxmax(), inplace=True)

print(comb_data['month'].head(10))

# Parse ‘year’ 
comb_data['year'] = comb_data['release_date'].dt.year
# fill Nan in ‘year’ with the median value of the ‘year’ column
comb_data['year'].fillna(str(comb_data['year'].median()), inplace=True)
print(comb_data['year'].head(10))


# Drop the original ‘release_date’ column 
comb_data.drop(columns =['release_date'], inplace=True)

#Fill the missing values in the ‘runtime’ column with the mean value.
comb_data['runtime'].fillna(comb_data['runtime'].mean(),inplace=True)
print(comb_data['runtime'].head(10))
comb_data['runtime']=comb_data.runtime.mask(comb_data.runtime == 0,comb_data['runtime'].mean())

#create a column for number of spoken languages
comb_data['spoken_languages_number'] = comb_data['spoken_languages'].apply(lambda x: len(x))

#create a function to make three new columns for spoken languages if movie has more than 1 sppken language

def parse_spoken_languages(x):
    if type(x) == str:
        return pd.Series(['Unknown','',''], index=['lang1', 'lang2', 'lang3'] )
    if len(x) == 1:
        return pd.Series([x[0]['name'],'NA','NA'], index=['lang1', 'lang2', 'lang3'] )
    if len(x) == 2:
        return pd.Series([x[0]['name'],x[1]['name'],'NA'], index=['lang1', 'lang2', 'lang3'] )
    if len(x) > 2:
        return pd.Series([x[0]['name'],x[1]['name'],x[2]['name']], index=['lang1', 'lang2', 'lang3'] )

comb_data[['lang1', 'lang2', 'lang3']]=comb_data['spoken_languages'].apply(parse_spoken_languages)
print(comb_data['lang1'].head(10))
comb_data.drop(columns=['spoken_languages'],inplace=True)

#Create a new column with the number of Keywords for each movie
comb_data['keywords_number'] = comb_data['Keywords'].apply(lambda x: len(x))
print(comb_data['keywords_number'].head(10))
comb_data.drop(columns=['Keywords'],inplace=True)

#Create a new column with the number of cast values for each movie
comb_data['cast_number']=comb_data['cast'].apply(lambda x: len(x))
print(comb_data['cast_number'].head(10))

#fill cast_number with mean where cast_number = 0
comb_data['cast_number']=comb_data.cast_number.mask(comb_data.cast_number == 0,comb_data['cast_number'].mean())

#create a function to make three new columns for top three cast members
def parse_cast(x):
    myindx = ['cast1', 'cast2', 'cast3']
    out = [-1]*3
    if type(x) != str:
        for i in range(min([3,len(x)])):
            out[i] = str(x[i]['id'])
    return pd.Series(out, index=myindx)

comb_data[['cast1', 'cast2', 'cast3']]=comb_data['cast'].apply(parse_cast)
print(comb_data['cast1'].head(10))
comb_data.drop(columns=['cast'],inplace=True)

#Create a new column with the number of crew values for each movie
comb_data['crew_number']=comb_data['crew'].apply(lambda x: len(x))

#create function to parse the Director from the ‘crew’ column
def parse_crew(x):
    myindx = ['Director']
    out = [-1]*1
    if type(x) != str:
        for item in x:
            if item['job'] == 'Director':
                    out[0] = str(item['id'])
    return pd.Series(out, index=myindx)
comb_data['Director']=comb_data['crew'].apply(parse_crew)
print(comb_data['Director'].head(10))
comb_data.drop(columns=['crew'],inplace=True)

#fill budget==0 with mean budget
comb_data.loc[comb_data['budget'] == 0,'budget'] = comb_data['budget'].mean()
print(comb_data['budget'].head(10))

#create a new column Budget_cast_ratio
comb_data['Budget_cast_ratio']=comb_data['budget']/comb_data['cast_number']
comb_data.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
comb_data['Budget_cast_ratio'].fillna(comb_data['Budget_cast_ratio'].mean(),inplace=True)
print(comb_data['Budget_cast_ratio'].head(10))

#create a new column budget_runtime_ratio

comb_data['Budget_runtime_ratio']=comb_data['budget']/comb_data['runtime']
comb_data.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
comb_data['Budget_runtime_ratio'].fillna(comb_data['Budget_runtime_ratio'].mean(),inplace=True)
print(comb_data['Budget_runtime_ratio'].head(10))

#create a new column mean_budget_by_year
comb_data['mean_budget_by_year']=comb_data.groupby('year').budget.transform('mean')
print(comb_data['mean_budget_by_year'].head(10))

#create a new column has_poster
comb_data['has_poster']=1
comb_data.loc[pd.isnull(comb_data['poster_path']) ,"has_poster"] = 0
print(comb_data['has_poster'].head(10))
comb_data.drop(columns=['poster_path'],inplace=True)



#find the data type of each variable
print("Types columns : \n" + str(comb_data.dtypes))

#Check the percentage of null values per variable
print(comb_data.isnull().sum()/comb_data.shape[0]*100) 


#transform categorcial variable using label encoder and create dummy variable using one-hot coding

cols = ['genres1', 'genres2', 'genres3'] 
allitems = list(set(comb_data[cols].values.ravel().tolist()))
labeler = LabelEncoder()
labeler.fit(allitems)
comb_data[cols] = comb_data[cols].apply(lambda x: labeler.transform(x)) 

#cols = ['prod1', 'prod2', 'prod3']
#allitems = list(set(comb_data[cols].values.ravel().tolist()))
#labeler = LabelEncoder()
#labeler.fit(allitems) 
#comb_data[cols] = comb_data[cols].apply(lambda x:   labeler.transform(x))

cols = ['lang1', 'lang2', 'lang3']
allitems = list(set(comb_data[cols].values.ravel().tolist()))
labeler = LabelEncoder() 
labeler.fit(allitems)
comb_data[cols] = comb_data[cols].apply(lambda x: labeler.transform(x))

#cols = ['country1', 'country2', 'country3']
#allitems = list(set(comb_data[cols].values.ravel().tolist()))
#labeler = LabelEncoder()
#labeler.fit(allitems) 
#comb_data[cols] = comb_data[cols].apply(lambda x: labeler.transform(x))

cols = ['weekday'] 
allitems = list(set(comb_data[cols].values.ravel().tolist()))
labeler = LabelEncoder()
labeler.fit(allitems)
comb_data[cols] = comb_data[cols].apply(lambda x: labeler.transform(x)) 

cols = ['month'] 
allitems = list(set(comb_data[cols].values.ravel().tolist()))
labeler = LabelEncoder()
labeler.fit(allitems)
comb_data[cols] = comb_data[cols].apply(lambda x: labeler.transform(x))

#use onehot code to create dummy variable
comb_data = pd.get_dummies(comb_data, columns=['genres1', 'genres2', 'genres3','lang1', 'lang2', 'lang3','weekday','month'])


#split test and train dataset
train = comb_data.loc[comb_data['source']=="train_data"]
test = comb_data.loc[comb_data['source']=="test_data"]


#Drop unnecessary columns:
test.drop(['revenue','source'],axis=1,inplace=True)
train.drop(['source'],axis=1,inplace=True)

#Export files as modified versions:
train.to_csv("D:/Mactores/Boxoffice Revenue/train_modified.csv",index=False)
test.to_csv("D:/Mactores/Boxoffice Revenue/test_modified.csv",index=False)
