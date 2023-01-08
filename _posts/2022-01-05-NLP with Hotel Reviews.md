---
layout: post
title: Hotel Review Analysis Using NLP Part 1
Pulished: False
---

Understanding how customers respond to thier stay through reviews is vital responding to changes in customer expecations and remaining competitive in the the hotel market. Feedback comes in the form of reviews with positive feedback and negative feedback. This notebook will explore 225 MB of hotel review data and prepare it for a classifiation modeling. The end goal will be to read reviews and predict the overall rating a customer gives thier experience at the hotel.

## Setting up this notebook 
I begin with with some core data science libraries Pandas and Numpy and will use the count vectorizer from SKlearn to breakdown the text of the reviews.

```python
## Import Libraries 
import pandas as pd # dataframe manipulation 
import numpy as np # dependancy for pyplot
import matplotlib.pyplot as plt # create visualizations

from sklearn.model_selection import train_test_split # prepare data for modeling
from sklearn.feature_extraction.text import CountVectorizer # process text data

```


```python
# Read in data
raw_data = pd.read_csv('data/Hotel_Reviews.csv')
```

# Exploritory Data Analyis 

### Data Dictionary

Each row is a hotel guest review of a stay along with some infromation about the hotel and the reviewer. The review is broken up with a positive and negative component and labled with tags that apply to the hotel stay being reviewed. Each review is stored alongside metadata and aggrgate metrics for both the hotel and reviewer. It appears each time a review is added mupltiple database records must be updated.  

**Hotel data**
- Hotel_Address - Property Address / business address
- Average_Score - Composite score for the hotel
- Hotel_Name - Property address/ business name
- Total_Number_of_Reviews - number of reviews completed by guests
- lat - latitude coodrinate
- lng - longitude coordinte



**Review specific data**
- Review_Date - date of feedback
- Negative_Review - text of negative portion of review
- Review_Total_Negative_Word_Counts - number of words in negative review
- Positive_Review - text of positive portion of review
- Review_Total_Positive_Word_Counts - number of words in positive review
- Reviewer_Score - the single number score a reviewer gives the hotel
- Tags - list features that apply to the stay being reviewed

**Reviewer Data**
- Reviewer_Nationality - Country of orgin of guest
- Total_Number_of_Reviews_Reviewer_Has_Given' - number of reviews the guest has given
- days_since_review - the time passed since the guests most recent review

**Other Data**
- Additional_Number_of_Scoring - It is not clear what this number represents. I suspect that this is a weighting factor that affects the average score calculation.



### The Data

Reading the data and looking at a few of it's top level properties I can see there are over half a million reviews. There is some cleaning that needs to happening before I get to modeling though. There a a few columns date or number columns stored as text as well as some missing location information. 

```python
# Read in data
raw_data = pd.read_csv('data/Hotel_Reviews.csv')
print(f'The data constains {raw_data.shape[0]} rows and {raw_data.shape[1]} columns.')
display(raw_data.head(5))
display(raw_data.info())
display(raw_data.describe())
```

    The data constains 515738 rows and 17 columns.
    


<!-- <div>
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
      <th>Hotel_Address</th>
      <th>Additional_Number_of_Scoring</th>
      <th>Review_Date</th>
      <th>Average_Score</th>
      <th>Hotel_Name</th>
      <th>Reviewer_Nationality</th>
      <th>Negative_Review</th>
      <th>Review_Total_Negative_Word_Counts</th>
      <th>Total_Number_of_Reviews</th>
      <th>Positive_Review</th>
      <th>Review_Total_Positive_Word_Counts</th>
      <th>Total_Number_of_Reviews_Reviewer_Has_Given</th>
      <th>Reviewer_Score</th>
      <th>Tags</th>
      <th>days_since_review</th>
      <th>lat</th>
      <th>lng</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>s Gravesandestraat 55 Oost 1092 AA Amsterdam ...</td>
      <td>194</td>
      <td>8/3/2017</td>
      <td>7.7</td>
      <td>Hotel Arena</td>
      <td>Russia</td>
      <td>I am so angry that i made this post available...</td>
      <td>397</td>
      <td>1403</td>
      <td>Only the park outside of the hotel was beauti...</td>
      <td>11</td>
      <td>7</td>
      <td>2.9</td>
      <td>[' Leisure trip ', ' Couple ', ' Duplex Double...</td>
      <td>0 days</td>
      <td>52.360576</td>
      <td>4.915968</td>
    </tr>
    <tr>
      <th>1</th>
      <td>s Gravesandestraat 55 Oost 1092 AA Amsterdam ...</td>
      <td>194</td>
      <td>8/3/2017</td>
      <td>7.7</td>
      <td>Hotel Arena</td>
      <td>Ireland</td>
      <td>No Negative</td>
      <td>0</td>
      <td>1403</td>
      <td>No real complaints the hotel was great great ...</td>
      <td>105</td>
      <td>7</td>
      <td>7.5</td>
      <td>[' Leisure trip ', ' Couple ', ' Duplex Double...</td>
      <td>0 days</td>
      <td>52.360576</td>
      <td>4.915968</td>
    </tr>
    <tr>
      <th>2</th>
      <td>s Gravesandestraat 55 Oost 1092 AA Amsterdam ...</td>
      <td>194</td>
      <td>7/31/2017</td>
      <td>7.7</td>
      <td>Hotel Arena</td>
      <td>Australia</td>
      <td>Rooms are nice but for elderly a bit difficul...</td>
      <td>42</td>
      <td>1403</td>
      <td>Location was good and staff were ok It is cut...</td>
      <td>21</td>
      <td>9</td>
      <td>7.1</td>
      <td>[' Leisure trip ', ' Family with young childre...</td>
      <td>3 days</td>
      <td>52.360576</td>
      <td>4.915968</td>
    </tr>
    <tr>
      <th>3</th>
      <td>s Gravesandestraat 55 Oost 1092 AA Amsterdam ...</td>
      <td>194</td>
      <td>7/31/2017</td>
      <td>7.7</td>
      <td>Hotel Arena</td>
      <td>United Kingdom</td>
      <td>My room was dirty and I was afraid to walk ba...</td>
      <td>210</td>
      <td>1403</td>
      <td>Great location in nice surroundings the bar a...</td>
      <td>26</td>
      <td>1</td>
      <td>3.8</td>
      <td>[' Leisure trip ', ' Solo traveler ', ' Duplex...</td>
      <td>3 days</td>
      <td>52.360576</td>
      <td>4.915968</td>
    </tr>
    <tr>
      <th>4</th>
      <td>s Gravesandestraat 55 Oost 1092 AA Amsterdam ...</td>
      <td>194</td>
      <td>7/24/2017</td>
      <td>7.7</td>
      <td>Hotel Arena</td>
      <td>New Zealand</td>
      <td>You When I booked with your company on line y...</td>
      <td>140</td>
      <td>1403</td>
      <td>Amazing location and building Romantic setting</td>
      <td>8</td>
      <td>3</td>
      <td>6.7</td>
      <td>[' Leisure trip ', ' Couple ', ' Suite ', ' St...</td>
      <td>10 days</td>
      <td>52.360576</td>
      <td>4.915968</td>
    </tr>
  </tbody>
</table>
</div> -->


    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 515738 entries, 0 to 515737
    Data columns (total 17 columns):
     #   Column                                      Non-Null Count   Dtype  
    ---  ------                                      --------------   -----  
     0   Hotel_Address                               515738 non-null  object 
     1   Additional_Number_of_Scoring                515738 non-null  int64  
     2   Review_Date                                 515738 non-null  object 
     3   Average_Score                               515738 non-null  float64
     4   Hotel_Name                                  515738 non-null  object 
     5   Reviewer_Nationality                        515738 non-null  object 
     6   Negative_Review                             515738 non-null  object 
     7   Review_Total_Negative_Word_Counts           515738 non-null  int64  
     8   Total_Number_of_Reviews                     515738 non-null  int64  
     9   Positive_Review                             515738 non-null  object 
     10  Review_Total_Positive_Word_Counts           515738 non-null  int64  
     11  Total_Number_of_Reviews_Reviewer_Has_Given  515738 non-null  int64  
     12  Reviewer_Score                              515738 non-null  float64
     13  Tags                                        515738 non-null  object 
     14  days_since_review                           515738 non-null  object 
     15  lat                                         512470 non-null  float64
     16  lng                                         512470 non-null  float64
    dtypes: float64(4), int64(5), object(8)
    memory usage: 66.9+ MB
    


    None

<!-- <div>
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
<!-- <table border="1" class="dataframe" style="width:10%">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Additional_Number_of_Scoring</th>
      <th>Average_Score</th>
      <th>Review_Total_Negative_Word_Counts</th>
      <th>Total_Number_of_Reviews</th>
      <th>Review_Total_Positive_Word_Counts</th>
      <th>Total_Number_of_Reviews_Reviewer_Has_Given</th>
      <th>Reviewer_Score</th>
      <th>lat</th>
      <th>lng</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>515738.000000</td>
      <td>515738.000000</td>
      <td>515738.000000</td>
      <td>515738.000000</td>
      <td>515738.000000</td>
      <td>515738.000000</td>
      <td>515738.000000</td>
      <td>512470.000000</td>
      <td>512470.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>498.081836</td>
      <td>8.397487</td>
      <td>18.539450</td>
      <td>2743.743944</td>
      <td>17.776458</td>
      <td>7.166001</td>
      <td>8.395077</td>
      <td>49.442439</td>
      <td>2.823803</td>
    </tr>
    <tr>
      <th>std</th>
      <td>500.538467</td>
      <td>0.548048</td>
      <td>29.690831</td>
      <td>2317.464868</td>
      <td>21.804185</td>
      <td>11.040228</td>
      <td>1.637856</td>
      <td>3.466325</td>
      <td>4.579425</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>5.200000</td>
      <td>0.000000</td>
      <td>43.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2.500000</td>
      <td>41.328376</td>
      <td>-0.369758</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>169.000000</td>
      <td>8.100000</td>
      <td>2.000000</td>
      <td>1161.000000</td>
      <td>5.000000</td>
      <td>1.000000</td>
      <td>7.500000</td>
      <td>48.214662</td>
      <td>-0.143372</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>341.000000</td>
      <td>8.400000</td>
      <td>9.000000</td>
      <td>2134.000000</td>
      <td>11.000000</td>
      <td>3.000000</td>
      <td>8.800000</td>
      <td>51.499981</td>
      <td>0.010607</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>660.000000</td>
      <td>8.800000</td>
      <td>23.000000</td>
      <td>3613.000000</td>
      <td>22.000000</td>
      <td>8.000000</td>
      <td>9.600000</td>
      <td>51.516288</td>
      <td>4.834443</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2682.000000</td>
      <td>9.800000</td>
      <td>408.000000</td>
      <td>16670.000000</td>
      <td>395.000000</td>
      <td>355.000000</td>
      <td>10.000000</td>
      <td>52.400181</td>
      <td>16.429233</td>
    </tr>
  </tbody>
</table>
</div> --> 

### Numeric Property Distibutions

Insights 
- Word counts distibution for positive and negative reviews are similar
- Additital Number of Scoring has a smiliar distribution shape as the total number of a reviews. Perhaps the total number of reviews is tied to addtional scoring
- The distibution of Reviewer Score and average score do not match. This seems suprising at first, but considering that the average score is always made up of a collection of individual scores the almost normal distribution of the average score makes sense. 


```python
i = 1
plt.subplots(3,3, figsize=(20,10))

for col in raw_data.select_dtypes(exclude='object').columns:
    
    plt.subplot(3,3,i)
    plt.hist(raw_data[col], bins=10)
    plt.title(f'EDA: {col}')
    plt.xlabel(col)
    i +=1

plt.tight_layout()
plt.show()

```


    
![png](/Kibat_NLP_PART_1_files/Kibat_NLP_PART_1_9_0.png)
    


### Data Cleaning
Guests can leave a 1-10 score for each visit, but after the aggregation I'm left with decimalized ratings. I'm going to simplify the scores by rounding them to integers.


```python
print(f'Rating categories in raw data {raw_data["Reviewer_Score"].value_counts().shape[0]}')

# Round each review score and then save in a new colums as an integer
raw_data['Rating Int'] = round(raw_data['Reviewer_Score']).astype(int)

# Look at the results

print(f'Rating categories after rounding {raw_data["Rating Int"].value_counts().shape[0]}')

print(raw_data['Rating Int'].value_counts())

```

    Rating categories in raw data 37
    Rating categories after rounding 9
    10    187744
    8     110155
    9     105722
    7      44088
    6      27800
    5      24188
    4       9436
    3       4406
    2       2199
    Name: Rating Int, dtype: int64
    

Generally data is expected to follow a normal distibution. In the case of reviews this is should not be exected since the motivation to leave a review may come an outlier experience. Given this I would expect to see reviews clustered around the high and low end of the scale. 

The actual distibution shows the reviews are tend to be on the higher side of the scale with ratings of 8 - 10.

This raises issues since any model we train will be biased toward a higher rating because our sample population consists of higher ratings. The model will have more information about what makes a good review than what makes a bad review. 



```python
#plot the distibution of ratings

plt.figure()
plt.hist(raw_data['Rating Int'])
plt.title('Distribution of Review Score')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.show()
```


    
![png](/Kibat_NLP_PART_1_files/Kibat_NLP_PART_1_13_0.png)
    

Here is the breakdown of numeric and non numeric fields. `days_since_review` and `Review_date` can easily be changed to numeric columns. `Reviewer_Nationality` could be converted using a one hot encoding. `Tags`, `Positive_Review` and `Negative_Review` could be coverted using count vecortizors.

**Numeric**
- Additional_Number_of_Scoring               
- Average_Score                               
- Review_Total_Negative_Word_Counts             
- Total_Number_of_Reviews                       
- Review_Total_Positive_Word_Counts             
- Total_Number_of_Reviews_Reviewer_Has_Given  
- Reviewer_Score                              
- lat                                         
- lng                                        
- Rating Int                                  

**Non-numeric**
- Hotel_Address
- Review_Date 
- Hotel_Name
- Reviewer_Nationality                        
- Negative_Review                             
- Positive_Review                            
- Tags                                        
- days_since_review      





```python
# Look at the number of countries represented 
raw_data['Reviewer_Nationality'].value_counts()
```




     United Kingdom               245246
     United States of America      35437
     Australia                     21686
     Ireland                       14827
     United Arab Emirates          10235
                                   ...  
     Cape Verde                        1
     Northern Mariana Islands          1
     Tuvalu                            1
     Guinea                            1
     Palau                             1
    Name: Reviewer_Nationality, Length: 227, dtype: int64



To make this practice easier on my laptop's CPU I'm going to sample the dataset. I sample the data and then verfiy the mix of my target classes matches between the orginal and sample set. 

```python
# create a new dataframe that is a sample of the entire data set
smaller_df = raw_data.sample(frac=.1, random_state=40)

## reindex the new dataframe
smaller_df = smaller_df.reset_index(drop=True)

#check the shapes to confrim a success 
print(f'Orginal data shape{raw_data.shape}')
print(f'Sampled data shape {smaller_df.shape}')
print(f'The sample dataframe has {round(smaller_df.shape[0]/raw_data.shape[0]*100,0)}% of the rows as the original df.')

```

    Orginal data shape(515738, 18)
    Sampled data shape (51574, 18)
    The sample dataframe has 10.0% of the rows as the original df.
    


```python
# Confrim proprtionate class representation between the orginal and new dataframes

# Create a df to store the class mixes
validation_df = pd.DataFrame()

# calculate the class mixes and add them to the new df
validation_df['Raw mix'] = round(raw_data.value_counts('Rating Int', normalize=True)*100,0)
validation_df['sample mix'] = round(smaller_df.value_counts('Rating Int', normalize=True)*100,0)

# show the new df
validation_df.sort_values(by=['Rating Int'])
```




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
      <th>Raw mix</th>
      <th>sample mix</th>
    </tr>
    <tr>
      <th>Rating Int</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>9.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>21.0</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>20.0</td>
      <td>21.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>36.0</td>
      <td>36.0</td>
    </tr>
  </tbody>
</table>
</div>



The unusual distibution of the review score will make modeling difficult. With the distibuation skewing toward the high and low end of the scale I can simplify that rating further into a good/bad classification. This leaves me with a dataset evenly distibuted between the two classes. 


```python
# Create a list of good scores
good_scores = [9,10]

# replace ratings with a binary 
smaller_df['Rating Int'] = np.where(smaller_df['Rating Int'].isin(good_scores),1,0)

```


```python
# look at the breakdown
smaller_df['Rating Int'].value_counts()
```

    1    29299
    0    22275
    Name: Rating Int, dtype: int64


Earlier it was noted `days_since_review` and `Review_date` can easily be changed to numeric columns. `Reviewer_Nationality` and  `Tags` could be transfromed as well. Using nationality as a predicitve feature is something that could add bias to the model. `Tags` seem to be more helpful in finding reviews on a certain topic than saying anything about the qaulity of the experience. As they are more descriptive than predeictive they will be dropped and the focus for classification will be on the review text. 

`days_since_review` has a number along with the word 'day' or 'days' with it. This can easily converted using `srt.split`.

`Review date` can be coverted using date and time features of pandas.




```python
## remove separate the number and text in the `days_since_review` columns and store the number chars as int

smaller_df['days_since_review'] = smaller_df['days_since_review'].str.split(' ').str[0].astype('int')
```


```python
# Look at the results
smaller_df['days_since_review'].value_counts()
```

    120    243
    1      243
    322    225
    338    203
    534    196
          ... 
    122     16
    615     16
    124     15
    121     13
    123     12
    Name: days_since_review, Length: 731, dtype: int64

```python
## convert the dates since last review to a dat time columns

smaller_df['Review_Date'] = pd.to_datetime(smaller_df['Review_Date'])
```


```python
# Look at the results

smaller_df['Review_Date'][:5]
```
    0   2017-02-17
    1   2016-02-16
    2   2016-07-09
    3   2015-12-07
    4   2017-05-14
    Name: Review_Date, dtype: datetime64[ns]

Now, I can drop the other columns that I won't use in modeling. I'm saving the reviews to breakdown next. 


```python
## drop other object columns

# get a list of object cols
object_cols_to_drop = list(smaller_df.select_dtypes(include='object').columns)

# remove the cols I want to keep
object_cols_to_drop.remove('Negative_Review')
object_cols_to_drop.remove('Positive_Review')

# drop the remaining object cols
smaller_df.drop(object_cols_to_drop, axis=1, inplace= True)
```


```python
# look at the remaining columns

smaller_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 51574 entries, 0 to 51573
    Data columns (total 14 columns):
     #   Column                                      Non-Null Count  Dtype         
    ---  ------                                      --------------  -----         
     0   Additional_Number_of_Scoring                51574 non-null  int64         
     1   Review_Date                                 51574 non-null  datetime64[ns]
     2   Average_Score                               51574 non-null  float64       
     3   Negative_Review                             51574 non-null  object        
     4   Review_Total_Negative_Word_Counts           51574 non-null  int64         
     5   Total_Number_of_Reviews                     51574 non-null  int64         
     6   Positive_Review                             51574 non-null  object        
     7   Review_Total_Positive_Word_Counts           51574 non-null  int64         
     8   Total_Number_of_Reviews_Reviewer_Has_Given  51574 non-null  int64         
     9   Reviewer_Score                              51574 non-null  float64       
     10  days_since_review                           51574 non-null  int32         
     11  lat                                         51244 non-null  float64       
     12  lng                                         51244 non-null  float64       
     13  Rating Int                                  51574 non-null  int32         
    dtypes: datetime64[ns](1), float64(4), int32(2), int64(5), object(2)
    memory usage: 5.1+ MB
    

Spliting the dataset into test and training sets in important when modeling. It provides an opportunity to look evaluate the perfromance of the model against labeled data it has not yet encountered. 


```python
# Separate the target column from the data column

X = smaller_df.drop('Rating Int',axis=1)
y = smaller_df['Rating Int']

#set up the test and train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

```

## Vectorization

Since the data has punctuation already removed the tokenization step of vecotorization is not needed. It is still helpful to remove the stopwords and set the `min_df` parameter. `min_df` is the parameter that determines the theshold for including a string in the bag of words. Strings that appear in fewer than the parameter speficies are excluded from the bag. This parameter was adjusted until both positive and negative reviews yielded 10 words. 

Once the feature words are selected and a sparse matix was formed for each, they were added to the main datafrme. The last step was to remove duplicate columns. In this case duplicate columns are those words that apppeared in both postive reviews and negative reviews. These words being ambiguious will not add anything of value to the model. 


```python

# Create the bag of words for positive reviews
positive_bagofwords = CountVectorizer(min_df = 15, stop_words='english')
positive_bagofwords.fit(X_train['Positive_Review'])

# transfrom the bag of words to a sparse matrix
positive_bagofwords_transfromed = positive_bagofwords.transform(X_train['Positive_Review'])

#Count the number of words the bag of words categorized

len(positive_bagofwords.get_feature_names_out())
```




    1805




```python
# Create a user friendly df from the bag of words and tag them as positive
positive_sparse_df = pd.DataFrame(columns=positive_bagofwords.get_feature_names_out(), data=positive_bagofwords_transfromed.toarray()).add_prefix('p_',)
display(positive_sparse_df)
```


<!-- <div>
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
      <th>p_00</th>
      <th>p_02</th>
      <th>p_10</th>
      <th>p_100</th>
      <th>p_100m</th>
      <th>p_10min</th>
      <th>p_10th</th>
      <th>p_11</th>
      <th>p_12</th>
      <th>p_13</th>
      <th>...</th>
      <th>p_wouldn</th>
      <th>p_wow</th>
      <th>p_wrong</th>
      <th>p_yards</th>
      <th>p_year</th>
      <th>p_years</th>
      <th>p_yes</th>
      <th>p_yogurt</th>
      <th>p_young</th>
      <th>p_yummy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>41254</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>41255</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>41256</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>41257</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>41258</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>41259 rows × 1805 columns</p>
</div> -->



```python
#create a sprase matrix for the negative reviews

negative_bagofwords = CountVectorizer(min_df = 25, stop_words='english')
negative_bagofwords.fit(X_train['Negative_Review'])

# transfrom the bag of words to a sparse matrix
negative_bagofwords_transfromed = negative_bagofwords.transform(X_train['Negative_Review'])

#Count the number of words the bag of words categorized
len(negative_bagofwords.get_feature_names_out())
```




    1640




```python
# Create a user friendly df from the bag of words
negative_sparse_df = pd.DataFrame(columns=negative_bagofwords.get_feature_names_out(), data=negative_bagofwords_transfromed.toarray()).add_prefix('n_')
display(negative_sparse_df)
```


<!-- <div>
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
      <th>n_00</th>
      <th>n_10</th>
      <th>n_100</th>
      <th>n_10pm</th>
      <th>n_11</th>
      <th>n_11pm</th>
      <th>n_12</th>
      <th>n_13</th>
      <th>n_14</th>
      <th>n_15</th>
      <th>...</th>
      <th>n_worth</th>
      <th>n_wouldn</th>
      <th>n_write</th>
      <th>n_written</th>
      <th>n_wrong</th>
      <th>n_year</th>
      <th>n_years</th>
      <th>n_yes</th>
      <th>n_young</th>
      <th>n_zero</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>41254</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>41255</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>41256</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>41257</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>41258</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>41259 rows × 1640 columns</p>
</div> -->



```python
# look at the lists of words chosen 

print(f'Negative bag of words {negative_bagofwords.get_feature_names_out()}')
print(f'Positive bag of words {positive_bagofwords.get_feature_names_out()}')
```

    Negative bag of words ['00' '10' '100' ... 'yes' 'young' 'zero']
    Positive bag of words ['00' '02' '10' ... 'yogurt' 'young' 'yummy']
    
Now that I have the useful columns isolated and both of the review columns broken down into sparse matrices I can create a datafame ready for machine learning modeling. 

```python
# combine the three dataframes
df = pd.concat([smaller_df,positive_sparse_df, negative_sparse_df], axis=1)
```


```python
# Look for duplicate columns
duplicate_col = df.columns[df.columns.duplicated()]
print(duplicate_col)
```

    Index([], dtype='object')
    


```python
df.shape
```

    (51574, 3459)

With this process complete I will follow up with another post following the process I used to model this data and compare a few different modeling methods. 
