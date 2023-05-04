#!/usr/bin/env python
# coding: utf-8

# <div style="border:solid black 2px; padding: 20px">
# </div>

# Background:
# 
# The project background is to identify patterns that determine whether a video game can be considered successful or not. The project is for a virtual store named "Ice," which sells video games from all over the world. Data related to user and expert reviews, genre, platform (e.g., Xbox or PlayStation), and historical sales data are available from open sources. The goal is to find the most promising games and plan a marketing campaign for 2017 based on the data from 2016.
# 
# The dataset contains abbreviations, such as ESRB, which stands for Entertainment Software Rating Board, an independent regulatory organization that evaluates game content and assigns age ratings such as Teen or Mature. The project aims to utilize this information and other relevant data to identify the patterns that make a video game successful and leverage these insights to improve the store's marketing strategy.

# <div class="alert alert-success">
# <b>Chamdani's comment v.1</b> <a class="tocSkip"></a>
# 
# Bagus, semua berjalan lancar.
# 
# </div>

# # Importing The Data

# In[206]:


import pandas as pd


# In[207]:


import numpy as np


# In[208]:


import matplotlib.pyplot as plt


# In[209]:


import seaborn as sns


# In[210]:


from scipy import stats as st


# In[211]:


data = pd.read_csv("/datasets/games.csv")


# In[212]:


data


# <div class="alert alert-success">
# <b>Chamdani's comment v.1</b> <a class="tocSkip"></a>
# 
# Bagus, semua berjalan lancar.
# 
# </div>

# # Preparing The Data

# Check data type and number of columns X rows.

# In[213]:


data.info()


# Change the column names into lower strings.

# In[214]:


data.rename(columns=str.lower, inplace=True)


# In[215]:


data


# In[216]:


data.isna().sum()


# From above summary, what we could do with the missing values:
# 1. name & genre = remain as the original value or drop since we will not be able to make up the name of the respective 2 games.
# 2. year_of_release = we can try to match the name of the game and fill in manually.
# 3. cirtic_score, user_score & rating = we can use the algorithm based on the number of general sales.
# The missing values in user_score & rating is most likely intentional (no rating and user/critic score are given).

# <div class="alert alert-success">
# <b>Chamdani's comment v.1</b> <a class="tocSkip"></a>
# 
# Bagus, semua berjalan lancar.
# 
# </div>

# In[217]:


data['name'].unique()


# In[218]:


data['year_of_release'].unique()


# Convert year_of_release and critic_score to integer, and user_score to floating.
# We do not convert sales number and user_score to integer since they contains decimal points.

# <div class="alert alert-success">
# <b>Chamdani's comment v.1</b> <a class="tocSkip"></a>
# 
# Bagus, semua berjalan lancar.
# 
# </div>

# In[219]:


data['year_of_release'] = data['year_of_release'].fillna(0).astype(int)


# In[220]:


data['critic_score'] = data['critic_score'].fillna(0).astype(int)


# Replace "tbd" value with NaN

# In[221]:


data['user_score'] = data['user_score'].replace('tbd', np.nan)
data['user_score'] = data['user_score'].astype(float)


# In[222]:


data.info()


# <div class="alert alert-success">
# <b>Chamdani's comment v.1</b> <a class="tocSkip"></a>
# 
# Bagus, semua berjalan lancar.
# 
# </div>

# # Analyzing The Data

# In[223]:


data.describe()


# In[224]:


year_grouping = data.groupby('year_of_release')['name'].count()
year_grouping 


# We can analyze the data using histogram by excluding the missing values.

# <div class="alert alert-success">
# <b>Chamdani's comment v.1</b> <a class="tocSkip"></a>
# 
# Bagus, semua berjalan lancar.
# 
# </div>

# In[225]:


plt.bar(year_grouping.index, year_grouping.values)
plt.xlim(1980,2016)


# From above histogram we could conclude that the production of games we rapidly increased during 2000s and reached its peak in 2008-2009.

# <div class="alert alert-success">
# <b>Chamdani's comment v.1</b> <a class="tocSkip"></a>
# 
# Bagus, semua berjalan lancar.
# 
# </div>

# In[226]:


data['total_sales'] = data[['na_sales', 'eu_sales', 'jp_sales', 'other_sales']].sum(axis=1)
data


# In[227]:


platform_grouping = data.groupby(['platform'])[['total_sales']].sum()
platform_grouping.sort_values(by='total_sales', ascending=False)


# Data Timeframe:
# 
# Since the main objective of this project is to determine the most compelling game for the marketing campaign in 2017, it would be best to set the timeframe period from 2011 onwards.
# The main reason we choose 2011 onwards timeframe is because the declining trend since the peak (2008-2009) has started to stabilize during 2011-2015. Given that period we would be able to find a pattern to predict 2017 trend in the market.

# <div class="alert alert-success">
# <b>Chamdani's comment v.1</b> <a class="tocSkip"></a>
# 
# Bagus, semua berjalan lancar.
# 
# </div>

# In[228]:


filtered_data = data[data['year_of_release'].between(2011,2016)]
filtered_data


# In[229]:


platform_grouping = filtered_data.groupby(['platform'])[['total_sales']].sum()
platform_grouping.sort_values(by='total_sales', ascending=False)


# In[230]:


platform_grouping = filtered_data.groupby(['platform','year_of_release'])[['total_sales']].sum()
platform_grouping.sort_values(by='year_of_release', ascending=False)


# From above platform data, we can conclude that:
# 1. For the last 3 years (2014-2016), PS4 was the most used game platform and followed by XOne
# 2. Before 2014, PS3 and 3Ds were more demanded.
# 
# Seeing above data, we can narrow the data timeframe further to 2014-2016 and focus on the top 5 platforms.

# <div class="alert alert-success">
# <b>Chamdani's comment v.1</b> <a class="tocSkip"></a>
# 
# Bagus, semua berjalan lancar.
# 
# </div>

# In[231]:


filtered_data = data[(data['year_of_release'].between(2014, 2016)) & (data['platform'].isin(['PS4', 'XOne', '3DS', 'PS3', 'X360','PC']))]

filtered_data


# In[232]:


sales_by_platform = filtered_data.groupby(['platform'])['total_sales'].sum()
sales_by_platform.sort_values(ascending=False)


# In[233]:


plt.subplots(figsize=(15,8))
sns.boxplot(x='platform', y='total_sales', data=filtered_data)


plt.xlabel('Platform')
plt.ylabel('Total Sales (in millions)')
plt.title('Total Sales per Platform (2014-2016)')

plt.show()


# Above boxplot shows that there are a lot of values above the upper whisker of the boxplot, this indicates that there are many data points that are larger than the typical range of values in the dataset. In other words, these data points are outliers that fall outside the range of values that are typically seen in the data.
# 
# There are a few possible reasons:
# 1. There are a few games that are extremely successful and have sold many more copies than other games in the same platform. 2. Another possibility is that the data is skewed and does not follow a normal distribution, which can cause the boxplot to be compressed and the whiskers to be shorter, making it easier for data points to fall outside the whiskers.
# 
# We would need to examine the data in more detail to determine the cause of the outliers and whether they are genuine data points or errors in the data. 

# <div class="alert alert-success">
# <b>Chamdani's comment v.1</b> <a class="tocSkip"></a>
# 
# Bagus, semua berjalan lancar.
# 
# </div>

# Sample check on the same game sales in different platform:

# In[234]:


sales_by_game_platform = filtered_data.groupby(['name', 'platform'])['total_sales'].sum()
sales_by_game_platform= sales_by_game_platform.sort_values(ascending=False)
sales_by_game_platform.head(15)


# From above data we could see the major difference of a game total_sales in each platform (for example: 'Call of Duty: Black Ops and 'Grand Theft Auto V'
# The differences are quite significant and could be double in value.

# In[235]:


game_by_genre = filtered_data.groupby('genre')['total_sales'].sum()
game_by_genre = game_by_genre.sort_values(ascending=False)
game_by_genre


# From above data, we can conclude that action, shooter and sports are the most popular game genre in the market.

# <div class="alert alert-success">
# <b>Chamdani's comment v.1</b> <a class="tocSkip"></a>
# 
# Bagus, semua berjalan lancar.
# 
# </div>

# # Profiling the data

# In[236]:


region_grouping = filtered_data.groupby('platform')[['na_sales', 'eu_sales', 'jp_sales', 'other_sales','total_sales']].sum()
region_grouping = region_grouping.sort_values(by='total_sales', ascending=False)
region_grouping


# From above table, the most popular platforms in each area would be=
# 1. na_sales = PS4 and XOne 
# 2. eu_sales = PS4 and XOne
# 3. jp_sales = 3DS
# 4. other_sales = PS4 and XOne 
# 
# in conclusion, all market except for Japan (jp_sales) has the same popularity pattern.

# <div class="alert alert-success">
# <b>Chamdani's comment v.1</b> <a class="tocSkip"></a>
# 
# Bagus, semua berjalan lancar.
# 
# </div>

# In[237]:


genre_grouping = filtered_data.groupby('genre')[['na_sales', 'eu_sales', 'jp_sales', 'other_sales','total_sales']].sum()
genre_grouping = genre_grouping.sort_values(by='total_sales', ascending=False)
genre_grouping


# From above table, the most popular genre in each area would be=
# 1. na_sales = Action, Shooter and Sports
# 2. eu_sales = Action, Shooter and Sports
# 3. jp_sales = Action and Role Playing
# 4. other_sales = Action, Shooter and Sports
# 
# in conclusion, all market except for Japan (jp_sales) has the same genre popularity pattern.

# <div class="alert alert-success">
# <b>Chamdani's comment v.1</b> <a class="tocSkip"></a>
# 
# Bagus, semua berjalan lancar.
# 
# </div>

# In[238]:


region_rating_grouping = filtered_data.groupby(['rating', 'platform'])
total_sales_by_region_rating = region_rating_grouping[['na_sales', 'eu_sales', 'jp_sales', 'other_sales', 'total_sales']].sum()
total_sales_by_region_rating


# 1. The PS4 platform is the most popular platform for games with an M rating, as it has sold significantly more units in North America, Europe, and other regions compared to other platforms.
# 2. Games with an E rating are generally more popular on the 3DS and PS4 platforms, while games with an E10+ rating are more popular on the PS4 and XOne platforms.
# 3. The 3DS platform appears to be more popular for games with a T rating, particularly in Japan where it has sold significantly more units compared to other platforms.
# 4. Games with an M rating have sold significantly more units in North America and Europe compared to Japan and other regions, while games with an E rating have sold more units in Japan compared to other regions.
# 5. The XOne platform appears to be less popular overall compared to the other platforms, as it has sold fewer units for each rating group and in each region compared to other platforms.

# <div class="alert alert-success">
# <b>Chamdani's comment v.1</b> <a class="tocSkip"></a>
# 
# Bagus, semua berjalan lancar.
# 
# </div>

# # Hypothesis Testing

# There are two hypotheses that we would like to test:
# 1. The average user rating for Xbox One and PC platforms is the same.
# 2. The average user rating for Action and Sports genres is different.
# 
# To test these hypotheses, we can use a two-sample t-test to compare the means of two groups (in the case of the first hypothesis) and a two-sample t-test 

# ### The average user rating for Xbox One and PC platforms is the same.

# Null hypothesis: The mean user rating for Xbox One and PC platforms are equal.
# Alternative hypothesis: The mean user rating for Xbox One and PC platforms are not equal.

# In[239]:


average_user_rating_XOne = filtered_data[filtered_data['platform'] == 'XOne']['user_score'].mean()

average_user_rating_XOne


# In[242]:


average_user_rating_PC = filtered_data[filtered_data['platform'] == 'PC']['user_score'].mean()

average_user_rating_PC


# In[256]:



alpha = 0.05

results = st.ttest_ind(average_user_rating_XOne , average_user_rating_PC, nan_policy='omit')

print('p-value: ', results.pvalue)

if results.pvalue < alpha:
    print("Kita menolak hipotesis nol")
else:
    print("Kita tidak dapat menolak hipotesis nol") 


# <div class="alert alert-success">
# <b>Chamdani's comment v.1</b> <a class="tocSkip"></a>
# 
# Bagus, semua berjalan lancar.
# 
# </div>

# ###  The average user rating for Action and Sports genres is different.

# Null hypothesis: The mean user rating for Action and Sports genres are equal.
# Alternative hypothesis: The mean user rating for Action and Sports genres are not equal.

# In[246]:


average_user_rating_action = filtered_data[filtered_data['genre'] == 'Action']['user_score'].mean()

average_user_rating_action


# In[247]:


average_user_rating_sports = filtered_data[filtered_data['genre'] == 'Sports']['user_score'].mean()

average_user_rating_sports


# In[255]:



alpha = 0.05

results = st.ttest_ind(average_user_rating_action , average_user_rating_sports,  nan_policy='omit')

print('p-value: ', results.pvalue)

if results.pvalue < alpha:
    print("Kita menolak hipotesis nol")
else:
    print("Kita tidak dapat menolak hipotesis nol") 


# <div class="alert alert-success">
# <b>Chamdani's comment v.1</b> <a class="tocSkip"></a>
# 
# Bagus, semua berjalan lancar.
# 
# </div>

# # General Conclusion

# The overall marketig strategy for 2017 based on above analysis:
# 1. Target the PS4 and XOne platforms: Based on the data, the PS4 and XOne platforms are the most popular platforms for games. Therefore, targeting these platforms could be a good way to reach a large audience.
# 
# 2. Focus on the most popular game genres: The data shows that action, shooter, and sports are the most popular game genres. Therefore, focusing on these genres in marketing efforts could be a good way to appeal to the target audience.
# 
# 3. Highlight games with high ratings: The data shows that games with an M rating are the most popular in North America and Europe. Therefore, highlighting games with high ratings could be a good way to appeal to this audience.
# 
# 4. Advertise in North America and Europe: The data shows that North America and Europe are the largest markets for video games. Therefore, focusing marketing efforts in these regions could be a good way to reach a large audience

# <div class="alert alert-success">
# <b>Chamdani's comment v.1</b> <a class="tocSkip"></a>
# 
# Bagus, semua berjalan lancar.
# 
# </div>
