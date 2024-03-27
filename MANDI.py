# Example import structure and basic cleaning of Kaggle dataset
import pandas as pd
# Create Pandas DF from cannabis.csv
df = pd.read_csv("cannabis.csv")
# Remove rows with missing values
df = df.dropna()
df.head()

import re
# Create a list of text features
df_features = ['Strain', 'Description', 'Effects', 'Flavor', 'Type']
# Lowercase and remove symbols
for each in df_features:
 df[each] = df[each].apply(lambda x: x.lower())
 df[each] = df[each].apply(lambda x: re.sub('[^a-zA-Z 0-9]', ' ',x))
 #print(df_features)
# Combine text features into new feature
df['combined_text'] = df['Type'] + ' ' + df['Effects'] + ' ' + df['Flavor'] + df['Description']
#print(df['combined_text'])


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
df['Effects'] = df['Effects'].fillna("")
tfidf_matrix = tfidf.fit_transform(df['Effects'])

from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df.index, index = df['Strain']).drop_duplicates()
#print(indices)

#print(indices['100 og'])

def get_recommendations(Strain, cosine_sim = cosine_sim):
 idx = (indices[Strain])
 sim_score = enumerate(cosine_sim[idx])
 sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)
 sim_score = sim_score[1:11]
 for i in sim_score:
  print(i)
  sim_index = [i[0] for i in sim_score]
  print(sim_index)
 print(df['Strain'].iloc[sim_index])

get_recommendations('drizella')