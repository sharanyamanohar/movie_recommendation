import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD,Reader,Dataset
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_csv(r'C:\Users\Admin\Desktop\movies.csv')
ratings = pd.read_csv(r'C:\Users\Admin\Desktop\ratings2.csv')
links = pd.read_csv(r'C:\Users\Admin\Desktop\Movie_recommendation system\data\links_small.csv')
links.dropna(subset='tmdbId',inplace=True)
ratings = pd.read_csv(r'C:\Users\Admin\Desktop\Movie_recommendation system\data\ratings_small.csv')

st.title('      INTERNSHIP FINAL PROJECT')
st.header(' 🎥 MOVIE RECOMMENDATION SYSTEMS  🎥')
st.write('Select dataset info for all the basic information regarding the dataset')
st.write('Select eda to get the visualizations')
st.write('Select get recommendations to get the top 10 movies recommended for you')

x = st.selectbox(label="",options= (['DATASET INFO','EDA','GET RECOMMENDATIONS']))


def get_poster_url(id):
    API_key = "a9940390778cc2fd7f3ee153bcec4d99"
    URL = f"https://api.themoviedb.org/3/movie/{id}?api_key=a9940390778cc2fd7f3ee153bcec4d99"
    PosterDB = requests.get(URL)
    todos = json.loads(PosterDB.text)
    path = todos['poster_path']
    url_to_poster = 'https://image.tmdb.org/t/p/w500' + path
    return url_to_poster

if x == 'DATASET INFO':
    z = st.radio(label="", options=('MOVIES DATA INFO', 'MOVIES_RATINGS INFO'))
    if z == 'MOVIES DATA INFO':
        st.write('Here is the First 5 rows of movies dataset')
        st.dataframe(df.head())

        st.write('Here is the Last 5 rows of movies dataset')
        st.dataframe(df.tail())

        st.write('Here is the description of movies dataset')
        st.dataframe(df.describe())

        st.write('Here is the shape of our movies dataset')
        st.write('The number of rows in the movies dataset are')
        st.write(df.shape[0])
        st.write('The number of columns in the movies dataset are')
        st.write(df.shape[1])

        st.write('Here are the columns of movies dataset')
        st.write(df.columns)

        st.write('Here is the info of movies dataset')
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()

        st.text(s)

        st.write('You can find the link for the movies dataset  https://www.kaggle.com/rounakbanik/movie-recommender-systems/data')

    elif z == 'MOVIES_RATINGS INFO':
        st.write('Here is First 5 rows of movies_ratings dataset')
        st.dataframe(ratings.head())

        st.write('Here is Last 5 rows of movies_ratings dataset')
        st.dataframe(ratings.tail())

        st.write('Here is the description of movies_ratings dataset')
        st.dataframe(ratings.describe())

        st.write('Here is the shape of our movies_ratings dataset')
        st.write('The number of rows in the movies_ratings dataset are')
        st.write(ratings.shape[0])
        st.write('The number of columns in the movies_ratings dataset are')
        st.write(ratings.shape[1])

        st.write('The columns of movies_ratings dataset are ')
        st.write(ratings.columns)

        st.write('Here is the info of movies_ratings dataset')
        buffer = io.StringIO()
        ratings.info(buf=buffer)
        s = buffer.getvalue()

        st.text(s)

        st.write(
            'You can find the link for the movies_ratings dataset  https://www.kaggle.com/rounakbanik/movie-recommender-systems/data')


elif x=='EDA':
    df = pd.read_csv(r'C:\Users\Admin\Desktop\movies.csv')
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    ratings = pd.read_csv(r'C:\Users\Admin\Desktop\ratings2.csv')
    submit = st.button(label='Get Visualizations')
    if submit:
        fig = plt.figure(figsize=(8, 5))
        sns.distplot(df['budget'])
        plt.title('Budget', weight='bold')
        st.pyplot(fig)
        st.header("The distribution of movie budgets shows an exponential decay.")

        st.title('Genres-Wordcloud')
        image = Image.open(r'C:\Users\Admin\Desktop\Images\genres.png')
        st.image(image)
        st.header('As we inferentiate that most common word is Drama,Romantic,Comedy')

        st.title("Language vs count")
        lang_df = pd.DataFrame(df['original_language'].value_counts())
        lang_df['language'] = lang_df.index
        lang_df.columns = ['number', 'language']
        fig = plt.figure(figsize=(8, 5))
        sns.barplot(lang_df.iloc[0:13], y='language', x='number')
        st.pyplot(fig)
        st.header("Maximum number of language used is English as count 22299 followed by French,Italian")

        st.title("Language vs count")
        lang_df = pd.DataFrame(df['original_language'].value_counts())
        lang_df['language'] = lang_df.index
        lang_df.columns = ['number', 'language']
        fig = plt.figure(figsize=(8, 5))
        sns.barplot(lang_df.iloc[1:13], x='language', y='number')
        st.pyplot(fig)
        st.header("Maximum number of language used is English as count 22299 followed by French,Italian")

        st.title('Popularity')
        fig = plt.figure(figsize=(8, 5))
        df['popularity'].plot(logy=True, kind='hist')
        plt.xlabel('popularity')
        st.pyplot(fig)
        st.header(
            "As the popularity score it seems to be extremely right skewed data with the mean of 2.7 and maximum reaching upto 294 and the 75% percentile is at 3.493 and almost all the data below 75%")

        st.title('Overview-Wordcloud')
        image = Image.open(r'C:\Users\Admin\Desktop\Images\Overview.png')
        st.image(image)
        st.header(
            "Life is the most commonly used word in Overview,followed by 'one' and 'find' are the most Movie Blurgs.Together with Love, Man and Girl, these wordclouds give us a pretty good idea of the most popular themes present in movies.")

        st.title('Title-Wordcloud')
        image = Image.open(r'C:\Users\Admin\Desktop\Images\Title.png')
        st.image(image)
        st.header(
            "As we can see 'LOVE' the title is common in most of the Movie title followed by 'LIFE','GIRL','MAN' and 'NIGHT'")

        plt.title('Released_year vs movies', weight='bold')
        year_df = pd.DataFrame(df['release_year'].value_counts())
        year_df['year'] = year_df.index
        year_df.columns = ['number', 'year']
        fig = plt.figure(figsize=(12, 5))
        sns.barplot(x='year', y='number', data=year_df.iloc[1:20])
        st.pyplot(fig)
        st.header("By the Relaesed_year we inferetiate that Most number of movies released in 2006")

        fig = plt.figure(figsize=(8, 5))
        ax = sns.distplot(df['vote_average'])
        plt.title('Vote Average', weight='bold')
        plt.xlabel('Vote_Average', weight='bold')
        plt.ylabel('Density', weight='bold')
        st.pyplot(fig)
        st.header(
            "There is a very small correlation between Vote Count and Vote Average. A large number of votes on a particular movie does not necessarily imply that the movie is good.")

        fig = plt.figure(figsize=(8, 5))
        ax = sns.distplot(df[(df['runtime'] < 300) & (df['runtime'] > 0)]['runtime'])
        plt.title('Runtime', weight='bold')
        plt.xlabel('Runtime', weight='bold')
        plt.ylabel('Density', weight='bold')
        st.pyplot(fig)
        st.header("Here we count that runtime is less than 300 but greater than 0 ")

        fig = plt.figure(figsize=(10, 10))
        df['genres'].value_counts()[:20].plot(kind='barh')
        plt.title("Movies genres ", fontsize=20)
        plt.ylabel("movie genres", fontsize=20)
        plt.xlabel("count", fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=17)
        st.pyplot(fig)
        st.header(
            "As per the above count plot it seems there is highest no.of TV series gener are Drama as compared to the other TV series. ")

        fig = plt.figure(figsize=(15, 7))
        df['cast'].value_counts()[:20].plot(kind='barh')
        plt.title("Movies cast ", fontsize=20)
        plt.ylabel("movie cast", fontsize=20)
        plt.xlabel("Count", fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=17)
        plt.show()
        st.pyplot(fig)
        st.header("As per the above count plot of cast it seems GeorgesMeeleis has acted in many TV series")

        fig = plt.figure(figsize=(15, 7))
        df['crew'].value_counts()[:20].plot(kind='barh')
        plt.title("Movies crew ", fontsize=20)
        plt.ylabel("movies crew", fontsize=20)
        plt.xlabel("Count", fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=17)
        plt.show()
        st.pyplot(fig)
        st.header("As per the above count plot of crew it seems JohnFord has directed many TV series")

        fig = plt.figure(figsize=(20, 20))

        st.title("Histogram ")

        fig = plt.figure(figsize=(20, 20))

        for i, col in enumerate(numerical_cols[:-1]):
            plt.subplot(10, 3, i + 1)
            plt.hist(df[col])
            plt.xlabel(col)
            plt.subplots_adjust(left=0.1,
                                bottom=0.1,
                                right=0.9,
                                top=0.9,
                                wspace=0.4,
                                hspace=0.4)
        plt.show()
        st.pyplot(fig)

        st.title("distribution")
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        fig = plt.figure(figsize=(20, 20))

        for i, col in enumerate(numerical_cols[:-1]):
            plt.subplot(10, 3, i + 1)
            sns.distplot(df[col], bins=20, kde=True, rug=True)
            plt.xlabel(col)
            plt.subplots_adjust(left=0.1,
                                bottom=0.1,
                                right=0.9,
                                top=0.9,
                                wspace=0.4,
                                hspace=0.4)
        st.pyplot(fig)

        st.title("Box plot")
        fig = plt.figure(figsize=(20, 20))

        for i, col in enumerate(numerical_cols[:-1]):
            plt.subplot(10, 3, i + 1)
            sns.boxplot(df[col])
            plt.xlabel(col)
            plt.subplots_adjust(left=0.1,
                                bottom=0.1,
                                right=0.9,
                                top=0.9,
                                wspace=0.4,
                                hspace=0.4)
        plt.show()
        st.pyplot(fig)

        st.header(
            "1)Very few TV series has generated the higher revenue as shown in the histogram. 2)The Vote average of the TV series between range 3 to 9 as shown in the bar plot. 3)The Vote average column is normally distrubuted as shown in the distribution plot 4)The runtime column has right tail which means it is right skewed as per the distribution plot.")

        fig = plt.figure(figsize=(10, 10))
        year_runtime = df[df['release_year'] != 'NaT'].groupby('release_year')['runtime'].mean()
        plt.plot(year_runtime.index, year_runtime)
        plt.xticks(np.arange(1900, 2024, 10.0))
        plt.title('Runtime vs Year_trend', weight='bold')
        plt.xlabel('Year', weight='bold')
        plt.ylabel('Runtime in min', weight='bold')
        st.pyplot(fig)
        st.header(
            "As we can inference that trends go down on 1917 till 50 min and gain it increse upto 110 almost the ranges lies 90 to 110")

        st.title("Production_countries vs revenue")
        fig = plt.figure(figsize=(17, 5))
        plt.subplot(1, 2, 1)
        sns.barplot(data=df.head(20), x='revenue', y='production_countries')
        st.pyplot(fig)
        st.header(
            "From the revenue vs production countries plot United Kingdom and United States of America occupy the 1st position")

        fig = plt.figure(figsize=(10, 10))
        axis1 = sns.barplot(x=df['vote_average'].head(10), y=df['title'].head(10), data=df)
        plt.xlim(4, 10)
        plt.title('Best Movies by average votes', weight='bold')
        plt.xlabel('Weighted Average Score', weight='bold')
        plt.ylabel('Movie Title', weight='bold')
        st.pyplot(fig)
        st.header("By the vote average we inferetiate that 'Toy Story'occupied the 1st position")

        scored_df = ratings.sort_values('rating', ascending=False)
        fig = plt.figure(figsize=(10, 10))
        ax = sns.barplot(x=scored_df['rating'].head(10), y=scored_df['title'].head(10), data=scored_df, palette='deep')
        plt.title('Best Rated & Most Popular Blend', weight='bold')
        plt.xlabel('Score', weight='bold')
        plt.ylabel('Movie Title', weight='bold')
        st.pyplot(fig)
        st.header("This are the top 10 movie title recieved 5 ratings")

        fig = plt.figure(figsize=(10, 10))
        ax = sns.barplot(x=df['popularity'].head(10), y=df['title'].head(10), data=df)
        plt.title('Most Popular by votes', weight='bold')
        plt.xlabel('Score of popularity', weight='bold')
        plt.ylabel('Movie Title', weight='bold')
        st.pyplot(fig)
        st.header(
            "From the popularity based ,we inferentiate that 'Toy story'  occupied the 1st position followed by 'Heat' and 'Jumaji' respectively.")

        plt.title('Production_countries')
        fig = plt.figure(figsize=(10, 10))
        df['production_countries'].value_counts().head(10).plot(kind='pie', autopct='%1.1f%%')
        st.pyplot(fig)
        st.header("As per the pie plot it seems USA has high production rate of making TV series")

        fig = plt.figure(figsize=(10, 5))
        s = ratings.sort_values(['rating'], ascending=False)[:20]
        plt.title('top movies by average ratings')
        sns.barplot(y='title', x='rating', data=s)
        st.pyplot(fig)
        st.header('Phantom of paradise is the top rated movie')











elif x =='GET RECOMMENDATIONS':
    select = st.selectbox(label='Select the type of recommendation',options=['Popularity based recommendations','Content based recommendations','Item based Collaborative filtering','User based Collaborative filtering','Hybrid recommendations'])
    if select == 'Popularity based recommendations':
        v = df['vote_count']
        R = df['vote_average']
        C = df['vote_average'].mean()
        m = df['vote_count'].quantile(0.70)
        df['weighted_average'] = ((R * v) + (C * m)) / (v + m)
        scaler = MinMaxScaler()
        movies_scaled = scaler.fit_transform(df[['weighted_average','popularity']])
        movies_tf = pd.DataFrame(movies_scaled,columns=['weighted_average','popularity'])
        df[['weight_average_tf', 'popularity_tf']] = movies_tf
        df['score'] = df['weight_average_tf']*0.5 + df['popularity_tf']*0.5
        df_pop = df.sort_values(['score'],ascending=False)
        submit = st.button('Get Recommendations based on popularity')
        if submit:

            col1, col2, col3, col4, col5 = st.columns(5)
            col6,col7,col8,col9,col10 = st.columns(5)
            with col1:

                st.image(get_poster_url(df_pop['id'].iloc[0]),caption=df_pop['title'].iloc[0],width=150)
            with col2:

                st.image(get_poster_url(df_pop['id'].iloc[1]),caption=df_pop['title'].iloc[1],width=150)

            with col3:

                st.image(get_poster_url(df_pop['id'].iloc[2]),caption=df_pop['title'].iloc[2],width=150)
            with col4:

                st.image(get_poster_url(df_pop['id'].iloc[3]),caption=df_pop['title'].iloc[3],width=150)
            with col5:

                st.image(get_poster_url(df_pop['id'].iloc[4]),caption=df_pop['title'].iloc[4],width=150)
            with col6:

                st.image(get_poster_url(df_pop['id'].iloc[5]),caption=df_pop['title'].iloc[5],width=150)
            with col7:

                st.image(get_poster_url(df_pop['id'].iloc[6]),caption=df_pop['title'].iloc[6],width=150)
            with col8:

                st.image(get_poster_url(df_pop['id'].iloc[7]),caption=df_pop['title'].iloc[7],width=150)
            with col9:

                st.image(get_poster_url(df_pop['id'].iloc[8]),caption=df_pop['title'].iloc[8],width=150)
            with col10:

                st.image(get_poster_url(df_pop['id'].iloc[9]),caption=df_pop['title'].iloc[9],width=150)
        else:
            print('Error')



    elif select == 'Content based recommendations':
        movie_id = links[links['tmdbId'].notnull()]['tmdbId'].astype(int)
        df_c = df[df['id'].isin(movie_id)]
        df_c.dropna(inplace=True)
        df_c.drop_duplicates(subset='title', inplace=True)
        user = st.selectbox('Please select a movie to get recommendations', options=df_c['title'].tolist())
        submit = st.button('Get recommendations based on content')

        tf = TfidfVectorizer(min_df=3, max_features=None, ngram_range=(1, 6), stop_words='english', analyzer='word')
        tf_idf = tf.fit_transform(df_c['overview'])
        sigmoid = sigmoid_kernel(tf_idf, tf_idf)
        indices = pd.Series(df_c['overview'].index, index=df_c['title'])


        def get_rec(title, sigmoid=sigmoid):
            idx = indices[title]
            sig_scores = list(enumerate(sigmoid[idx]))
            sig_scores1 = sorted(sig_scores, key=lambda x: x[1], reverse=True)
            sig_scores2 = sig_scores1[1:11]
            movie_indices = [i[0] for i in sig_scores2]
            return df_c['title'].iloc[movie_indices]


        def get_id(title):
            idx = indices[title]
            sig_scores = list(enumerate(sigmoid[idx]))
            sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
            sig_scores = sig_scores[1:11]
            movie_indices = [i[0] for i in sig_scores]
            return df_c['id'].iloc[movie_indices]


        if submit:
            rec = get_rec(user)
            ids = get_id(user)
            col1, col2, col3, col4, col5 = st.columns(5)
            col6, col7, col8, col9, col10 = st.columns(5)
            with col1:
                st.image(get_poster_url(ids.iloc[0]), caption=rec.iloc[0], width=150)
            with col2:
                st.image(get_poster_url(ids.iloc[1]), caption=rec.iloc[1], width=150)
            with col3:
                st.image(get_poster_url(ids.iloc[2]), caption=rec.iloc[2], width=150)
            with col4:
                st.image(get_poster_url(ids.iloc[3]), caption=rec.iloc[3], width=150)
            with col5:
                st.image(get_poster_url(ids.iloc[4]), caption=rec.iloc[4], width=150)
            with col6:
                st.image(get_poster_url(ids.iloc[5]), caption=rec.iloc[5], width=150)
            with col7:
                st.image(get_poster_url(ids.iloc[6]), caption=rec.iloc[6], width=150)
            with col8:
                st.image(get_poster_url(ids.iloc[7]), caption=rec.iloc[7], width=150)
            with col9:
                st.image(get_poster_url(ids.iloc[8]), caption=rec.iloc[8], width=150)
            with col10:
                st.image(get_poster_url(ids.iloc[9]), caption=rec.iloc[9], width=150)


    elif select == 'Hybrid based filtering':
        movie_id = links[links['tmdbId'].notnull()]['tmdbId'].astype(int)
        df_c = df[df['id'].isin(movie_id)]
        df_c.dropna(inplace=True)
        df_c.drop_duplicates(subset='title', inplace=True)
        user = st.selectbox('Please select a movie to get recommendations', options=df_c['title'].tolist())
        user = st.selectbox('Please select a movie to get recommendations', options=df_c['movieId'].tolist())


        def convert_int(x):
            try:
                return int(x)
            except:
                return np.nan


        md['id'] = md['id'].apply(convert_int)
        md[md['id'].isnull()]
        md = md.drop([19730, 29503, 35587])
        md['id'] = md['id'].astype('int')
        smd = md[md['id'].isin(links_small)]
        id_map = pd.read_csv(r'C:\Users\Admin\Desktop\Movie_recommendation system\data\links_small.csv')[
            ['movieId', 'tmdbId']]
        id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
        id_map.columns = ['movieId', 'id']
        id_map = id_map.merge(smd[['title', 'id']], on='id').set_index('title')
        # id_map = id_map.set_index('tmdbId')
        indices_map = id_map.set_index('id')


    elif select == 'Item based Collaborative filtering':
        movies_id = ratings['movieId'].unique()
        df.dropna(inplace=True)
        df.drop_duplicates(subset='title', inplace=True)
        df_l = df[df['id'].isin(movies_id)]
        ratings1 = ratings[ratings['movieId'].isin(df['id'])]

        ids = st.selectbox('Please select a movie id',options=df_l.id.to_list())


        def create_matrix(df):
            p = len(df['movieId'].unique())
            q = len(df['userId'].unique())

            map_user = dict(zip(np.unique(df["userId"]), list(range(q))))
            map_movie = dict(zip(np.unique(df["movieId"]), list(range(p))))

            map_user_i = dict(zip(list(range(q)), np.unique(df["userId"])))
            map_mov_i = dict(zip(list(range(p)), np.unique(df["movieId"])))

            user_index = [map_user[i] for i in df['userId']]
            movie_index = [map_movie[i] for i in df['movieId']]

            matrix = csr_matrix((df["rating"], (movie_index, user_index)), shape=(p, q))

            return matrix,map_user,map_movie, map_user_i,map_mov_i


        matrix, map_user,map_movie, map_user_i,map_mov_i = create_matrix(ratings1)



        def find_similar_movies(movie_id, matrix, k, metric='cosine', show_distance=False):

            neighbour_ids = []

            movie_ind = map_movie[movie_id]
            movie_vec = matrix[movie_ind]
            k += 1
            kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
            kNN.fit(matrix)
            movie_vec = movie_vec.reshape(1, -1)
            neighbour = kNN.kneighbors(movie_vec, return_distance=show_distance)
            for i in range(0, k):
                n = neighbour.item(i)
                neighbour_ids.append(map_mov_i[n])
            neighbour_ids.pop(0)
            return neighbour_ids


        movie_titles = dict(zip(df_l['id'], df_l['title']))
        movie_ids = dict(zip(df_l['title'], df_l['id']))

        similar_ids = find_similar_movies(ids, matrix, k=10)
        movie_title = movie_titles[ids]
        l1=[]
        l2=[]
        for i in similar_ids:
            l1.append(movie_titles[i])
        for i in l1:
            l2.append(movie_ids[i])

        submit = st.button('Get collaborative filtered recommendations')
        if submit:
            st.write(f"Since you watched {movie_title}")
            st.write(f"Following are the top ten recommendations for you")
            col1, col2, col3, col4, col5 = st.columns(5)
            col6, col7, col8, col9, col10 = st.columns(5)
            with col1:
                st.image(get_poster_url(l2[0]),caption=l1[0],width=150)
            with col2:
                st.image(get_poster_url(l2[1]),caption=l1[1],width=150)
            with col3:

                st.image(get_poster_url(l2[2]),caption=l1[2],width=150)
            with col4:

                st.image(get_poster_url(l2[3]),caption=l1[3],width=150)
            with col5:

                st.image(get_poster_url(l2[4]),caption=l1[4],width=150)
            with col6:

                st.image(get_poster_url(l2[5]),caption=l1[5],width=150)
            with col7:

                st.image(get_poster_url(l2[6]),caption=l1[6],width=150)
            with col8:

                st.image(get_poster_url(l2[7]),caption=l1[7],width=150)
            with col9:

                st.image(get_poster_url(l2[8]),caption=l1[8],width=150)
            with col10:

                st.image(get_poster_url(l2[9]),caption=l1[9],width=150)


    elif select=='User based Collaborative filtering':
        movies_id = ratings['movieId'].unique()
        df_l = df[df['id'].isin(movies_id)]
        ratings1 = ratings[ratings['movieId'].isin(df['id'])]
        ids = st.selectbox('Please select a user id', options=ratings1['userId'].unique())
        rating_matrix = ratings1.pivot_table(index='userId', columns='movieId', values='rating')
        rating_matrix = rating_matrix.fillna(0)


        def sim(user_id, r_matrix, k=10):
            user = r_matrix[r_matrix.index == user_id]
            other_users = r_matrix[r_matrix.index != user_id]
            sim = cosine_similarity(user, other_users)[0].tolist()
            idx = other_users.index.to_list()
            idx_sim = dict(zip(idx, sim))
            idx_sim_sorted = sorted(idx_sim.items(), key=lambda x: x[1])
            idx_sim_sorted.reverse()
            top_user_similarities = idx_sim_sorted[:10]
            users = [i[0] for i in top_user_similarities]
            return users

        s = sim(ids,rating_matrix)


        def recommend_movie(user_index, similar_user_indices, r_matrix, items=10):
            similar_users = r_matrix[r_matrix.index.isin(similar_user_indices)]
            similar_users = similar_users.mean(axis=0)
            similar_df = pd.DataFrame(similar_users, columns=['mean'])
            user_df = r_matrix[r_matrix.index == user_index]
            user_df_transposed = user_df.transpose()
            user_df_transposed.columns = ['rating']
            user_df_transposed = user_df_transposed[user_df_transposed['rating'] == 0]
            movies_unseen = user_df_transposed.index.tolist()
            similar_users_filtered = similar_df[similar_df.index.isin(movies_unseen)]
            similar_users_ordered = similar_df.sort_values(by=['mean'], ascending=False)

            top_movies = similar_users_ordered.head(items)
            top_movie_indices = top_movies.index.tolist()
            movie_title = df_l[df_l['id'].isin(top_movie_indices)]['title']
            movie_id = df_l[df_l['id'].isin(top_movie_indices)]['id']

            return list(zip(movie_title, movie_id))

        z = recommend_movie(ids,s,rating_matrix)

        submit = st.button('Get recommendations')
        if submit:
            st.write(f"Following are the top ten recommendations for you")
            col1, col2, col3, col4, col5 = st.columns(5)
            col6, col7, col8, col9, col10 = st.columns(5)
            with col1:
                st.image(get_poster_url(z[0][1]), caption=z[0][0], width=150)
            with col2:
                st.image(get_poster_url(z[1][1]), caption=z[1][0], width=150)
            with col3:
                st.image(get_poster_url(z[2][1]), caption=z[2][0], width=150)
            with col4:
                st.image(get_poster_url(z[3][1]), caption=z[3][0], width=150)
            with col5:
                st.image(get_poster_url(z[4][1]), caption=z[4][0], width=150)
            with col6:
                st.image(get_poster_url(z[5][1]), caption=z[5][0], width=150)
            with col7:
                st.image(get_poster_url(z[6][1]), caption=z[6][0], width=150)
            with col8:
                st.image(get_poster_url(z[7][1]), caption=z[7][0], width=150)
            with col9:
                st.image(get_poster_url(z[8][1]), caption=z[8][0], width=150)
            with col10:
                st.image(get_poster_url(z[9][1]), caption=z[9][0], width=150)



    elif select=='Hybrid recommendations':
        links = pd.read_csv(r"C:\Users\Admin\Desktop\Movie_recommendation system\data\links_small.csv")
        df = pd.read_csv(r"C:\Users\Admin\Desktop\movies.csv")
        ratings = pd.read_csv(r"C:\Users\Admin\Desktop\Movie_recommendation system\data\ratings_small.csv")
        svd = SVD()
        reader = Reader()
        data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
        trainset = data.build_full_trainset()
        svd.fit(trainset)


        def convert_int(x):
            try:
                return int(x)
            except:
                return np.nan


        id_map = links[['movieId', 'tmdbId']]
        id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
        id_map.columns = ['movieId', 'id']
        id_map = id_map.merge(df[['title', 'id']], on='id').set_index('title')
        indices_map = id_map.set_index('id')

        movie_id = links[links['tmdbId'].notnull()]['tmdbId'].astype(int)
        df_c = df[df['id'].isin(movie_id)]
        df_c.dropna(inplace=True)
        df_c.drop_duplicates(subset='title', inplace=True)
        # user = st.selectbox('Please select a movie to get recommendations',options=df_c['title'].tolist())
        # submit = st.button('Get recommendations based on content')
        from nltk import WordNetLemmatizer

        lemma = WordNetLemmatizer()


        def lemmatize_text(text):
            return [lemma.lemmatize(text)]


        df_c['overview'] = df_c.overview.apply(lemmatize_text)
        df_c['overview'] = df_c['overview'].apply(lambda x: ' '.join(x))
        tf = TfidfVectorizer(min_df=3, max_features=None, ngram_range=(1, 6), stop_words='english', analyzer='word')
        tf_idf = tf.fit_transform(df_c['overview'])
        sigmoid = sigmoid_kernel(tf_idf, tf_idf)
        indices = pd.Series(df_c['overview'].index, index=df_c['title'])

        userid =st.selectbox(label='Userid',options=ratings.userId.unique())
        movie = st.selectbox(label='movie name',options=df_c.title.to_list())


        def hybrid(userId, title):
            idx = indices[title]
            tmdbId = id_map.loc[title]['id']
            movie_id = id_map.loc[title]['movieId']

            sim_scores = list(enumerate(sigmoid[int(idx)]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:26]
            movie_indices = [i[0] for i in sim_scores]

            movies = df_c.iloc[movie_indices][['title', 'id']]
            movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
            movies = movies.sort_values('est', ascending=False)
            return movies.head(10)
        submit = st.button('Get Recommendations')
        if submit:
          h = hybrid(userid,movie)
          st.write(f"Following are the top ten recommendations for you based on Hybrid technique")
          col1, col2, col3, col4, col5 = st.columns(5)
          col6, col7, col8, col9, col10 = st.columns(5)
          with col1:
              st.image(get_poster_url(h['id'].iloc[0]), caption=h['title'].iloc[0], width=150)
          with col2:
              st.image(get_poster_url(h['id'].iloc[1]), caption=h['title'].iloc[1], width=150)
          with col3:
              st.image(get_poster_url(h['id'].iloc[2]), caption=h['title'].iloc[2], width=150)
          with col4:
              st.image(get_poster_url(h['id'].iloc[3]), caption=h['title'].iloc[3], width=150)
          with col5:
              st.image(get_poster_url(h['id'].iloc[4]), caption=h['title'].iloc[4], width=150)
          with col6:
              st.image(get_poster_url(h['id'].iloc[5]), caption=h['title'].iloc[5], width=150)
          with col7:
              st.image(get_poster_url(h['id'].iloc[6]), caption=h['title'].iloc[6], width=150)
          with col8:
              st.image(get_poster_url(h['id'].iloc[7]), caption=h['title'].iloc[7], width=150)
          with col9:
              st.image(get_poster_url(h['id'].iloc[8]), caption=h['title'].iloc[8], width=150)
          with col10:
              st.image(get_poster_url(h['id'].iloc[9]), caption=h['title'].iloc[9], width=150)