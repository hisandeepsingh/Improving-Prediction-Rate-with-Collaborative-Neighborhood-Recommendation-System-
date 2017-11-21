import pandas as pd
import graphlab

# pass in column names for each CSV and read them using pandas.


# Reading users file:
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,
                    encoding='latin-1')

# Reading ratings file:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,
                      encoding='latin-1')

# Reading items file:
i_cols = ['movie id', 'movie title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
          'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
          'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols,
                    encoding='latin-1')

# content of each file to understand them better.



print ("Users "),
print users.shape
users.head()

print("\n Ratings"),
print ratings.shape
ratings.head()

print("\n items"),
print items.shape
items.head()

# Now  divide the ratings data set into test and train data for making models.
# Luckily GroupLens provides pre-divided data wherein the test data has 10 ratings for each user,
# i.e. 9430 rows in total. Lets load that:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_base = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')
#ratings_base.shape
#ratings_test.shape

# lets convert these in SFrames.


train_data = graphlab.SFrame(ratings_base)
test_data = graphlab.SFrame(ratings_test)

# A Simple Popularity Model
popularity_model = graphlab.popularity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating')

#Lets use this model to make top 5 recommendations for first 5 users and see what comes out

#Get recommendations for first 5 users and print them
#users = range(1,6) specifies user ID of first 5 users
#k=5 specifies top 5 recommendations to be given
print("Popularity Model")
popularity_recomm = popularity_model.recommend(users=range(1,6),k=5)
popularity_recomm.print_rows(num_rows=25)

#A Collaborative Filtering Model
#Lets create a model based on item similarity as follow:
#Train Model
item_sim_pearson = graphlab.item_similarity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating', similarity_type='pearson')
item_sim_cosine= graphlab.item_similarity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating', similarity_type='cosine')
item_sim_jaccard = graphlab.item_similarity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating', similarity_type='jaccard')

#Make Recommendations:
print("\n Collaborative Filtering Model(pearson)")
item_sim_recomm = item_sim_pearson.recommend(users=range(1,6),k=5)
item_sim_recomm.print_rows(num_rows=25)
print("\n Collaborative Filtering Model(cosine)")
item_sim_recomm1 = item_sim_cosine.recommend(users=range(1,6),k=5)
item_sim_recomm1.print_rows(num_rows=25)
print("\n Collaborative Filtering Model(jaccard)")
item_sim_recomm2 = item_sim_jaccard.recommend(users=range(1,6),k=5)
item_sim_recomm2.print_rows(num_rows=25)

#graphlab.item_similarity_recommender.compare_models(train_data, [popularity_model, item_sim_model,item_sim_model1,item_sim_model2],metric='precision_recall')



#Evaluating Recommendation Engines
#Lets compare both the models  built till now based on precision-recall characteristics:
model_performance = graphlab.compare(test_data, [popularity_model, item_sim_pearson,item_sim_cosine,item_sim_jaccard])
graphlab.show_comparison(model_performance,[popularity_model, item_sim_pearson,item_sim_cosine,item_sim_jaccard])

# factorization method
#fact_rec_model=graphlab.recommender.factorization_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating', user_data=None, item_data=None, num_factors=8, regularization=1e-08, linear_regularization=1e-10, side_data_factorization=True, nmf=False, binary_target=False, max_iterations=50, sgd_step_size=0, random_seed=0, solver='auto', verbose=True)
#Make Recommendations:
#fact_sim_recomm = fact_rec_model.recommend(users=range(1,6),k=5)
#fact_sim_recomm.print_rows(num_rows=25)
graphlab.canvas.show()
graphlab.canvas.set_target('ipynb')
