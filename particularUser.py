import pandas as pd
from surprise import KNNBasic
from surprise import KNNBaseline
from surprise import Dataset
from surprise import evaluate, print_perf
import timeit
data = Dataset.load_builtin('ml-100k')
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,
                    encoding='latin-1')

bsl_options = {'method': 'sgd',
               'learning_rate': .00005,
               }
sim_options = {'name': 'pearson_baseline'}

algo1= KNNBaseline(sim_options=sim_options)
algo=KNNBasic()

#evaluate(algo, data, measures=['RMSE', 'MAE'])

# Retrieve the trainset.
trainset = data.build_full_trainset()
algo.train(trainset)
algo1.train(trainset)
userid = str(196)
itemid = str(242)
actual_rating = 4

print "prediction by KNN+ALS"
print algo1.predict(userid, itemid, actual_rating)