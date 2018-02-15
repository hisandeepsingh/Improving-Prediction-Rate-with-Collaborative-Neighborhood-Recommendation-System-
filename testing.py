import pandas as pd
import surprise
from surprise import SVD
from surprise import KNNBasic
from surprise import KNNBaseline
from surprise import KNNWithMeans
from surprise import NormalPredictor
from surprise import Dataset
from surprise import evaluate, print_perf
import timeit
data = Dataset.load_builtin('ml-100k')
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,
                    encoding='latin-1')
#print users.head()
#data.split(n_folds=2)
# initializing all algorithms
bsl_options = {'method': 'sgd',
               'learning_rate': .00005,
		'n_epochs': 20,
               }
sim_options = {'name': 'pearson_baseline'}

algo3= KNNBaseline(sim_options=sim_options)
algo1 = SVD()
algo2= KNNBasic()
algo4=KNNWithMeans()
algo5=NormalPredictor()

#start_time1=time.time()
start = timeit.default_timer()
perf1 = evaluate(algo1, data, measures=['RMSE', 'MAE'])
stop = timeit.default_timer()
print("--- %s seconds ---" % (stop-start))
#start_time2=time.time()
start1 = timeit.default_timer()
perf2 = evaluate(algo2, data, measures=['RMSE', 'MAE'])
stop1 = timeit.default_timer()
print ("...%s seconds..."%(stop1-start1))
#start_time3=time.time()
perf3 = evaluate(algo3, data, measures=['RMSE', 'MAE'])
#print ("...%s seconds..."%(time.time()-start_time3))
#start_time4=time.time()
perf4 = evaluate(algo4, data, measures=['RMSE', 'MAE'])
#print ("...%s seconds..."%(time.time()-start_time4))
#start_time5=time.time()
perf5 = evaluate(algo5, data, measures=['RMSE', 'MAE'])
#print ("...%s seconds..."%(time.time()-start_time5))
