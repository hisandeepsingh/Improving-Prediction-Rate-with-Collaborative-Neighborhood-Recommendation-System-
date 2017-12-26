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
algo1 = SVD()
algo2= KNNBasic()
algo3=KNNBaseline()
algo4=KNNWithMeans()
algo5=NormalPredictor()

