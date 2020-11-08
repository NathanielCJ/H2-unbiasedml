import pandas as pd
import random
import numpy as np
import sklearn as sk
from scipy import stats
import statistics
import cvxpy as cvx
import xgboost
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
xgboost.__version__
from scipy.linalg import svd
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import math

from mpl_toolkits.mplot3d import Axes3D
import random
import time

import warnings
warnings.filterwarnings('ignore')

#counterFactualListCollection is a dict of the generations of counterfactualLists for an individual
def displayCounterFactuals(counterFactualListCollection, generation):
    dataFrame = counterFactualListCollection[generation]
    ages = []
    priors_counts = []
    juv_fel_counts = []
    person_encoding = []
    for row in dataFrame.index:
        # store x, y, z for the 3D scatterplot
        ages.append(dataFrame.at[row, 'age'])
        priors_counts.append(dataFrame.at[row, 'priors_count'])
        juv_fel_counts.append(dataFrame.at[row, 'juv_fel_count'])
        
        # sex_Male|sex_Female|race_African_American|race_Caucasian|race_Hispanic|race_Other|c_charge_degree_F 
        binary_encoding = [dataFrame.at[row, 'sex_Male'],
                           dataFrame.at[row, 'sex_Female'],
                           dataFrame.at[row, 'race_African_American'],
                           dataFrame.at[row, 'race_Caucasian'],
                           dataFrame.at[row, 'race_Hispanic'],
                           dataFrame.at[row, 'race_Other'],
                           dataFrame.at[row, 'c_charge_degree_F']]
        
        # use unique encoding of a person as index for color on 3D display
        person_encoding.append(int("".join(str(x) for x in binary_encoding), 2))
       
    fig = plt.figure()
    ax = Axes3D(fig)
    
    # plot the values
    ax.scatter(ages, priors_counts, person_encoding, zdir='z', s=20, c=person_encoding, depthshade=True)
    ax.set_xlabel('age')
    ax.set_ylabel('priors_count')
    ax.set_zlabel('juv_fel_count')
    
    plt.pause(0.05)
    
def displayDriver(counterFactualListCollection):
    print(len(counterFactualListCollection))
    i = 0
    while i < len(counterFactualListCollection):
        displayCounterFactuals(counterFactualListCollection, i)
        plt.show()
        i = i+1

def train():
    file_name = 'compas-scores-two-years.csv'
    full_data = pd.read_csv(file_name)
    full_data.race.value_counts()

    print(full_data.shape)

    # remove groups with few instances 
    full_data = full_data.query("race != 'Asian'").query("race != 'Native American'")

    # group by felony or misdemenor charge
    full_data.groupby(['c_charge_degree','is_recid'])['id'].count().reset_index()

    # turn charge degree to numbers 
    full_data['c_charge_degree'] = pd.Categorical(full_data['c_charge_degree'])
    # change numbers into dummies (1 for present 0 for absent)
    dummies = pd.get_dummies(full_data['c_charge_degree'], prefix='charge')
    full_data = pd.concat([full_data, dummies], axis=1)

    # remove bad data
    full_data = full_data.query("days_b_screening_arrest <= 30") \
            .query("days_b_screening_arrest >= -30")\
            .query("is_recid != -1")\
            .query("c_charge_degree != 'O'") \
            .query("score_text != 'N/A'" )

    print(full_data.shape)


    # randomize race for later use
    full_data['race_random'] = np.random.permutation(full_data['race'])

    # check how many random to the same thing
    np.sum(full_data['race']==full_data['race_random'])

    # check counts of recidivism by race
    full_data.groupby(['race','is_recid'])['id'].count().reset_index()

    # keep relevant columns 
    columns_kept = ['sex', 'age', 'age_cat', 'race', 'juv_fel_count', 'priors_count', 'c_charge_degree', \
                    'is_recid', 'decile_score', 'two_year_recid', 'c_jail_in', 'c_jail_out', 'race_random', \
                    'charge_F', 'charge_M', 'score_text', 'id']
    full_data = full_data.loc[:, columns_kept]

    full_data.set_index('id')

    full_data.head()

    learning_data = full_data.copy(deep=True)
    features_to_transform = ['age_cat', 'sex', 'race', 'c_charge_degree']

    for feature in features_to_transform:
        dummies = pd.get_dummies(learning_data[feature], prefix=feature)
        learning_data = pd.concat([learning_data, dummies], axis = 1)
    learning_data.head()

    learning_data.columns = learning_data.columns.str.replace('-', '_')

    learning_data['score_factor'] = np.where(learning_data['score_text'] == 'Low', 'Low', 'MediumHigh')
    dummies = pd.get_dummies(learning_data['score_factor'])
    learning_data = pd.concat([learning_data, dummies] , axis = 1)
    learning_data.head()

    X_Labels = ['sex_Male', 'sex_Female', 'age', 'race_African_American', 'race_Caucasian', 'priors_count',\
                'race_Hispanic', 'race_Other', 'juv_fel_count', 'c_charge_degree_F', 'c_charge_degree_M']
    Y_Labels = ['Low']


    X =  learning_data.loc[:, X_Labels]
    Y =  learning_data.loc[:, Y_Labels]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size =  0.2, random_state =  4)

    # find and later remove linearly correlated pairs (not for xgboost but for LOCO later)
    corr_thresh = .7
    for column_a in X.columns:
        for column_b in X.columns:
            if column_a is not column_b:
                if X[column_a].corr(X[column_b]) > corr_thresh:
                    print(column_a + " " + column_b)


    xgb_full = xgboost.DMatrix(X, label=Y)
    xgb_train =  xgboost.DMatrix(X_train, label = Y_train)
    xgb_test = xgboost.DMatrix(X_test, label = Y_test)
    # use validation set to choose # of trees
    params = {
        "eta": 0.002,
        "max_depth": 4,
        "objective": 'binary:logistic',
        "eval_metric":"auc",
        "subsample": 0.5
    }
    #model_train = xgboost.train(params, xgb_train, 10000, evals = [(xgb_test, "test")], verbose_eval=1000)
    model = xgboost.train(params, xgb_train, 10000, evals = [(xgb_test, "test")], verbose_eval= 1000)
    xgboost.cv(params,xgb_full, nfold = 3, metrics="auc" , num_boost_round=10)
    learning_data['pred'] = model.predict(xgb_full)
    return learning_data

class CounterFactFactory:
    # do we need to mutate catagorical and continous data seperately? 
    # we probably need to ask for some bounds on each feature

    # bounds is a dict with form {<feature_name>: (<"catagorical"/"continous">, <number catagories/continous scale>)}
    # continous scale is max value for generating the initial population continous features
    # -(continous scale) is min value for generating the initial population
    def __init__(self, indiv, model_pred_func, bounds, fit_func, 
               selection_rate=.5, prob_mut=.2, prob_cross=.5, init_pop_size=1000,
               round_func=np.around, generations=200,
               **kwargs):
        self.model_pred_func = model_pred_func  # f(x)
        self.indiv = indiv                      # x
        self.c_fact_list = pd.DataFrame()       # subset of I
        self.feature_bounds = bounds            # may be used for mutation
        self.fitness = fit_func                 # d(x, c)
        self.pop_size = init_pop_size
        self.round_func = round_func
        self.p_s = selection_rate
        self.p_m = prob_mut
        self.p_c = prob_cross
        self.indiv_class = self.round_func(self.model_pred_func(xgboost.DMatrix(self.indiv)))
        self.indiv_class = self.indiv_class[0]
        self.gen = generations
        
        self.catL = [] 
        self.conL = []
        self.feature_names =  []
        
        self.c_fact_list_collection = {}
        
        
        for feature_name in self.feature_bounds:
            bounds = self.feature_bounds[feature_name]
            self.feature_names.append(feature_name)
            if bounds[0].startswith("cat"):
                self.catL.append(feature_name)
            elif bounds[0].startswith("cont"):
                self.conL.append(feature_name)
            else:
                print("error in bounds dictionary")
                print("expected catagorical or continous and got neither")
                raise ValueError()



    # step 7
    def genRandomCFactuals(self):
        
        temp = None
        if not self.c_fact_list.empty:
            temp = self.c_fact_list.columns
            
        # keep generating counter factuals until our population is up to size
        while self.c_fact_list.shape[0] < self.pop_size:
            # generate new individuals
            new_indivs = pd.DataFrame()
            for feature_name in self.feature_names:
                bounds = self.feature_bounds[feature_name]
                if bounds[0].startswith("cat"):
                    # catagorical random
                    new_indivs[feature_name] = np.random.randint(0, bounds[1], size=self.pop_size-self.c_fact_list.shape[0])
                elif bounds[0].startswith("cont"):
                    # continous random
                    new_indivs[feature_name] = np.random.rand(self.pop_size-self.c_fact_list.shape[0], 1) * (bounds[1])
                else:
                    print("error in bounds dictionary")
                    print("expected catagorical or continous and got neither")
            # calculate new predictions and filter
            new_indivs['prediction'] = new_indivs.apply(lambda row : 
                                                      self.round_func(self.model_pred_func(xgboost.DMatrix(pd.DataFrame(row).T)))[0],
                                                      axis = 1)
            new_indivs = new_indivs[new_indivs.prediction != self.indiv_class]

            # add the survivors to the dataframe
            if temp is None:
                temp = new_indivs.columns
                
            self.c_fact_list = pd.concat([self.c_fact_list, new_indivs], axis=0)
            self.c_fact_list = self.c_fact_list[temp]
            
    # step 8
    def selection(self):
        if self.c_fact_list is pd.DataFrame.empty:
            print("attempted to select on an empty population")
            print("try calling genRandomCFactuals first?")
            print("if this appears from a call of getCFactuals then its a bug. please report it")
            return

        # sort by fitness and then keep the top half
        self.c_fact_list['fitness'] = self.c_fact_list.apply(lambda row : 
                                                             self.fitness(self.indiv, row, self.catL, self.conL),
                                                             axis=1)
        self.c_fact_list = (self.c_fact_list.loc[self.c_fact_list.fitness.sort_values(ascending=False).index
                                                 ])[:int(self.c_fact_list.shape[0]*self.p_s)]

    # step 9
    def mutation(self):
        # get individuals to mutate
        to_mut_indivs = self.c_fact_list.sample(frac=self.p_m, axis=0)

        # mutate the individuals
        for index, individual in to_mut_indivs.iterrows():
          # do the mutation in place
          individual[0] = 0

        # set the counter factuals to their mutated version
        self.c_fact_list.loc[to_mut_indivs.index] = to_mut_indivs

        # recalculate the predictions
        self.c_fact_list.drop('prediction', inplace=True, axis=1)
        self.c_fact_list.drop('fitness', inplace=True, axis=1)
        self.c_fact_list['prediction'] = self.c_fact_list.apply(lambda row : 
                                                self.round_func(self.model_pred_func(xgboost.DMatrix(pd.DataFrame(row).T)))[0],
                                                axis = 1)
        
        # recalculate the fitnesses for use later
        self.c_fact_list['fitness'] = self.c_fact_list.apply(lambda row : 
                                                             self.fitness(self.indiv, row, self.catL, self.conL),
                                                             axis = 1)

    # step 10
    def crossover(self):
        # make sure to recalculate the fitnesses, and predictions
        # make sure to filter out
        # idea: 
        #1. sort by fitness, and keep top p_c portion. 
        #2. cross over between c_0 & c_1, c_2 & c_3, and so on
        #3. to cross over, swap each feature value with 50% probability
        self.c_fact_list['fitness'] = self.c_fact_list.apply(lambda row : 
                                                             self.fitness(self.indiv, row, self.catL, self.conL),
                                                             axis = 1)
        self.c_fact_list = (self.c_fact_list.loc[self.c_fact_list.fitness.sort_values(ascending=False).index])
        self.c_fact_list.drop('prediction', inplace=True, axis=1)
        self.c_fact_list.drop('fitness', inplace=True, axis=1)
        
        count = self.c_fact_list.shape[0] * self.p_c
        i = 0
        while i < count:
            for feature_name in self.feature_bounds:
                if(np.random.random_sample() > 0.5):
                    old_val = self.c_fact_list.loc[i, feature_name]
                    self.c_fact_list.loc[i, feature_name] = self.c_fact_list.loc[i+1, feature_name]
                    self.c_fact_list.loc[i, feature_name] = old_val
            i = i + 2
        pass
    
        self.c_fact_list['prediction'] = self.c_fact_list.apply(lambda row : 
                                        self.round_func(self.model_pred_func(xgboost.DMatrix(pd.DataFrame(row).T)))[0],
                                        axis = 1)

        # recalculate the fitnesses for use later
        self.c_fact_list['fitness'] = self.c_fact_list.apply(lambda row : 
                                                             self.fitness(self.indiv, row, self.catL, self.conL),
                                                             axis = 1)
        

    # step 11
    def filterCFacts(self):
        # filter the population of counter factuals (self.c_fact_list) to remove all
        # individuals that have the same prediction class (self.indiv_class) as the original 
        # individual (self.indiv)
        self.c_fact_list = self.c_fact_list[self.c_fact_list.prediction != self.indiv_class]
        pass

    # step 12
    def getCFactuals(self):

        for generation in range(self.gen):
            self.genRandomCFactuals()
            self.c_fact_list.reset_index(drop=True, inplace=True)
            self.selection()
            self.c_fact_list.reset_index(drop=True, inplace=True)
            self.mutation()
            self.c_fact_list.reset_index(drop=True, inplace=True)
            self.crossover()
            self.c_fact_list.reset_index(drop=True, inplace=True)
            self.filterCFacts()
            self.c_fact_list.reset_index(drop=True, inplace=True)
            self.c_fact_list_collection[generation] = self.c_fact_list
            
            if(generation == self.gen-1):
                print("Generation", generation, "with max fitness", self.c_fact_list.fitness.max())
            
        return self.c_fact_list.loc[0, :]

    

    
def d(indiv, other, cat, con):
    
    agg = pd.DataFrame(indiv)
    agg.loc[1] = other
    agg.reset_index(drop=True, inplace=True)
    
    ncat = len(cat)
    ncon = len(con)
    
    xcat = agg.loc[:, cat]
    xcon = agg.loc[:, con]
        
    # MAD normalized L1 Norm
    normedl1norm = 0
    for feature in con:
        mad = burden_data.loc[:, feature].mad()
        normedl1norm += abs(xcon.loc[0, feature] - xcon.loc[1, feature]) / mad
    
    # simpMat
    # both pos
    xcat = xcat.astype('int32')
    PosMat = xcat.loc[0] & xcat.loc[1] 
    NegMat = (1 - xcat.loc[0]) & (1 - xcat.loc[1])
    total = xcat.shape[1]
    dist = (PosMat.sum() + NegMat.sum())/total
    
    n = ncat + ncon
    return ncon*normedl1norm/n + ncat*dist/n


b = {
    'sex_Male':   ("catagorical", 2),
    'sex_Female': ("catagorical", 2),
    'age':        ("continous", 100),
    'race_African_American': ("catagorical", 2),
    'race_Caucasian': ("catagorical", 2),
    'priors_count':   ("continous", 100),
    'race_Hispanic':  ("catagorical", 2),
    'race_Other':     ("catagorical", 2),
    'juv_fel_count':  ("continous", 100),
    'c_charge_degree_F': ("catagorical", 2),
    'c_charge_degree_M': ("catagorical", 2)
}

# cdf = pd.DataFrame(columns=['fitness']+X_Labels)
burden_data = None

def inv_dist(ind, otr, cat, con):
    return 1 / d(ind, otr, cat, con)

#for i in range(0, int(learning_data.shape[0])):

def generator(i):
    global burden_data
    learning_data = train()
    burden_data = learning_data.loc[:, X_Labels]
    cdf = pd.DataFrame(columns=['fitness']+X_Labels)
    burden_indiv = pd.DataFrame(burden_data.iloc[i]).T
    c_fact = CounterFactFactory(indiv=burden_indiv, model_pred_func=model.predict, bounds=b, 
                            fit_func=inv_dist, init_pop_size=100, generations=25)

    print("working on indiv", str(i))
    final_c_fact = c_fact.getCFactuals()
    cdf = cdf.append(final_c_fact)

    print("final set of lists")
    #displayCounterFactuals(c_fact.c_fact_list_collection, 0)
    #displayDriver(c_fact.c_fact_list_collection)
    #print(c_fact.c_fact_list)
    return c_fact.c_fact_list_collection
