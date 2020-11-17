#!/usr/bin/env python
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
import math
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, wait
from deap import base, creator, tools, algorithms

import warnings
warnings.filterwarnings('ignore')

class Genetic_Burden_Generator:
    def __init__(self, data, label_column):
        self.data = data
        self.label_column = label_column

    # takes in a csv of preprocessed data
    # outputs predictions and models
    def train_model(self):
        X =  self.data.drop(self.label_column, axis=1)
        Y =  self.data.loc[:, self.label_column]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size =  0.2, random_state =  4)        

        xgb_full = xgboost.DMatrix(X, label=Y)
        xgb_train =  xgboost.DMatrix(X_train, label = Y_train)
        xgb_test = xgboost.DMatrix(X_test, label = Y_test)
        # use validation set to choose # of trees
        params = {
            "eta": 0.002,
            "max_depth": 4,
            "objective": 'binary:logistic',
            "eval_metric":"auc",
            #"tree_method": 'gpu_hist',
            "subsample": 0.5
        }
        #model_train = xgboost.train(params, xgb_train, 10000, evals = [(xgb_test, "test")], verbose_eval=1000)
        model = xgboost.train(params, xgb_train, 10000, evals = [(xgb_test, "test")], verbose_eval= 1000)
        xgboost.cv(params,xgb_full, nfold = 3, metrics="auc" , num_boost_round=10)
        
        self.preds = model.predict(xgb_full)
        
        return (self.preds, model)

    def find_c_facts(self, model_predict_function, 
                     column_to_type:dict=None, bounds_dict:dict=None,
                     vebose=False):
        X_Labels = [str(x).strip() for x in self.data.columns]
        X_Labels.remove(self.label_column)
        
        if bounds_dict is not None:
            b = bounds_dict
        elif column_to_type is not None:            
            b = {}
            for feature in X_Labels:
                if column_to_type[feature].startswith("cont"):
                    b[feature] = ("cont", self.data[feature].max()*1.5)
                if column_to_type[feature].startswith("cata"):
                    b[feature] = ("cata", self.data[feature].nunique())
        else:
            b = {}
            for feature in X_Labels:
                b[feature] = ("cont", self.data[feature].max())
        ordered_cols = X_Labels

        madArr = [0] * len(ordered_cols)
        for feature in ordered_cols:
            bounds = b[feature]
            if bounds[0].startswith("cont"):
                madArr[ordered_cols.index(feature)] = self.data.loc[:, feature].mad()


        cat_list = []
        con_list = []

        for feature in ordered_cols:
            if b[feature][0].startswith('cata'):
                cat_list.append(ordered_cols.index(feature))
            else:
                con_list.append(ordered_cols.index(feature))

        def create_bound_func_arr():
            ret = [0] * len(X_Labels)
            for i in cat_list:
                ret[i] = toolbox.cata_2_var
            for i in con_list:
                ret[i] = toolbox.con_var
            return ret

        def create_indiv():
            ret = [0] * len(X_Labels)
            for i in cat_list:
                ret[i] = toolbox.cata_2_var()
            for i in con_list:
                ret[i] = toolbox.con_var()
            return creator.Individual(ret)

        def mutate(indiv, func_arr, prob_mut):
            for i in range(len(indiv)):
                if(random.random() <= prob_mut):
                    indiv[i] = func_arr[i]()
            return indiv,
        
        def distance(indiv, other, cat, con, madArray, pred_func, columns=X_Labels):    
            dfm = pd.DataFrame([indiv, other], columns=columns)    
            preds = pred_func(xgboost.DMatrix(dfm))
            
            if preds.round().sum() != 1:
                return -1,
            
            ncat = len(cat)
            ncon = len(con)
                    
            # MAD normalized L1 Norm
            normedl1norm = 0
            if ncon > 0:
                for index in con:
                    mad = madArray[index]
                    normedl1norm += abs(indiv[index] - other[index]) / mad

                
            # simpMat
            # both pos
            dist = 0
            if ncat > 0:
                npindiv = np.array(indiv).astype(int)[cat]
                npother = np.array(other).astype(int)[cat]
                    
                PosMat = npindiv & npother 
                NegMat = (1 - npindiv) & (1 - npother)
                total = npother.shape[0]
                dist = 1-((PosMat.sum() + NegMat.sum())/total)
            
            n = ncat + ncon
            return (1/(ncon*normedl1norm/n + ncat*dist/n),)

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        toolbox.register("cata_2_var", random.randint, 0, 1)
        toolbox.register("con_var", random.uniform, 0, 100)

        toolbox.register("individual", create_indiv)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", mutate, func_arr=create_bound_func_arr(), prob_mut=.5)
        toolbox.register("select", tools.selTournament, tournsize=3)

        c_facts = []

        pop_size = len(X_Labels)**2
        gens = 60

        for i in range(self.data.shape[0]):
            iper = self.data.loc[:,X_Labels].iloc[i].values.tolist()
            toolbox.register("evaluate", distance, other=iper, cat=cat_list, con=con_list, madArray=madArr, pred_func=model_predict_function)

            
            pop = toolbox.population(n=pop_size)
            hof = tools.HallOfFame(1)

            pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=gens, halloffame=hof, verbose=False)
            if vebose:
                print(str(i), ": Best individual is: %s\nwith fitness: %s" % (hof[0], hof[0].fitness))
            
            cfact_final = list(hof[0])
            cfact_final.append(hof[0].fitness.values[0])
                
            c_facts.append(cfact_final)
                
            toolbox.unregister("evaluate")
            
        c_facts_df = pd.DataFrame(c_facts, columns=X_Labels+['fitness'], 
                                  index=self.data.index)
        print(c_facts_df.shape)
        c_facts_df.head()
        
        self.c_facts = c_facts_df
        
        return c_facts_df

    def get_minimal_model(self, group, save_path=None):
        from datetime import datetime

        model_out_df = pd.DataFrame()
        model_out_df['label'] = self.data[self.label_column]
        model_out_df['group'] = self.data[group]
        model_out_df['fitness'] = self.c_facts['fitness']
        print(model_out_df.groupby('group', as_index=True)[['fitness']].mean())

        model_out_df['prediction'] = self.preds
        if save_path is None:
            model_out_df.to_csv("burden_model_" + datetime.now().strftime("%m_%d_%Y_%H_%M") +".csv")
        else:
            model_out_df.to_csv(save_path)
            
        return model_out_df
            


if __name__ == "__main__":
    data = pd.read_csv("./clean_compas.csv")
    burGen = Genetic_Burden_Generator(data.iloc[:17], "Low")
    preds, model = burGen.train_model()
    c_facts = burGen.find_c_facts(model.predict, 
                 column_to_type=None, bounds_dict=None,
                 vebose=True)
    mini = burGen.get_minimal_model("race_African_American")