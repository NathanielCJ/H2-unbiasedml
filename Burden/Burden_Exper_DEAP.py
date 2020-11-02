#!/usr/bin/env python
# coding: utf-8

# In[6]:


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

class Model:
    
    def __init__(self, datadf, cfdf):
        self.pred = datadf['pred'].copy()
        self.label = datadf['Low'].copy()
        self.datadf = datadf.copy()
        self.cfdf = cfdf.copy()
        self.datadf['cfact_dist'] = 1/cfdf['fitness']
        self.datadf = self.datadf.query("cfact_dist == cfact_dist")
        
    def logits(self):
        raw_logits = np.clip(np.log(self.pred / (1 - self.pred)), -100, 100)
        return raw_logits

    def num_samples(self):
        return len(self.pred)

    def base_rate(self):
        """
        Percentage of samples belonging to the positive class
        """
        return np.mean(self.label)

    def accuracy(self):
        return self.accuracies().mean()

    def precision(self):
        return (self.label[self.pred.round() == 1]).mean()

    def recall(self):
        return (self.label[self.label == 1].round()).mean()

    def tpr(self):
        """
        True positive rate
        """
        return np.mean(np.logical_and(self.pred.round() == 1, self.label == 1))

    def fpr(self):
        """
        False positive rate
        """
        return np.mean(np.logical_and(self.pred.round() == 1, self.label == 0))

    def tnr(self):
        """
        True negative rate
        """
        return np.mean(np.logical_and(self.pred.round() == 0, self.label == 0))

    def fnr(self):
        """
        False negative rate
        """
        return np.mean(np.logical_and(self.pred.round() == 0, self.label == 1))

    def fn_cost(self):
        """
        Generalized false negative cost
        """
        return 1 - self.pred[self.label == 1].mean()

    def fp_cost(self):
        """
        Generalized false positive cost
        """
        return self.pred[self.label == 0].mean()

    def accuracies(self):
        return self.pred.round() == self.label
    
    def get_burden(self):
        return self.datadf['cfact_dist'].mean()

    def burden_eq_opp(self, othr, mix_rates=None):
        has_mix_rates = not (mix_rates is None)
        if not has_mix_rates:
            mix_rates = self.eq_odds_optimal_mix_rates(othr)
            sp2p, sn2p, op2p, on2p = tuple(mix_rates)
            
        print(1-sp2p, sn2p, 1-op2p, on2p, "pburden")
        # find indices close to the border and flip them
        if self.get_burden() > othr.get_burden():
            burdened_predictor = "self"
        else:
            burdened_predictor = "other"
            
        if burdened_predictor is "other":
            priv_df = self.datadf.copy()
            disa_df = othr.datadf.copy()
            priv_n2p = sn2p[0]
            disa_n2p = on2p[0]
        else:
            priv_df = othr.datadf.copy()
            disa_df = self.datadf.copy()
            priv_n2p = on2p[0]
            disa_n2p = sn2p[0]
            
        priv_neg = priv_df[priv_df['y_pred'] == 0]
        disa_neg = disa_df[disa_df['y_pred'] == 0]
        
        num_priv_n2p = int(priv_n2p * priv_df.shape[0]) 
        priv_ind = np.asarray(priv_neg.sort_values('cfact_dist', ascending = True).id)[:num_priv_n2p]     
        num_disa_n2p = int(disa_n2p * disa_df.shape[0]) 
        disa_ind = np.asarray(disa_neg.sort_values('cfact_dist', ascending = True).id)[:num_disa_n2p]


        priv_df.loc[priv_ind, "pred"] = 1 - priv_df.loc[priv_ind, "pred"]
        disa_df.loc[disa_ind, "pred"] = 1 - disa_df.loc[disa_ind, "pred"]
        
        fair_self = None
        fair_othr = None
        
        if burdened_predictor is "other":
            fair_self = Model(priv_df, self.cfdf)
            fair_othr = Model(disa_df, othr.cfdf)
        else:
            fair_self = Model(disa_df, self.cfdf)
            fair_othr = Model(priv_df, othr.cfdf)
            
        if not has_mix_rates:
            return fair_self, fair_othr, mix_rates
        else:
            return fair_self, fair_othr
        
    def burden_eq_odds(self, othr, mix_rates=None):
        has_mix_rates = not (mix_rates is None)
        if not has_mix_rates:
            mix_rates = self.eq_odds_optimal_mix_rates(othr)
            sp2p, sn2p, op2p, on2p = tuple(mix_rates)
            
        print(1-sp2p, sn2p, 1-op2p, on2p, "burden")
        # find indices close to the border and flip them
        if self.get_burden() > othr.get_burden():
            burdened_predictor = "self"
        else:
            burdened_predictor = "other"
            
        if burdened_predictor is "self":
            priv_df = self.datadf
            disa_df = othr.datadf
        else:
            priv_df = othr.datadf
            disa_df = self.datadf
            
        self_df = self.datadf.copy(deep=True)
        self_df_pos = self_df[self_df["y_pred"] == 1]
        self_df_neg = self_df[self_df["y_pred"] == 0]
        othr_df = othr.datadf.copy(deep=True)
        othr_df_pos = othr_df[othr_df["y_pred"] == 1]
        othr_df_neg = othr_df[othr_df["y_pred"] == 0]
        
            
        num_sn2p = int(self_df.shape[0] *  (sn2p[0]))
        sn2p_indices = np.asarray(self_df_neg.sort_values('cfact_dist', ascending = True).id)[:num_sn2p]
        
        num_on2p = int(othr_df.shape[0] *  (on2p[0]))
        on2p_indices = np.asarray(othr_df_neg.sort_values('cfact_dist', ascending = True).id)[:num_on2p]
        
        num_sp2n = int(self_df.shape[0] *  (1 - sp2p[0]))
        sp2n_indices = np.asarray(self_df_pos.sort_values('cfact_dist', ascending = True).id)[:num_sp2n]
        
        num_op2n = int(othr_df.shape[0] *  (1 - op2p[0]))
        op2n_indices = np.asarray(othr_df_pos.sort_values('cfact_dist', ascending = True).id)[:num_op2n]
        
        # flip those values
        self_df.loc[sn2p_indices, 'pred'] = 1 - self_df.loc[sn2p_indices, 'pred']      
        self_df.loc[sp2n_indices, 'pred'] = 1 - self_df.loc[sp2n_indices, 'pred']      

        othr_df.loc[on2p_indices, 'pred'] = 1 - othr_df.loc[on2p_indices, 'pred']
        othr_df.loc[op2n_indices, 'pred'] = 1 - othr_df.loc[op2n_indices, 'pred']      
        
        fair_self = Model(self_df, self.cfdf)
        fair_othr = Model(othr_df, othr.cfdf)

        if not has_mix_rates:
            return fair_self, fair_othr, mix_rates
        else:
            return fair_self, fair_othr
        
    def eq_odds(self, othr, mix_rates=None):
        has_mix_rates = not (mix_rates is None)
        if not has_mix_rates:
            mix_rates = self.eq_odds_optimal_mix_rates(othr)
            sp2p, sn2p, op2p, on2p = tuple(mix_rates)
            
        # select random indices to flip in our model
        self_fair_pred = self.datadf.copy()
        self_pp_indices = self_fair_pred[self_fair_pred["pred"] >= .5]
        self_pn_indices = self_fair_pred[self_fair_pred["pred"] < .5]
        self_pp_indices = self_pp_indices.sample(frac=(1-sp2p[0]))
        self_pn_indices = self_pn_indices.sample(frac=sn2p[0])
        
        # flip randomly the predictions in our model
        self_fair_pred.loc[self_pn_indices.id] = 1 - self_fair_pred.loc[self_pn_indices.id]
        self_fair_pred.loc[self_pp_indices.id] = 1 - self_fair_pred.loc[self_pp_indices.id]

        # select random indices to flip in the other model
        othr_fair_pred = othr.datadf.copy()
        othr_pp_indices = othr_fair_pred[othr_fair_pred["pred"] >= .5]
        othr_pn_indices = othr_fair_pred[othr_fair_pred["pred"] < .5]
        othr_pp_indices = othr_pp_indices.sample(frac=(1-op2p[0]))
        othr_pn_indices = othr_pn_indices.sample(frac=on2p[0])

        # dlip randomly the precitions of the other model
        othr_fair_pred.loc[othr_pn_indices.id] = 1 - othr_fair_pred.loc[othr_pn_indices.id]
        othr_fair_pred.loc[othr_pp_indices.id] = 1 - othr_fair_pred.loc[othr_pp_indices.id]

        # create new model objects with the now fair predictions
        
        fair_self = Model(self_fair_pred, self.cfdf)
        fair_othr = Model(othr_fair_pred, othr.cfdf)

        if not has_mix_rates:
            return fair_self, fair_othr, mix_rates
        else:
            return fair_self, fair_othr   
        
        
    def eq_odds_optimal_mix_rates(self, othr):
        sbr = float(self.base_rate())
        obr = float(othr.base_rate())

        sp2p = cvx.Variable(1)
        sp2n = cvx.Variable(1)
        sn2p = cvx.Variable(1)
        sn2n = cvx.Variable(1)

        op2p = cvx.Variable(1)
        op2n = cvx.Variable(1)
        on2p = cvx.Variable(1)
        on2n = cvx.Variable(1)

        sfpr = self.fpr() * sp2p + self.tnr() * sn2p
        sfnr = self.fnr() * sn2n + self.tpr() * sp2n
        ofpr = othr.fpr() * op2p + othr.tnr() * on2p
        ofnr = othr.fnr() * on2n + othr.tpr() * op2n
        error = sfpr + sfnr + ofpr + ofnr

        sflip = 1 - self.pred
        sconst = self.pred
        oflip = 1 - othr.pred
        oconst = othr.pred

        sm_tn = np.logical_and(self.pred.round() == 0, self.label == 0)
        sm_fn = np.logical_and(self.pred.round() == 0, self.label == 1)
        sm_tp = np.logical_and(self.pred.round() == 1, self.label == 1)
        sm_fp = np.logical_and(self.pred.round() == 1, self.label == 0)

        om_tn = np.logical_and(othr.pred.round() == 0, othr.label == 0)
        om_fn = np.logical_and(othr.pred.round() == 0, othr.label == 1)
        om_tp = np.logical_and(othr.pred.round() == 1, othr.label == 1)
        om_fp = np.logical_and(othr.pred.round() == 1, othr.label == 0)

        spn_given_p = (sn2p * (sflip * sm_fn).mean() + sn2n * (sconst * sm_fn).mean()) / sbr +                       (sp2p * (sconst * sm_tp).mean() + sp2n * (sflip * sm_tp).mean()) / sbr

        spp_given_n = (sp2n * (sflip * sm_fp).mean() + sp2p * (sconst * sm_fp).mean()) / (1 - sbr) +                       (sn2p * (sflip * sm_tn).mean() + sn2n * (sconst * sm_tn).mean()) / (1 - sbr)

        opn_given_p = (on2p * (oflip * om_fn).mean() + on2n * (oconst * om_fn).mean()) / obr +                       (op2p * (oconst * om_tp).mean() + op2n * (oflip * om_tp).mean()) / obr

        opp_given_n = (op2n * (oflip * om_fp).mean() + op2p * (oconst * om_fp).mean()) / (1 - obr) +                       (on2p * (oflip * om_tn).mean() + on2n * (oconst * om_tn).mean()) / (1 - obr)

        constraints = [
            sp2p == 1 - sp2n,
            sn2p == 1 - sn2n,
            op2p == 1 - op2n,
            on2p == 1 - on2n,
            sp2p <= 1,
            sp2p >= 0,
            sn2p <= 1,
            sn2p >= 0,
            op2p <= 1,
            op2p >= 0,
            on2p <= 1,
            on2p >= 0,
            spp_given_n == opp_given_n,
            spn_given_p == opn_given_p,
        ]

        prob = cvx.Problem(cvx.Minimize(error), constraints)
        prob.solve()

        res = np.array([sp2p.value, sn2p.value, op2p.value, on2p.value])
        return res

    def distance(self, othr):
        x = (self.fpr() - othr.fpr()) ** 2
        y = (self.tpr() - othr.tpr()) ** 2
        return math.sqrt(x + y)
    
    def __repr__(self):
        return '\n'.join([
            'Accuracy:\t%.3f' % self.accuracy(),
            'F.P. cost:\t%.3f' % self.fp_cost(),
            'F.N. cost:\t%.3f' % self.fn_cost(),
            'Base rate:\t%.3f' % self.base_rate(),
            'Avg. score:\t%.3f' % self.pred.mean(),
        ])


# In[7]:

def Burden_compas_res():

    file_name = './compas-scores-two-years.csv'
    full_data = pd.read_csv(file_name)
    full_data.race.value_counts()

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
    full_data = full_data.query("days_b_screening_arrest <= 30")         .query("days_b_screening_arrest >= -30")        .query("is_recid != -1")        .query("c_charge_degree != 'O'")         .query("score_text != 'N/A'" )        .query("race != 'Asian'")        .query("race != 'Native American'")        .query("race != 'Other'")        .query("race != 'Hispanic'")

    # randomize race for later use
    full_data['race_random'] = np.random.permutation(full_data['race'])

    # check how many random to the same thing
    np.sum(full_data['race']==full_data['race_random'])

    # check counts of recidivism by race
    full_data.groupby(['race','is_recid'])['id'].count().reset_index()

    # keep relevant columns 
    columns_kept = ['sex', 'age', 'age_cat', 'race', 'juv_fel_count', 'priors_count', 'c_charge_degree',                 'is_recid', 'decile_score', 'two_year_recid', 'c_jail_in', 'c_jail_out', 'race_random',                 'charge_F', 'charge_M', 'score_text', 'id']
    full_data = full_data.loc[:, columns_kept]

    full_data = full_data.set_index('id')




    # In[8]:


    learning_data = full_data.copy(deep=True)
    features_to_transform = ['age_cat', 'sex', 'race', 'c_charge_degree']

    for feature in features_to_transform:
        dummies = pd.get_dummies(learning_data[feature], prefix=feature)
        learning_data = pd.concat([learning_data, dummies], axis = 1)


    learning_data.columns = learning_data.columns.str.replace('-', '_')

    learning_data['score_factor'] = np.where(learning_data['score_text'] == 'Low', 'Low', 'MediumHigh')
    dummies = pd.get_dummies(learning_data['score_factor'])
    learning_data = pd.concat([learning_data, dummies] , axis = 1)


    # In[9]:


    X_Labels = ['sex_Male', 'age', 'race_African_American', 'priors_count',            'juv_fel_count', 'c_charge_degree_F', 'c_charge_degree_M']
    Y_Labels = ['Low']


    X =  learning_data.loc[:, X_Labels]
    Y =  learning_data.loc[:, Y_Labels]
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
    model = xgboost.Booster(params)
    model.load_model('compas_xgboost.json')
    xgboost.cv(params,xgb_full, nfold = 3, metrics="auc" , num_boost_round=10)
    learning_data['pred'] = model.predict(xgb_full)

    # In[6]:

    fpr, tpr , thresholds = roc_curve(learning_data['Low'], learning_data['pred'])
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    print("Optimal Threshold obtained using difference of TPR and FPR " + str(optimal_threshold))
    learning_data['y_pred'] = np.where(learning_data['pred'] > optimal_threshold, 1, 0)

    # In[14]:
    burden_data = learning_data.copy()
    c_facts_df = pd.read_csv("compas_cfact_10_29_2020_09_53.csv")
    c_facts_df.index = c_facts_df.id

    burden_data['fitness'] = c_facts_df.fitness
    burden_data['dist'] = 1/c_facts_df.fitness

    burden_data.head()

    # In[15]:

    print(burden_data.groupby('race_African_American', as_index=False)[['fitness','dist']].mean())

    # In[16]:

    # In[16]:

    burden_data_df = learning_data.copy()
    burden_data_df['fitness'] = c_facts_df.fitness
    burden_data_df['id'] = burden_data_df.index

    aa_df = burden_data_df.loc[:, X_Labels+['y_pred', 'pred', 'id']+Y_Labels][burden_data_df['race_African_American']  == 1]
    c_df = burden_data_df.loc[:, X_Labels+['y_pred', 'pred', 'id']+Y_Labels][burden_data_df['race_African_American']  == 0]

    aa_m = Model(aa_df, c_facts_df.loc[aa_df.index])
    c_m = Model(c_df, c_facts_df.loc[c_df.index])

    bf_aa_m, bf_c_m, _ = aa_m.burden_eq_odds(c_m)
    rf_aa_m, rf_c_m, _ = aa_m.eq_odds(c_m)
    p_bf_aa_m, p_bf_c_m, _ = aa_m.burden_eq_opp(c_m)

    burden_data_df.head()


    # In[17]:


    b_agg_fair = pd.concat([bf_aa_m.datadf, bf_c_m.datadf])
    b_agg_fair['is_recid'] = learning_data.loc[b_agg_fair.index, 'is_recid']
    pb_agg_fair = pd.concat([p_bf_aa_m.datadf, p_bf_c_m.datadf])
    pb_agg_fair['is_recid'] = learning_data.loc[pb_agg_fair.index, 'is_recid']
    r_agg_fair = pd.concat([rf_aa_m.datadf, rf_c_m.datadf])
    r_agg_fair['is_recid'] = learning_data.loc[r_agg_fair.index, 'is_recid']

    acc_ret = {
        'random':r_agg_fair[r_agg_fair['pred'] >= .5][r_agg_fair['is_recid'] == 0].shape[0]/r_agg_fair.shape[0],
        'burden':b_agg_fair[b_agg_fair['pred'] >= .5][b_agg_fair['is_recid'] == 0].shape[0]/b_agg_fair.shape[0],
        'partial burden':pb_agg_fair[pb_agg_fair['pred'] >= .5][pb_agg_fair['is_recid'] == 0].shape[0]/pb_agg_fair.shape[0],
        'none': learning_data[learning_data['y_pred'] == 1][learning_data['is_recid'] == 0].shape[0]/learning_data.shape[0]
    }

    # ### Demographic parity

    # In[47]:


    rf_aa_dp = rf_aa_m.datadf['pred'].round().mean() 
    rf_c_dp  = rf_c_m.datadf['pred'].round().mean()

    bf_aa_dp = bf_aa_m.datadf['pred'].round().mean()
    bf_c_dp  = bf_c_m.datadf['pred'].round().mean()

    aa_dp = aa_m.datadf['pred'].round().mean()
    c_dp  = c_m.datadf['pred'].round().mean()

    p_bf_aa_dp = p_bf_aa_m.datadf['pred'].round().mean()
    p_bf_c_dp  = p_bf_c_m.datadf['pred'].round().mean()


    demo_part_ret = {
        'random': (rf_c_dp, rf_aa_dp),
        'partial burden': (p_bf_c_dp, p_bf_aa_dp),
        'burden': (bf_c_dp, bf_aa_dp),
        'none': (c_dp, aa_dp)
    }


    # ### equ. odds and opp
    # 
    # Pr{^Y= 1|A= 0,Y= 1}= Pr{^Y= 1|A= 1,Y= 1} is opp

    # In[48]:


    rf_aa_eqop = rf_aa_m.datadf.query("Low == 1")['pred'].round().mean()
    rf_c_eqop  = rf_c_m.datadf.query("Low == 1")['pred'].round().mean()

    bf_aa_eqop = bf_aa_m.datadf.query("Low == 1")['pred'].round().mean()
    bf_c_eqop  = bf_c_m.datadf.query("Low == 1")['pred'].round().mean()

    p_bf_aa_eqop = p_bf_aa_m.datadf.query("Low == 1")['pred'].round().mean()
    p_bf_c_eqop  = p_bf_c_m.datadf.query("Low == 1")['pred'].round().mean()

    aa_eqop = aa_m.datadf.query("Low == 1")['pred'].round().mean()
    c_eqop  = c_m.datadf.query("Low == 1")['pred'].round().mean()

    eq_opp_ret = {
        'random': (rf_c_eqop, rf_aa_eqop),
        'partial burden': (p_bf_c_eqop, p_bf_aa_eqop),
        'burden': (bf_c_eqop, bf_aa_eqop),
        'none': (c_eqop, aa_eqop)
    }

    # In[52]:


    rf_aa_eq1 = rf_aa_m.datadf.query("Low == 1")['pred'].round().mean()
    rf_c_eq1  = rf_c_m.datadf.query("Low == 1")['pred'].round().mean()
    rf_aa_eq0 = rf_aa_m.datadf.query("Low == 0")['pred'].round().mean()
    rf_c_eq0  = rf_c_m.datadf.query("Low == 0")['pred'].round().mean()


    bf_aa_eq1 = bf_aa_m.datadf.query("Low == 1")['pred'].round().mean()
    bf_c_eq1  = bf_c_m.datadf.query("Low == 1")['pred'].round().mean()
    bf_aa_eq0 = bf_aa_m.datadf.query("Low == 0")['pred'].round().mean()
    bf_c_eq0  = bf_c_m.datadf.query("Low == 0")['pred'].round().mean()

    p_bf_aa_eq1 = p_bf_aa_m.datadf.query("Low == 1")['pred'].round().mean()
    p_bf_c_eq1  = p_bf_c_m.datadf.query("Low == 1")['pred'].round().mean()
    p_bf_aa_eq0 = p_bf_aa_m.datadf.query("Low == 0")['pred'].round().mean()
    p_bf_c_eq0  = p_bf_c_m.datadf.query("Low == 0")['pred'].round().mean()


    aa_eq1 = aa_m.datadf.query("Low == 1")['pred'].round().mean()
    c_eq1  = c_m.datadf.query("Low == 1")['pred'].round().mean()
    aa_eq0 = aa_m.datadf.query("Low == 0")['pred'].round().mean()
    c_eq0  = c_m.datadf.query("Low == 0")['pred'].round().mean()


    eq_odd_ret = {
        'random': ((rf_c_eq0, rf_aa_eq0), (rf_c_eq1, rf_aa_eq1)),
        'partial burden': ((p_bf_c_eq0, p_bf_aa_eq0), (p_bf_c_eq1, p_bf_aa_eq1)),
        'burden': ((bf_c_eq0, bf_aa_eq0), (bf_c_eq1, bf_aa_eq1)),
        'none': ((c_eq0, aa_eq0), (c_eq1, aa_eq1))
    }

    return {
        'eq odds': eq_odd_ret,
        'eq opp' : eq_opp_ret,
        'accuracy' : acc_ret,
        'demo parity' : demo_part_ret
    }


if __name__ == "__main__":
    print(Burden_compas_res())
