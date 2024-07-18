import torch
from torch.utils.data import Dataset
from torch.autograd import grad
import numpy as np

class FairnessDataset(Dataset):
    '''
    An abstract dataset class wrapped around Pytorch Dataset class.
    
    Dataset consists of 3 parts; X, Y, Z.
    '''
    def __init__(self, X, Y, Z):
        self.X = X
        self.Y = Y
        self.Z = Z

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        x, y, z = self.X[index], self.Y[index], self.Z[index]
        return x, y, z

def DPDisparity(n_yz, each_z = False):
    '''
    Demographic disparity: max_z{|P(yhat=1|z=z)-P(yhat=1)|}

    Parameters
    ----------
    n_yz : dictionary
        #(yhat=y,z=z)
    each_z : Bool
        Returns for each sensitive group or max
    '''
    z_set = list(set([z for _, z in n_yz.keys()]))
    
    dp = []
    p11 = sum([n_yz[(1,z)] for z in z_set]) / sum([n_yz[(1,z)]+n_yz[(0,z)] for z in z_set])
    for z in z_set:
        try:
            dp_z = abs(n_yz[(1,z)]/(n_yz[(0,z)] + n_yz[(1,z)]) - p11)
        except ZeroDivisionError:
            if n_yz[(1,z)] == 0: 
                dp_z = 0
            else:
                dp_z = 1
        dp.append(dp_z)
    if each_z:
        return dp
    else:
        return max(dp)

def EIDisparity(n_eyz, each_z = False):
    '''
    Equal improvability disparity: max_z{|P(yhat_max=1|z=z,y_hat=0)-P(yhat_max=1|y_hat=0)|}

    Parameters
    ----------
    n_eyz : dictionary
        #(yhat_max=e,yhat=y,z=z)
    each_z : Bool
        Returns for each sensitive group or max
    '''
    z_set = list(set([z for _,_, z in n_eyz.keys()]))
    
    ei = []
    if sum([n_eyz[(1,0,z)]+n_eyz[(0,0,z)] for z in z_set])==0:
        p10 = 0
    else:
        p10 = sum([n_eyz[(1,0,z)] for z in z_set]) / sum([n_eyz[(1,0,z)]+n_eyz[(0,0,z)] for z in z_set])

    for z in z_set:
        if n_eyz[(1,0,z)] == 0: 
            ei_z = 0
        else:
            ei_z = n_eyz[(1,0,z)]/(n_eyz[(0,0,z)] + n_eyz[(1,0,z)])
        ei.append(abs(ei_z-p10))
    if each_z:
        return ei
    else:
        return max(ei)

def BEDisparity(n_eyz, each_z = False):
    '''
    Bounded effort disparity: max_z{|P(yhat_max=1,y_hat=0|z=z)-P(yhat_max=1,y_hat=0)|}

    Parameters
    ----------
    n_eyz : dictionary
        #(yhat_max=e,yhat=y,z=z)
    each_z : Bool
        Returns for each sensitive group or max
    '''
    z_set = list(set([z for _,_, z in n_eyz.keys()]))
    
    be = []
    if sum([n_eyz[(1,0,z)]+n_eyz[(0,0,z)]+n_eyz[(1,1,z)]+n_eyz[(0,1,z)] for z in z_set])==0:
        p1 = 0
    else:
        p1 = sum([n_eyz[(1,0,z)] for z in z_set]) / sum([n_eyz[(1,0,z)]+n_eyz[(0,0,z)]+n_eyz[(1,1,z)]+n_eyz[(0,1,z)] for z in z_set])
    for z in z_set:
        if n_eyz[(1,0,z)] == 0: 
            be_z = 0
        else:
            be_z = n_eyz[(1,0,z)]/(n_eyz[(1,0,z)]+n_eyz[(0,0,z)]+n_eyz[(1,1,z)]+n_eyz[(0,1,z)])
        be.append(abs(be_z-p1))
    if each_z:
        return be
    else:
        return max(be)

def EODisparity(n_eyz, each_z = False):
    '''
    Equal opportunity disparity: max_z{|P(yhat=1|z=z,y=1)-P(yhat=1|y=1)|}

    Parameters
    ----------
    n_eyz : dictionary
        #(yhat=e,y=y,z=z)
    each_z : Bool
        Returns for each sensitive group or max
    '''
    z_set = list(set([z for _,_, z in n_eyz.keys()]))
    
    eod = []
    p11 = sum([n_eyz[(1,1,z)] for z in z_set]) / sum([n_eyz[(1,1,z)]+n_eyz[(0,1,z)] for z in z_set])
    for z in z_set:
        try:
            eod_z = abs(n_eyz[(1,1,z)]/(n_eyz[(0,1,z)] + n_eyz[(1,1,z)]) - p11)
        except ZeroDivisionError:
            if n_eyz[(1,1,z)] == 0: 
                eod_z = 0
            else:
                eod_z = 1
        eod.append(eod_z)
    if each_z:
        return eod
    else:
        return max(eod)
    
def EODDDisparity(n_eyz, each_z = False):
    '''
    Equalized odds disparity: max_z_y{|P(yhat=1|z=z,y=y)-P(yhat=1|y=y)|}

    Parameters
    ----------
    n_eyz: dictionary
        #(yhat=e,y=y,z=z)
    each_z : Bool
        Returns for each sensitive group or max
    '''
    z_set = list(set([z for _,_,z in n_eyz.keys()]))
    y_set = list(set([y for _,y,_ in n_eyz.keys()]))
    
    eoddd = []
    for y in y_set:
        p = sum([n_eyz[(1,y,z)] for z in z_set]) / sum([n_eyz[(1,y,z)]+n_eyz[(0,y,z)] for z in z_set])
        for z in z_set:
            try:
                eoddd_z = abs(n_eyz[(1,y,z)]/(n_eyz[(0,y,z)] + n_eyz[(1,y,z)]) - p)
            except ZeroDivisionError:
                if n_eyz[(1,y,z)] == 0: 
                    eoddd_z = 0
                else:
                    eoddd_z = 1
            eoddd.append(eoddd_z)
    if each_z:
        return eoddd
    else:
        return max(eoddd)

def model_performance(Y, Z, Yhat, Yhat_max, tau):
    Ypred = (Yhat>tau)*1
    Ypred_max = (Yhat_max>tau)*1
    acc = np.mean(Y==Ypred)

    n_yz = {}
    n_eyz = {}
    n_mez = {}

    for y_hat in [0,1]: 
        for y in [0,1]:
            for z in [0,1]:
                n_eyz[(y_hat,y,z)] = np.sum((Ypred==y_hat)*(Y==y)*(Z==z))
                n_mez[(y_hat,y,z)] = np.sum((Ypred_max==y_hat)*(Ypred==y)*(Z==z))
    
    for y in [0,1]:
        for z in [0,1]:
            n_yz[(y,z)] = np.sum((Ypred==y)*(Z==z))

    return acc, DPDisparity(n_yz), EODisparity(n_eyz), EODDDisparity(n_eyz), EIDisparity(n_mez), BEDisparity(n_mez)


def aw_calc(z, Yhat, y_true):
    """
    calculate application welfare
    """
    Yhat_0, Yhat_1 = Yhat[z==0], Yhat[z==1]
    y_true_0, y_true_1 = y_true[z==0], y_true[z==1]
    s0 = Yhat_0 - y_true_0
    s1 = Yhat_1 - y_true_1
    res_0 = np.where(s0 > 0, 0, s0)
    res_1 = np.where(s1 > 0, 0, s1)
    # res_2 = np.where(Yhat > y_true, 1, 0)
    aw_a, aw_b = np.mean(res_0), np.mean(res_1)
    aw_all = (len(Yhat_0)*aw_a + len(Yhat_1)*aw_b)/len(Yhat)
    return aw_a, aw_b, aw_all



def generate_res():
    test = {'accuracy':[],
            'imp_all':[],
            'imp_a':[],
            'imp_b':[],
            'safety':[],
            'aw_a':[],
            'aw_b':[],
            'aw_all':[],
            'ei_disparity':[],
            'be_disparity':[],
            'dp_disparity':[],
            'eo_disparity':[],
            'eodd_disparity':[]}
    
    train = {'accuracy':[],
            'imp_all':[],
            'imp_a':[],
            'imp_b':[],
            'safety':[],
            'aw_a':[],
            'aw_b':[],
            'aw_all':[],
            'ei_disparity':[],
            'be_disparity':[],
            'dp_disparity':[],
            'eo_disparity':[],
            'eodd_disparity':[]}
    
    val = {'accuracy':[],
            'imp_all':[],
            'imp_a':[],
            'imp_b':[],
            'safety':[],
            'aw_a':[],
            'aw_b':[],
            'aw_all':[],
            'ei_disparity':[],
            'be_disparity':[],
            'dp_disparity':[],
            'eo_disparity':[],
            'eodd_disparity':[]}
    
    return train, val, test

def append_res(l,i_all, ia, ib,sfty, awa,awb, awall,acc,ei,be,dp,eo,eodd):
    l['imp_all'].append(i_all)
    l['imp_a'].append(ia)
    l['imp_b'].append(ib)
    l['safety'].append(sfty)
    l['aw_a'].append(awa)
    l['aw_b'].append(awb)
    l['aw_all'].append(awall)
    l['accuracy'].append(acc)
    l['ei_disparity'].append(ei)
    l['be_disparity'].append(be)
    l['dp_disparity'].append(dp)
    l['eo_disparity'].append(eo)
    l['eodd_disparity'].append(eodd)

def get_res(l):
    res = {}
    res['imp_all_mean'] = np.mean(l['imp_all'])
    res['imp_all_var'] = np.std(l['imp_all'])
    res['imp_all_list'] = l['imp_all']
    res['imp_a_mean'] = np.mean(l['imp_a'])
    res['imp_a_var'] = np.std(l['imp_a'])
    res['imp_a_list'] = l['imp_a']
    res['imp_b_mean'] = np.mean(l['imp_b'])
    res['imp_b_var'] = np.std(l['imp_b'])
    res['imp_b_list'] = l['imp_b']
    res['safety_mean'] = np.mean(l['safety'])
    res['safety_var'] = np.std(l['safety'])
    res['safety_list'] = l['safety']  
    res['swf_mean'] = res['safety_mean'] + res['imp_all_mean']
    res['swf_var'] = res['safety_var'] + res['imp_all_var']
    res['swf_list'] = l['safety'] + l['imp_all']
    res['aw_b_mean'] = np.mean(l['aw_b'])
    res['aw_b_var'] = np.var(l['aw_b'])
    res['aw_b_list'] = l['aw_b']
    res['aw_a_mean'] = np.mean(l['aw_a'])
    res['aw_a_var'] = np.var(l['aw_a'])
    res['aw_a_list'] = l['aw_a'] 
    res['aw_all_mean'] = np.mean(l['aw_all'])
    res['aw_all_var'] = np.var(l['aw_all'])
    res['aw_all_list'] = l['aw_all'] 
    res['accuracy_mean'] = np.mean(l['accuracy'])
    res['accuracy_var'] = np.std(l['accuracy'])
    res['accuracy_list'] = l['accuracy']
    res['ei_mean'] = np.mean(l['ei_disparity'])
    res['ei_var'] = np.std(l['ei_disparity'])
    res['ei_list'] = l['ei_disparity']
    res['be_mean'] = np.mean(l['be_disparity'])
    res['be_var'] = np.std(l['be_disparity'])
    res['be_list'] = l['be_disparity']
    res['dp_mean'] = np.mean(l['dp_disparity'])
    res['dp_var'] = np.std(l['dp_disparity'])
    res['dp_list'] = l['dp_disparity']
    res['eo_mean'] = np.mean(l['eo_disparity'])
    res['eo_var'] = np.std(l['eo_disparity'])
    res['eo_list'] = l['eo_disparity']
    res['eodd_mean'] = np.mean(l['eodd_disparity'])
    res['eodd_var'] = np.std(l['eodd_disparity'])
    res['eodd_list'] = l['eodd_disparity']
    return res

