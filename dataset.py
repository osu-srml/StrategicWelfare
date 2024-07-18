import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from folktables import ACSDataSource, ACSIncome

def arrays_to_tensor(X, Y, Z, XZ, device):
    return torch.FloatTensor(X).to(device), torch.FloatTensor(Y).to(device), torch.FloatTensor(Z).to(device), torch.FloatTensor(XZ).to(device)

class IncomeDataset():
    def __init__(self, device):
        self.device = device

        train_dataset, test_dataset = self.preprocess_income_dataset()

        self.Z_train_ = train_dataset['z']
        self.Y_train_ = train_dataset['y']
        self.X_train_ = train_dataset.drop(labels=['z','y'], axis=1)
        self.Z_test_ = test_dataset['z']
        self.Y_test_ = test_dataset['y']
        self.X_test_ = test_dataset.drop(labels=['z','y'], axis=1)

        self.prepare_ndarray()


    def preprocess_income_dataset(self):
        '''
        Function to load and preprocess Income dataset

        Return
        ------
        train_dataset : dataframe
            train dataset
        test_dataset : dataframe
            test dataset
        '''
        data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
        ca_data = data_source.get_data(states=["CA"], download=True)

        ca_features, ca_labels, _ = ACSIncome.df_to_pandas(ca_data)

        ca_features['SEX'] = ca_features['SEX'].map({2.0: 1, 1.0: 0}).astype(int)

        ca_data = pd.concat([ca_features,ca_labels],axis=1)

        ca_data['PINCP'] = ca_data['PINCP'].map({True: 1, False: 0}).astype(int)
        df = ca_data
        indices = np.random.choice(len(df), 20000, replace=False)
        df = df.iloc[indices]
        df=df.rename(columns = {'SEX':'z'})
        CategoricalFeatures=['COW','MAR', 'OCCP', 'POBP', 'RELP', 'RAC1P']

        # df = pd.get_dummies(df, columns=CategoricalFeatures, drop_first=True)
        df = df.drop(columns=CategoricalFeatures)
        df = df.rename(columns = {'PINCP':'y'})
        y = df['y']
        df = df.drop(columns=['y'])
        df.insert(len(df.columns),column='y',value=y)

        train_dataset, test_dataset = train_test_split(df, test_size=0.2)
        train_dataset = train_dataset.reset_index(drop=True)
        test_dataset = test_dataset.reset_index(drop=True)

        scaler = StandardScaler()

        train_dataset[['AGEP','WKHP','SCHL']] = scaler.fit_transform(train_dataset[['AGEP','WKHP','SCHL']])
        test_dataset[['AGEP','WKHP','SCHL']] = scaler.transform(test_dataset[['AGEP','WKHP','SCHL']])



        return train_dataset, test_dataset

    def prepare_ndarray(self):
        self.X_train = self.X_train_.to_numpy(dtype=np.float64)
        self.Y_train = self.Y_train_.to_numpy(dtype=np.float64)
        self.Z_train = self.Z_train_.to_numpy(dtype=np.float64)
        self.XZ_train = np.concatenate([self.X_train, self.Z_train.reshape(-1,1)], axis=1)

        self.X_test = self.X_test_.to_numpy(dtype=np.float64)
        self.Y_test = self.Y_test_.to_numpy(dtype=np.float64)
        self.Z_test = self.Z_test_.to_numpy(dtype=np.float64)
        self.XZ_test = np.concatenate([self.X_test, self.Z_test.reshape(-1,1)], axis=1)
        
        self.sensitive_attrs = sorted(list(set(self.Z_train)))
        return None

    def get_dataset_in_ndarray(self):
        return (self.X_train, self.Y_train, self.Z_train, self.XZ_train),\
               (self.X_test, self.Y_test, self.Z_test, self.XZ_test)

    def get_dataset_in_tensor(self, validation=False, val_portion=.0):
        X_train_, Y_train_, Z_train_, XZ_train_ = arrays_to_tensor(
            self.X_train, self.Y_train, self.Z_train, self.XZ_train, self.device)
        X_test_, Y_test_, Z_test_, XZ_test_ = arrays_to_tensor(
            self.X_test, self.Y_test, self.Z_test, self.XZ_test, self.device)
        return (X_train_, Y_train_, Z_train_, XZ_train_),\
               (X_test_, Y_test_, Z_test_, XZ_test_)

    # def set_improvable_features(self):
    #     self.U_index = np.setdiff1d(np.arange(785),[1])
    #     self.C_index = [1]
    #     self.C_min = [1]
    #     self.C_max = [24]

class GermanDataset():
    def __init__(self, device):
        self.device = device

        train_dataset, test_dataset = self.preprocess_german_dataset()

        self.Z_train_ = train_dataset['z']
        self.Y_train_ = train_dataset['y']
        self.X_train_ = train_dataset.drop(labels=['z','y'], axis=1)
        self.Z_test_ = test_dataset['z']
        self.Y_test_ = test_dataset['y']
        self.X_test_ = test_dataset.drop(labels=['z','y'], axis=1)

        self.prepare_ndarray()


    def preprocess_german_dataset(self):
        '''
        Function to load and preprocess German dataset

        Return
        ------
        train_dataset : dataframe
            train dataset+
        test_dataset : dataframe
            test dataset
        '''
        dataset = pd.read_csv('data/german.data',header = None, delim_whitespace = True)

        dataset.columns=['Existing-Account-Status','Month-Duration','Credit-History','Purpose','Credit-Amount','Saving-Account','Present-Employment','Instalment-Rate','Sex','Guarantors','Residence','Property','Age','Installment','Housing','Existing-Credits','Job','Num-People','Telephone','Foreign-Worker','Status']
        dataset.head(5)

        CategoricalFeatures=['Credit-History','Purpose','Present-Employment', 'Sex','Guarantors','Property','Installment','Telephone','Foreign-Worker','Existing-Account-Status','Saving-Account','Housing','Job']

        NumericalFeatures =['Month-Duration','Credit-Amount']

        data_encode=dataset.copy()
        label_encoder = LabelEncoder()
        for x in CategoricalFeatures:
            data_encode[x]=label_encoder.fit_transform(data_encode[x])
            data_encode[x].unique()
        data_encode.head(5)

        data_encode.loc[data_encode['Age']<=30,'Age'] = 0
        data_encode.loc[data_encode['Age']>30,'Age'] = 1


        data_encode=data_encode.rename(columns = {'Age':'z'})

        data_encode

        data_encode.loc[data_encode['Status']==2,'Status'] = 0
        data_encode=data_encode.rename(columns = {'Status':'y'})
        data_encode

        scaler = StandardScaler()

        train_dataset = data_encode[:800].copy()
        train_dataset[NumericalFeatures] = scaler.fit_transform(train_dataset[NumericalFeatures])
        test_dataset = data_encode[800:].copy()
        test_dataset[NumericalFeatures] = scaler.transform(test_dataset[NumericalFeatures])
        
        return train_dataset, test_dataset
        

    def prepare_ndarray(self):
        self.X_train = self.X_train_.to_numpy(dtype=np.float64)
        self.Y_train = self.Y_train_.to_numpy(dtype=np.float64)
        self.Z_train = self.Z_train_.to_numpy(dtype=np.float64)
        self.XZ_train = np.concatenate([self.X_train, self.Z_train.reshape(-1,1)], axis=1)

        self.X_test = self.X_test_.to_numpy(dtype=np.float64)
        self.Y_test = self.Y_test_.to_numpy(dtype=np.float64)
        self.Z_test = self.Z_test_.to_numpy(dtype=np.float64)
        self.XZ_test = np.concatenate([self.X_test, self.Z_test.reshape(-1,1)], axis=1)
        
        self.sensitive_attrs = sorted(list(set(self.Z_train)))
        return None

    def get_dataset_in_ndarray(self):
        return (self.X_train, self.Y_train, self.Z_train, self.XZ_train),\
               (self.X_test, self.Y_test, self.Z_test, self.XZ_test)

    def get_dataset_in_tensor(self, validation=False, val_portion=.0):
        X_train_, Y_train_, Z_train_, XZ_train_ = arrays_to_tensor(
            self.X_train, self.Y_train, self.Z_train, self.XZ_train, self.device)
        X_test_, Y_test_, Z_test_, XZ_test_ = arrays_to_tensor(
            self.X_test, self.Y_test, self.Z_test, self.XZ_test, self.device)
        return (X_train_, Y_train_, Z_train_, XZ_train_),\
               (X_test_, Y_test_, Z_test_, XZ_test_)

    # def set_improvable_features(self):
    #     self.U_index = [3,8,12,16,17,18,19]
    #     self.S_index = [4,5,13,15]
    #     self.C_index = [3,7,9]
    #     self.C_min = [0,0,0]
    #     self.C_max = [4,2,3]


class SyntheticDataset():
    def __init__(self, device):
        self.device = device

        train_dataset, test_dataset = self.createData()

        self.Z_train_ = train_dataset['z']
        self.Y_train_ = train_dataset['y']
        self.X_train_ = train_dataset.drop(labels=['z','y'], axis=1)
        self.Z_test_ = test_dataset['z']
        self.Y_test_ = test_dataset['y']
        self.X_test_ = test_dataset.drop(labels=['z','y'], axis=1)

        self.prepare_ndarray()


    def createData(self, train_samples=16000, test_samples=4000, z1_mean=0.3, z2_mean=0.5):
    
        num_samples = train_samples + test_samples
        
        xs, ys, zs = [], [], []

        x_dist = {(1,1): {'mean': (0.4,0.3), 'cov': np.array([[0.1,0.0], [0.0,0.1]])},
              (1,0): {'mean': (0.1,0.4), 'cov': np.array([[0.2,0.0], [0.0,0.2]])},
              (0,0):  {'mean':(-0.1,-0.2), 'cov': np.array([[0.4,0.0], [0.0,0.4]])},
              (0,1): {'mean': (-0.2,-0.3), 'cov': np.array([[0.2,0.0], [0.0,0.2]])}}
        y_means = [z1_mean, z2_mean]
        np.random.seed(0)
        for i in range(num_samples):
            z = np.random.binomial(n = 1, p = 0.4, size = 1)[0]
            y = np.random.binomial(n = 1, p = y_means[z], size = 1)[0]
            x = np.random.multivariate_normal(mean = x_dist[(y,z)]['mean'], cov = x_dist[(y,z)]['cov'], size = 1)[0]
            xs.append(x)
            ys.append(y)
            zs.append(z)

        data = pd.DataFrame(zip(np.array(xs).T[0], np.array(xs).T[1], ys, zs), columns = ['x1', 'x2', 'y', 'z'])
        # data = data.sample(frac=1).reset_index(drop=True)
        train_dataset = data[:train_samples].copy()
        test_dataset = data[train_samples:].copy()
        #scaler = StandardScaler()
        #train_dataset[['x1','x2']] = scaler.fit_transform(train_dataset[['x1','x2']])
        #test_dataset[['x1','x2']] = scaler.transform(test_dataset[['x1','x2']])
        return train_dataset, test_dataset        

    def prepare_ndarray(self):
        self.X_train = self.X_train_.to_numpy(dtype=np.float64)
        self.Y_train = self.Y_train_.to_numpy(dtype=np.float64)
        self.Z_train = self.Z_train_.to_numpy(dtype=np.float64)
        self.XZ_train = np.concatenate([self.X_train, self.Z_train.reshape(-1,1)], axis=1)

        self.X_test = self.X_test_.to_numpy(dtype=np.float64)
        self.Y_test = self.Y_test_.to_numpy(dtype=np.float64)
        self.Z_test = self.Z_test_.to_numpy(dtype=np.float64)
        self.XZ_test = np.concatenate([self.X_test, self.Z_test.reshape(-1,1)], axis=1)
        
        self.sensitive_attrs = sorted(list(set(self.Z_train)))
        return None

    def get_dataset_in_ndarray(self):
        return (self.X_train, self.Y_train, self.Z_train, self.XZ_train),\
               (self.X_test, self.Y_test, self.Z_test, self.XZ_test)

    def get_dataset_in_tensor(self, validation=False, val_portion=.0):
        X_train_, Y_train_, Z_train_, XZ_train_ = arrays_to_tensor(
            self.X_train, self.Y_train, self.Z_train, self.XZ_train, self.device)
        X_test_, Y_test_, Z_test_, XZ_test_ = arrays_to_tensor(
            self.X_test, self.Y_test, self.Z_test, self.XZ_test, self.device)
        return (X_train_, Y_train_, Z_train_, XZ_train_),\
               (X_test_, Y_test_, Z_test_, XZ_test_)




class QuadraticDataset():
    def __init__(self, device):
        self.device = device

        train_dataset, test_dataset = self.createData()

        self.Z_train_ = train_dataset['z']
        self.Y_train_ = train_dataset['y']
        self.X_train_ = train_dataset.drop(labels=['z','y'], axis=1)
        self.Z_test_ = test_dataset['z']
        self.Y_test_ = test_dataset['y']
        self.X_test_ = test_dataset.drop(labels=['z','y'], axis=1)

        self.prepare_ndarray()


    def createData(self, train_samples=16000, test_samples=4000, a=-2, b=2):
    
        num_samples = train_samples + test_samples
        
        xs, ys, zs = [], [], []

        x_dist = {0: {'mean': (0.15,0.15), 'cov': np.array([[0.1,0.0], [0.0,0.1]])},
              1: {'mean': (0.85,0.85), 'cov': np.array([[0.1,0.0], [0.0,0.1]])},}

        np.random.seed(0)
        for i in range(num_samples):
            z = np.random.binomial(n = 1, p = 0.2, size = 1)[0]
            x = np.random.multivariate_normal(mean = x_dist[z]['mean'], cov = x_dist[z]['cov'], size = 1)[0]
            y = a*(x[0]**2 + x[1]**2) + b*(x[0] + x[1]) + np.random.normal(loc=0,scale=0.05)
            if y > 0.5:
                y = 1
            else:
                y = 0
            xs.append(x)
            ys.append(y)
            zs.append(z)

        data = pd.DataFrame(zip(np.array(xs).T[0], np.array(xs).T[1], ys, zs), columns = ['x1', 'x2', 'y', 'z'])
        train_dataset = data[:train_samples].copy()
        test_dataset = data[train_samples:].copy()

        return train_dataset, test_dataset        

    def prepare_ndarray(self):
        self.X_train = self.X_train_.to_numpy(dtype=np.float64)
        self.Y_train = self.Y_train_.to_numpy(dtype=np.float64)
        self.Z_train = self.Z_train_.to_numpy(dtype=np.float64)
        self.XZ_train = np.concatenate([self.X_train, self.Z_train.reshape(-1,1)], axis=1)

        self.X_test = self.X_test_.to_numpy(dtype=np.float64)
        self.Y_test = self.Y_test_.to_numpy(dtype=np.float64)
        self.Z_test = self.Z_test_.to_numpy(dtype=np.float64)
        self.XZ_test = np.concatenate([self.X_test, self.Z_test.reshape(-1,1)], axis=1)
        
        self.sensitive_attrs = sorted(list(set(self.Z_train)))
        return None

    def get_dataset_in_ndarray(self):
        return (self.X_train, self.Y_train, self.Z_train, self.XZ_train),\
               (self.X_test, self.Y_test, self.Z_test, self.XZ_test)

    def get_dataset_in_tensor(self, validation=False, val_portion=.0):
        X_train_, Y_train_, Z_train_, XZ_train_ = arrays_to_tensor(
            self.X_train, self.Y_train, self.Z_train, self.XZ_train, self.device)
        X_test_, Y_test_, Z_test_, XZ_test_ = arrays_to_tensor(
            self.X_test, self.Y_test, self.Z_test, self.XZ_test, self.device)
        return (X_train_, Y_train_, Z_train_, XZ_train_),\
               (X_test_, Y_test_, Z_test_, XZ_test_)

