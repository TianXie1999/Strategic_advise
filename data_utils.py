import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import torch
from torch.utils.data import Dataset
from torch.autograd import grad
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def arrays_to_tensor(X, Y, Z, XZ, device):
    return torch.FloatTensor(X).to(device), torch.FloatTensor(Y).to(device), torch.FloatTensor(Z).to(device), torch.FloatTensor(XZ).to(device)

class CreditDataset():
    def __init__(self, device, decision):
        self.device = device
        self.decision = decision
        train_dataset, test_dataset = self.preprocess_credit_dataset()
        self.Z_train_ = train_dataset['z']
        self.Y_train_ = train_dataset['y']
        self.X_train_ = train_dataset.drop(labels=['z','y'], axis=1)
        self.Z_test_ = test_dataset['z']
        self.Y_test_ = test_dataset['y']
        self.X_test_ = test_dataset.drop(labels=['z','y'], axis=1)
        self.prepare_ndarray()


    def preprocess_credit_dataset(self):
        '''
        Function to load and preprocess Credit dataset

        Return
        ------
        train_dataset : dataframe
            train dataset+
        test_dataset : dataframe
            test dataset
        '''
        seed = 0
        np.random.seed(seed)
        if self.decision:
            data = pd.read_csv('data/balanced_test_dataset.csv')
            data.drop(['question'], axis=1, inplace=True)
            data.rename(columns={'qualification_status':'SeriousDlqin2yrs'}, inplace=True)
        else:
            data = pd.read_csv('data/cs-training1.csv')
        data.dropna(inplace=True)
        data['z'] = np.where(data['age'] > 35, 0, 1)

        # full data set
        X_all = data.drop(['SeriousDlqin2yrs','age'], axis=1)
        # X_all = data[['RevolvingUtilizationOfUnsecuredLines','DebtRatio', 'MonthlyIncome']]

        self.mean = np.mean(X_all, axis=0)
        self.std_dev = np.std(X_all, axis=0)
        
        # zero mean, unit variance
        for col in X_all.columns:
            X_all[col] = preprocessing.scale(X_all[col])
        
        # outcomes
        Y_all = data['SeriousDlqin2yrs']
        X_all.insert(len(X_all.columns),column='y',value=Y_all)

        # balance classes
        default_indices = np.where(Y_all == 1)[0][:10000]
        other_indices = np.where(Y_all == 0)[0][:10000]
        indices = np.concatenate((default_indices, other_indices))

        df_balanced = X_all.iloc[indices]

        # shuffle arrays
        p = np.random.permutation(len(indices))
        df_full = df_balanced.iloc[p]
        train_dataset, test_dataset = train_test_split(df_full, test_size=0.2)
        train_dataset = train_dataset.reset_index(drop=True)
        test_dataset = test_dataset.reset_index(drop=True)

        return train_dataset, test_dataset
        

    def prepare_ndarray(self):
        self.X_train = self.X_train_.to_numpy()
        self.Y_train = self.Y_train_.to_numpy()
        self.Z_train = self.Z_train_.to_numpy()
        self.XZ_train = np.concatenate([self.X_train, self.Z_train.reshape(-1,1)], axis=1)

        self.X_test = self.X_test_.to_numpy()
        self.Y_test = self.Y_test_.to_numpy()
        self.Z_test = self.Z_test_.to_numpy()
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



class LawDataset():
    def __init__(self, device, decision):
        self.device = device
        self.decision = decision
        train_dataset, test_dataset = self.preprocess_law_dataset()
        self.Z_train_ = train_dataset['z']
        self.Y_train_ = train_dataset['y']
        self.X_train_ = train_dataset.drop(labels=['z','y'], axis=1)
        self.Z_test_ = test_dataset['z']
        self.Y_test_ = test_dataset['y']
        self.X_test_ = test_dataset.drop(labels=['z','y'], axis=1)
        self.prepare_ndarray()


    def preprocess_law_dataset(self):
        '''
        Function to load and preprocess Law dataset

        Return
        ------
        train_dataset : dataframe
            train dataset+
        test_dataset : dataframe
            test dataset
        '''
        seed = 0
        np.random.seed(seed)
        if self.decision:
            data = pd.read_csv('data/bar_pass_data_sample.csv')
        else:
            data = pd.read_csv('data/bar_pass_data.csv')
        # Only use these features 
        data.dropna(inplace=True)
        data['z'] = np.where(data['sex'] == 1.0, 0, 1)
        data['y'] = np.where(data['pass'] == 1, 1, 0)
        data.drop(['sex'], axis=1, inplace=True)
        data.drop(['pass'], axis=1, inplace=True)

        # normalize data
        X_all = data.drop(['y'], axis=1)
        self.mean = np.mean(X_all, axis=0)
        self.std_dev = np.std(X_all, axis=0)
        X_all = preprocessing.scale(X_all)
        Y_all = data['y']
        X_all = pd.DataFrame(X_all, columns=data.drop(['y'], axis=1).columns)
        X_all['y'] = Y_all
        X_all.dropna(inplace=True)
        data = X_all
        
        train_dataset, test_dataset = train_test_split(data, test_size=0.2)
        train_dataset = train_dataset.reset_index(drop=True)
        test_dataset = test_dataset.reset_index(drop=True)

        return train_dataset, test_dataset
        

    def prepare_ndarray(self):
        self.X_train = self.X_train_.to_numpy()
        self.Y_train = self.Y_train_.to_numpy()
        self.Z_train = self.Z_train_.to_numpy()
        self.XZ_train = np.concatenate([self.X_train, self.Z_train.reshape(-1,1)], axis=1)

        self.X_test = self.X_test_.to_numpy()
        self.Y_test = self.Y_test_.to_numpy()
        self.Z_test = self.Z_test_.to_numpy()
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




class ACSIncome():
    def __init__(self, device, decision):
        self.device = device
        self.decision = decision
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
            train dataset+
        test_dataset : dataframe
            test dataset
        '''
        seed = 0
        np.random.seed(seed)
        if self.decision:
            data = pd.read_csv('data/ACSIncome_sample_raw.csv')
        else:
            data = pd.read_csv('data/ACSIncome.csv')
            # sample 10000 data points
            data = data.sample(10000)
        # Only use these features 
        data = data[['AGEP','SCHL','WKHP','SEX','PINCP']]
        data.dropna(inplace=True)
        data['z'] = np.where(data['AGEP'] > 35, 0, 1)
        data['y'] = np.where(data['PINCP'] > 50000, 1, 0)
        data.drop(['AGEP'], axis=1, inplace=True)
        data.drop(['PINCP'], axis=1, inplace=True)
        
        train_dataset, test_dataset = train_test_split(data, test_size=0.2)
        train_dataset = train_dataset.reset_index(drop=True)
        test_dataset = test_dataset.reset_index(drop=True)

        return train_dataset, test_dataset
        
    def prepare_ndarray(self):
        self.X_train = self.X_train_.to_numpy()
        self.Y_train = self.Y_train_.to_numpy()
        self.Z_train = self.Z_train_.to_numpy()
        self.XZ_train = np.concatenate([self.X_train, self.Z_train.reshape(-1,1)], axis=1)

        self.X_test = self.X_test_.to_numpy()
        self.Y_test = self.Y_test_.to_numpy()
        self.Z_test = self.Z_test_.to_numpy()
        self.XZ_test = np.concatenate([self.X_test, self.Z_test.reshape(-1,1)], axis=1)
        
        self.sensitive_attrs = sorted(list(set(self.Z_train)))
        return None

    def get_dataset_in_tensor(self, validation=False, val_portion=.0):
        X_train_, Y_train_, Z_train_, XZ_train_ = arrays_to_tensor(
            self.X_train, self.Y_train, self.Z_train, self.XZ_train, self.device)
        X_test_, Y_test_, Z_test_, XZ_test_ = arrays_to_tensor(
            self.X_test, self.Y_test, self.Z_test, self.XZ_test, self.device)
        return (X_train_, Y_train_, Z_train_, XZ_train_),\
               (X_test_, Y_test_, Z_test_, XZ_test_)


class ACSPublic():
    def __init__(self, device, decision):
        self.device = device
        self.decision = decision
        train_dataset, test_dataset = self.preprocess_public_dataset()
        self.Z_train_ = train_dataset['z']
        self.Y_train_ = train_dataset['y']
        self.X_train_ = train_dataset.drop(labels=['z','y'], axis=1)
        self.Z_test_ = test_dataset['z']
        self.Y_test_ = test_dataset['y']
        self.X_test_ = test_dataset.drop(labels=['z','y'], axis=1)
        self.prepare_ndarray()


    def preprocess_public_dataset(self):
        '''
        Function to load and preprocess Public dataset

        Return
        ------
        train_dataset : dataframe
            train dataset+
        test_dataset : dataframe
            test dataset
        '''
        seed = 0
        np.random.seed(seed)
        if self.decision:
            data = pd.read_csv('data/ACSPublicoverage_sample.csv')
        else:
            data = pd.read_csv('data/ACSPublicoverage.csv')
            data = data.sample(10000)
        # Only use these features 
        data = data[['AGEP','SCHL','PINCP','SEX','PUBCOV']]
        data.dropna(inplace=True)
        data['z'] = np.where(data['AGEP'] > 35, 0, 1)
        data.drop(['AGEP'], axis=1, inplace=True)
        data.rename(columns={'PUBCOV':'y'}, inplace=True)

        # full data set
        X_all = data.drop(['y'], axis=1)

        self.mean = np.mean(X_all, axis=0)
        self.std_dev = np.std(X_all, axis=0)
        
        # zero mean, unit variance
        for col in X_all.columns:
            X_all[col] = preprocessing.scale(X_all[col])
        
        # outcomes
        Y_all = data['y']
        X_all.insert(len(X_all.columns),column='y',value=Y_all)
        data = X_all

        train_dataset, test_dataset = train_test_split(data, test_size=0.2)
        train_dataset = train_dataset.reset_index(drop=True)
        test_dataset = test_dataset.reset_index(drop=True)

        return train_dataset, test_dataset
        
    def prepare_ndarray(self):
        self.X_train = self.X_train_.to_numpy()
        self.Y_train = self.Y_train_.to_numpy()
        self.Z_train = self.Z_train_.to_numpy()
        self.XZ_train = np.concatenate([self.X_train, self.Z_train.reshape(-1,1)], axis=1)

        self.X_test = self.X_test_.to_numpy()
        self.Y_test = self.Y_test_.to_numpy()
        self.Z_test = self.Z_test_.to_numpy()
        self.XZ_test = np.concatenate([self.X_test, self.Z_test.reshape(-1,1)], axis=1)
        
        self.sensitive_attrs = sorted(list(set(self.Z_train)))
        return None

    def get_dataset_in_tensor(self, validation=False, val_portion=.0):
        X_train_, Y_train_, Z_train_, XZ_train_ = arrays_to_tensor(
            self.X_train, self.Y_train, self.Z_train, self.XZ_train, self.device)
        X_test_, Y_test_, Z_test_, XZ_test_ = arrays_to_tensor(
            self.X_test, self.Y_test, self.Z_test, self.XZ_test, self.device)
        return (X_train_, Y_train_, Z_train_, XZ_train_),\
               (X_test_, Y_test_, Z_test_, XZ_test_)


class ACSPAP():
    def __init__(self, device, decision):
        self.device = device
        self.decision = decision
        train_dataset, test_dataset = self.preprocess_pap_dataset()
        self.Z_train_ = train_dataset['z']
        self.Y_train_ = train_dataset['y']
        self.X_train_ = train_dataset.drop(labels=['z','y'], axis=1)
        self.Z_test_ = test_dataset['z']
        self.Y_test_ = test_dataset['y']
        self.X_test_ = test_dataset.drop(labels=['z','y'], axis=1)
        self.prepare_ndarray()


    def preprocess_pap_dataset(self):
        '''
        Function to load and preprocess PAP dataset

        Return
        ------
        train_dataset : dataframe
            train dataset+
        test_dataset : dataframe
            test dataset
        '''
        seed = 0
        np.random.seed(seed)
        if self.decision:
            data = pd.read_csv('data/ACSPAP_sample.csv')
        else:
            data = pd.read_csv('data/ACSPAP.csv')
            data = data.sample(10000)
        # Only use these features 
        data.dropna(inplace=True)
        data['z'] = np.where(data['AGEP'] > 35, 0, 1)
        data.drop(['AGEP'], axis=1, inplace=True)
        data.rename(columns={'PAP':'y'}, inplace=True)
        data['y'] = np.where(data['y'] == True, 1, 0)

        # full data set
        X_all = data.drop(['y'], axis=1)

        self.mean = np.mean(X_all, axis=0)
        self.std_dev = np.std(X_all, axis=0)
        
        # zero mean, unit variance
        for col in X_all.columns:
            X_all[col] = preprocessing.scale(X_all[col])
        
        # outcomes
        Y_all = data['y']
        X_all.insert(len(X_all.columns),column='y',value=Y_all)
        data = X_all

        train_dataset, test_dataset = train_test_split(data, test_size=0.2)
        train_dataset = train_dataset.reset_index(drop=True)
        test_dataset = test_dataset.reset_index(drop=True)

        return train_dataset, test_dataset
        
    def prepare_ndarray(self):
        self.X_train = self.X_train_.to_numpy()
        self.Y_train = self.Y_train_.to_numpy()
        self.Z_train = self.Z_train_.to_numpy()
        self.XZ_train = np.concatenate([self.X_train, self.Z_train.reshape(-1,1)], axis=1)

        self.X_test = self.X_test_.to_numpy()
        self.Y_test = self.Y_test_.to_numpy()
        self.Z_test = self.Z_test_.to_numpy()
        self.XZ_test = np.concatenate([self.X_test, self.Z_test.reshape(-1,1)], axis=1)
        
        self.sensitive_attrs = sorted(list(set(self.Z_train)))
        return None

    def get_dataset_in_tensor(self, validation=False, val_portion=.0):
        X_train_, Y_train_, Z_train_, XZ_train_ = arrays_to_tensor(
            self.X_train, self.Y_train, self.Z_train, self.XZ_train, self.device)
        X_test_, Y_test_, Z_test_, XZ_test_ = arrays_to_tensor(
            self.X_test, self.Y_test, self.Z_test, self.XZ_test, self.device)
        return (X_train_, Y_train_, Z_train_, XZ_train_),\
               (X_test_, Y_test_, Z_test_, XZ_test_)



class Hiring():
    def __init__(self, device, decision):
        self.device = device
        self.decision = decision
        train_dataset, test_dataset = self.preprocess_hiring_dataset()
        self.Z_train_ = train_dataset['z']
        self.Y_train_ = train_dataset['y']
        self.X_train_ = train_dataset.drop(labels=['z','y'], axis=1)
        self.Z_test_ = test_dataset['z']
        self.Y_test_ = test_dataset['y']
        self.X_test_ = test_dataset.drop(labels=['z','y'], axis=1)
        self.prepare_ndarray()


    def preprocess_hiring_dataset(self):
        '''
        Function to load and preprocess Hiring dataset

        Return
        ------
        train_dataset : dataframe
            train dataset+
        test_dataset : dataframe
            test dataset
        '''
        seed = 0
        np.random.seed(seed)
        if self.decision:
            data = pd.read_csv('data/hiring_data_sample.csv')
        else:
            data = pd.read_csv('data/hiring_data.csv')
            data = data.sample(10000)
        # Only use these features 
        data.dropna(inplace=True)
        data['z'] = data['age']
        data.drop(['age'], axis=1, inplace=True)
        data.rename(columns={'Employed':'y'}, inplace=True)

        # full data set
        X_all = data.drop(['y'], axis=1)

        self.mean = np.mean(X_all, axis=0)
        self.std_dev = np.std(X_all, axis=0)
        
        # zero mean, unit variance
        for col in X_all.columns:
            X_all[col] = preprocessing.scale(X_all[col])
        
        # outcomes
        Y_all = data['y']
        X_all.insert(len(X_all.columns),column='y',value=Y_all)
        data = X_all

        train_dataset, test_dataset = train_test_split(data, test_size=0.2)
        train_dataset = train_dataset.reset_index(drop=True)
        test_dataset = test_dataset.reset_index(drop=True)

        return train_dataset, test_dataset
        
    def prepare_ndarray(self):
        self.X_train = self.X_train_.to_numpy()
        self.Y_train = self.Y_train_.to_numpy()
        self.Z_train = self.Z_train_.to_numpy()
        self.XZ_train = np.concatenate([self.X_train, self.Z_train.reshape(-1,1)], axis=1)

        self.X_test = self.X_test_.to_numpy()
        self.Y_test = self.Y_test_.to_numpy()
        self.Z_test = self.Z_test_.to_numpy()
        self.XZ_test = np.concatenate([self.X_test, self.Z_test.reshape(-1,1)], axis=1)
        
        self.sensitive_attrs = sorted(list(set(self.Z_train)))
        return None

    def get_dataset_in_tensor(self, validation=False, val_portion=.0):
        X_train_, Y_train_, Z_train_, XZ_train_ = arrays_to_tensor(
            self.X_train, self.Y_train, self.Z_train, self.XZ_train, self.device)
        X_test_, Y_test_, Z_test_, XZ_test_ = arrays_to_tensor(
            self.X_test, self.Y_test, self.Z_test, self.XZ_test, self.device)
        return (X_train_, Y_train_, Z_train_, XZ_train_),\
               (X_test_, Y_test_, Z_test_, XZ_test_)



class Dataset_prep(Dataset):
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
