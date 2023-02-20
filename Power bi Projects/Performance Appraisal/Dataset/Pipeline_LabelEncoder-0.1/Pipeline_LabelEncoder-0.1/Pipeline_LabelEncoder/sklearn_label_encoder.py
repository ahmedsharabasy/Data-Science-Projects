from sklearn.preprocessing import LabelEncoder


class PipelineLabelEncoder:
    def __init__(self,columns = None):
        # Array containing column names for label encoding
        self.columns = columns
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        '''
        Here, 'X' indicated input dataframe
        '''
        input_df = X.copy()
        '''
        1. Encodes columns of input dataframe specified in self.columns using LabelEncoder().
        2. Encodes all columns of input dataframe specified if no column is specified.
        '''
        if self.columns is not None:
            for col in self.columns:
                lab_enc = LabelEncoder()
                input_df[col] = lab_enc.fit_transform(input_df[col])
                le_name_mapping = dict(zip(lab_enc.classes_, lab_enc.transform(lab_enc.classes_)))
                print('Feature', col)
                print('mapping', le_name_mapping)
        else:
            for colname,col in input_df.iteritems():
                input_df[colname] = LabelEncoder().fit_transform(col)
        return input_df
    
    def fit_transform(self,X,y=None):
        print('Inside fit transform')
        return self.fit(X,y).transform(X)
