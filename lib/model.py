from typing import Tuple, Union
import pandas as pd
from lightautoml.automl.presets.text_presets import TabularNLPAutoML
from lightautoml.tasks import Task
from lightautoml.report import ReportDecoNLP
from navec import Navec

class task1():
    
    def __init__(self, df):
        self.df = df
        
    @property
    def train(self):
        path = 'navec_hudlit_v1_12B_500K_300d_100q.tar'
        navec = Navec.load(path)
        train, test = train_test_split(self.df, train_size = 0.8, stratify=data.iloc[:, -1])
        
        roles = {'target':'is_bad',
         'text': 'description'}

        task = Task('binary')

        automl_gpu = TabularNLPAutoML(task = task,
                                      timeout = 3600,
                                      gpu_ids = 'all',
                                      text_params={'lang':'ru'})
        
        oof_pred_gpu = automl_gpu.fit_predict(train.iloc[: , [1, -1]], roles=roles, verbose=-1)
        not_nan_gpu = np.any(~np.isnan(oof_pred_gpu.data), axis = 1)
        
        print('Train AUC Score: {}'.format(roc_auc_score(train[roles['target']].values[not_nan_gpu], \
                                                 oof_pred_gpu.data[not_nan_gpu][:, 0])))
        
    @property
    def test(self):
        
        automl2 = torch.load(f"/content/gdrive/MyDrive/Avito_weights.pth")
        pred = automl_gpu.predict(self.df)
        prediction = pd.DataFrame( {'prediction': list(pred.data[:, 0])}, index = range(len(df.index)))
        
        return prediction['prediction']

def task2(description: str) -> Union[Tuple[int, int], Tuple[None, None]]:
    description_size = len(description)
    if description_size % 2 == 0:
        return None, None
    else:
        return 0, description_size
