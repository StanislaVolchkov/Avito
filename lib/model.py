from typing import Tuple, Union
import pandas as pd
from lightautoml.automl.presets.text_presets import TabularNLPAutoML
from lightautoml.tasks import Task
from lightautoml.report import ReportDecoNLP
from navec import Navec
path = 'navec_hudlit_v1_12B_500K_300d_100q.tar'
navec = Navec.load(path)

def task1(df):
    
  automl2 = load_model(torch.load(f"/content/gdrive/MyDrive/Avito_weights.pth"))
  pred = automl_gpu.predict(test)
  prediction = pd.DataFrame( {'prediction': list(pred.data[:, 0])}, index = range(len(df.index)))

  return prediction


def task2(description: str) -> Union[Tuple[int, int], Tuple[None, None]]:
    description_size = len(description)
    if description_size % 2 == 0:
        return None, None
    else:
        return 0, description_size
