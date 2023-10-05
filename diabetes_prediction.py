################################################
# End-to-End Diabetes Machine Learning Pipeline III
################################################
###### diyelim ki biz pipeline oluştueduk ve bu modeli kullanılması için paylaştık. Hastaneye yeni hasta geldi biz bu hastayı bu modele verirsek çalışmaz çünkü biz feature engineering ile yeni değişkenler türettik bu değişkenler hassta verilerinde yok
import joblib
import pandas as pd

df = pd.read_csv("datasets/diabetes.csv")

random_user = df.sample(1, random_state=45)

new_model = joblib.load("voting_clf.pkl")

new_model.predict(random_user)
#buraya kadarki işlemler hata verdi. İlk yorum satırında sebebi açıklandı.
# veri setini tekrar data preprocessing fonksiyonundan geçirip istediğimiz forma getirdik
from diabetes_pipeline import diabetes_data_prep

X, y = diabetes_data_prep(df)

random_user = X.sample(1, random_state=50)

new_model = joblib.load("voting_clf.pkl")

new_model.predict(random_user)
