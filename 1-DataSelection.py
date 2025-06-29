import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import os

from library import getFullDataFrame, saveSampleDataFrame

# random 100.000 örnek oluşturma

samplesChunk = 100_000  # Her parçadan alınacak örnek sayısı
dataFrame= getFullDataFrame()
selectedDataFrame=dataFrame.sample(n=samplesChunk, random_state=42)

selectedDataFrame.columns = ['Limit'] + [f'Fea{i}' for i in range(1, selectedDataFrame.shape[1])]
# Sayısal değişkenleri seç
numeric_cols = selectedDataFrame.select_dtypes(include=[np.number]).columns

# IQR yöntemi ile aykırı değer kontrolü ve işlenmesi
for col in numeric_cols:
    Q1 = selectedDataFrame[col].quantile(0.25)
    Q3 = selectedDataFrame[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR


    # Aykırı değerleri sınır değerlerle değiştir
    selectedDataFrame[col] = np.where(selectedDataFrame[col] < lower_bound, lower_bound,
                             np.where(selectedDataFrame[col] > upper_bound, upper_bound, selectedDataFrame[col]))


# MinMaxScaler ile [0, 1] aralığına dönüştürme
scaler = MinMaxScaler()
selectedDataFrame[numeric_cols] = scaler.fit_transform(selectedDataFrame[numeric_cols])

# Sonuçları kaydet
saveSampleDataFrame(selectedDataFrame)