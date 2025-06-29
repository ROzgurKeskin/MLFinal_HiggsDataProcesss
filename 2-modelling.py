import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
import os

from library import getSampleDataFrameFromFile, saveSelectedFeaturesDataFrame

# ----------------------------------------
# 1.adımda oluşturulan datayı yüklüyoruz
# ----------------------------------------
processedDataFrame= getSampleDataFrameFromFile()

x = processedDataFrame.drop(columns=['Limit'], axis=1)  # hedef değişken dışındaki özellikleri al
y = processedDataFrame['Limit']  # Hedef değişkeni al



# ----------------------------------------
# ANOVA ile en iyi özellikler seçiliyor
# ----------------------------------------
print("En iyi 15 özellik seçiliyor...")
selector = SelectKBest(score_func=f_classif, k=15)
x_selected = selector.fit_transform(x, y)

selected_features = x.columns[selector.get_support()].tolist()

print("Seçilen özellikler:")
for i, feature in enumerate(selected_features, 1):
    print(f"{i:>2}. {feature}")


# ----------------------------------------
# Seçilen veriyi kaydet
# ----------------------------------------
selected_df = pd.concat ([processedDataFrame[['Limit']], pd.DataFrame(x_selected, columns=selected_features)], axis=1)
saveSelectedFeaturesDataFrame(selected_df)

print("Seçilen özelliklerle yeni veri seti kaydedildi: SelectedFeatureData.csv")