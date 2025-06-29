import os
import pandas as pd

# random 100.000 örnek oluşturma

sourceBasePath = "d:\\project\\source\\higgs"  # Kaynak dosyanın yolu
sourceFilePath = f"{sourceBasePath}\\higgs.csv.gz"  # Gzip sıkıştırılmış CSV dosyasının yolu
processedFileName = "SampleHiggsData.csv"  # İşlenmiş dosyanın kaydedileceği yol
selectedFeatureFileName = "SelectedFeatureData.csv" #seçilen özelliklerle yeni bir dosya oluşturulacak

samplesChunk = 100_000  # Her parçadan alınacak örnek sayısı


def getFullDataFrame():
    """
    Returns the full DataFrame loaded from the source file.
    """
    return pd.read_csv(sourceFilePath, compression='gzip', header=None)

def getSampleDataFrameFromFile():
    """
    Returns a sample DataFrame with a specified number of random samples.
    """
    return getDataFrameFromFile(sourceBasePath, processedFileName)

def getSelectedFeatureDataFrame():
    """
    Returns a sample DataFrame with a specified number of random samples.
    """
    return getDataFrameFromFile(sourceBasePath, selectedFeatureFileName)


def saveSampleDataFrame(dataFrame):
    """
    Saves the provided DataFrame to the processed file path.
    """
    saveDataframeToCsvFile(sourceBasePath,processedFileName, dataFrame)

def saveSelectedFeaturesDataFrame(dataFrame):
    saveDataframeToCsvFile(sourceBasePath, selectedFeatureFileName, dataFrame)

def saveDataframeToCsvFile(directory, fileName, dataFrame):
    os.makedirs(directory, exist_ok=True)  # data klasörü yoksa oluştur
    dataFrame.to_csv(f'{directory}\\{fileName}', index=False)  # Seçilen özellikleri kaydet

def getDataFrameFromFile(directory, fileName):
    """
    Returns a DataFrame loaded from a specified CSV file in the given directory.
    """
    return pd.read_csv(f"{directory}\\{fileName}")
