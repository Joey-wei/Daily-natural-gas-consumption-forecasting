# coding=utf-8

import pandas as pd
import numpy as np
import xlwt as xlwt
from sklearn.preprocessing import MinMaxScaler

class Pca():
    def __init__(self):
        pass

    def zeroMean(self, dataMat):
        meanVal = np.mean(dataMat, axis=0)
        newData = dataMat - meanVal
        return newData, meanVal

    def percentage2n(self, eigVals, percentage):
        sortArray = np.sort(eigVals)
        sortArray = sortArray[-1::-1]
        arraySum = sum(sortArray)
        tmpSum = 0
        num = 0
        for i in sortArray:
            tmpSum += i
            num += 1
            if tmpSum >= arraySum * percentage:
                return num

    def pca_done(self, dataMat, percentage=1):
        newData, meanVal = self.zeroMean(dataMat)
        covMat = np.cov(newData, rowvar=0)
        eigVals, eigVects = np.linalg.eig(np.mat(covMat))
        if(percentage==1):
            n = len(eigVals)
        else:
            n = self.percentage2n(eigVals, percentage)

        eigValIndice = np.argsort(eigVals)
        n_eigValIndice = eigValIndice[-1:-(n + 1):-1]
        n_eigVect = eigVects[:, n_eigValIndice]
        lowDDataMat = newData * n_eigVect
        reconMat = (lowDDataMat * n_eigVect.T) + meanVal
        return lowDDataMat, reconMat, n, eigVals


if __name__ == '__main__':
    pca = Pca()
    sheet_name = 'LD'
    df = pd.read_excel('FS_data.xlsx', sheet_name, header=0, encoding='utf-8')  # read data
    data_Length =22
    dataMat = df.iloc[:, 2:(data_Length+2)].values  # get original dataset，type:np.array
    scaler = MinMaxScaler(feature_range=(0.1, 0.9))
    data = scaler.fit_transform(dataMat)
    n = len(data)
    m = len(data[0])
    lowDDataMat, reconMat, kValue, eigValue = pca.pca_done(data, 1)

    '''---------------------------Outputs：Reconstructed Dataset，PCA components，eigen value--------------------------------------'''
    workBook = xlwt.Workbook()
    sheet1 = workBook.add_sheet('reconData')
    sheet2 = workBook.add_sheet('lowDData')
    sheet3 = workBook.add_sheet('eigVData')
    reconData = np.array(reconMat).flatten().tolist()
    lowDData = np.array(lowDDataMat).flatten().tolist()
    eigVData = np.array(eigValue).flatten().tolist()

    for i1 in range(n):
        for j1 in range(kValue):
            sheet2.write(i1, j1, lowDData[i1 * kValue + j1].real)

    for i2 in range(n):
        for j2 in range(m):
            sheet1.write(i2, j2, reconData[i2 * m + j2].real)

    for i3 in range(kValue):
        sheet3.write(i3, 0, eigVData[i3].real)

    workBook.save("PCA_"+sheet_name+".xls")
