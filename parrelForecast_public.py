import pandas as pd
import numpy as np
import random
import xlwt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class paraForecast:

    def __init__(self, label,feature, k):

        self._feature = feature
        self._m, self._n = self._feature.shape
        self._w = np.zeros(self._n)
        self._label = label
        self._k = k

    def Kmeans(self, label, counter):
        k_center = np.sort(random.sample(list(label), self._k))
        temp_label = np.column_stack((label, label*0))
        while counter > 0:
            for i in range(self._m):
                min = 999999
                record = 0
                for j in range(self._k):
                    temp = self.E_distance(temp_label[i, 0], k_center[j])
                    if temp <= min:
                        min = temp
                        record = j
                temp_label[i, -1] = record

            temp_center = []

            for k in range(self._k):
                extract_arr = temp_label[temp_label[:, -1] == k]
                temp_center.append(np.average(extract_arr[:, 0]))

            if abs(np.sum(np.array(k_center)-np.array(temp_center))) < self._k:
                return temp_label
            else:
                k_center = temp_center
                counter = counter-1

        return temp_label

    def seperate(self, label_group):
        extract_label = []
        extract_feature = []

        for k in range(self._k):
            temp_index = []
            extract_arr = label_group[label_group[:, -1] == k]
            temp_label = []
            temp_feature = []
            for i in range(len(extract_arr)):
                temp_index.append(np.argwhere(label_group[:, 0] == extract_arr[i][0]))

            for j in range(len(temp_index)):
                index = temp_index[j][0][0]
                label = np.array(self._label[index])
                feature = np.array(self._feature[index])

                temp_label.append(label)
                temp_feature.append(feature)

            a = np.array(temp_label).flatten()
            b = np.array(temp_feature).flatten()
            shaped_label = np.reshape(a, (-1, 1))
            shaped_feature = np.reshape(b, (-1, self._n))

            extract_label.append(shaped_label)
            extract_feature.append(shaped_feature)

        return extract_label, extract_feature

    def parrelModel(self, extract_label, extract_feature):
        model_group = []
        for k in range(self._k):
            X_train, X_test, y_train, y_test = train_test_split(extract_feature[k], extract_label[k], random_state=666)
            model = LinearRegression()
            model.fit(X_train, y_train)
            model_group.append(model)

        return model_group

    def parrelForecast(self, model_group, Matrix_year, test_label, test_feature):
        m, n = Matrix_year.shape
        predict_all = []
        for i in range(len(test_label)):
            result = 0
            for j in range(n):
                k_j = int(Matrix_year[i][j])
                model_j = model_group[k_j]

                predict_j = model_j.predict(np.array(test_feature[i]).reshape(1, -1))
                result = result + predict_j[0][0]*(j+1)/sum(range(1, n+1))
            predict_all.append(result)
        return np.array(predict_all).flatten()

    def getYearMatrix(self, label_group):
        mark_arr = label_group[:, -1]
        temp = np.delete(mark_arr, 424)
        a = np.reshape(temp, (-1, 365))
        matrix_3ys = a.transpose()
        return matrix_3ys

    def E_distance(self, x, y):
        return abs(x - y)


def out_file(data_group, Doc_name, f_name):

    f = xlwt.Workbook()
    sheet_name = 'Sheet1'
    sh = f.add_sheet(sheet_name, cell_overwrite_ok=False)

    for i in range(len(data_group)):
        predict_data = np.array(data_group[i]).flatten().tolist()
        sh.write(0, i, "f_name"+str(i+2))
        for j in range(len(predict_data)):
            sh.write(j + 1, i, predict_data[j])

    f_xls = f_name + Doc_name + '.xls'
    f.save(f_xls)

if __name__ == '__main__':
    final_result = []
    sheet_name = 'AT_ALL'
    file_name = 'forecastingData.xlsx'
    for i in range(2, 12):
        print(str(i))
        # file_name = 'FS_data.xlsx'
        # sheet_name = 'AT'
        train_length = 1096
        k_value = i
        df = pd.read_excel(file_name, sheet_name, header=0)
        train_feature = np.array(df.iloc[:train_length, 2:])
        test_feature = np.array(df.iloc[train_length:, 2:])
        train_label = df.iloc[:train_length, 1]
        test_label = df.iloc[train_length:, 1]

        f = paraForecast(train_label, train_feature, k_value)
        train_label_group = f.Kmeans(train_label, 1000)
        Matrix_year = f.getYearMatrix(train_label_group) # 365 days in a year


        extract_train_label, extract_train_feature = f.seperate(train_label_group) # subseries
        model_group = f.parrelModel(extract_train_label, extract_train_feature) # model matrix group
        predict_value = f.parrelForecast(model_group, Matrix_year, test_label, test_feature) #parallel forecast


        final_result.append(predict_value)
    f_name = 'outputs/'
    Doc_name = sheet_name +'_WPMA'
    out_file(final_result, Doc_name, f_name)





