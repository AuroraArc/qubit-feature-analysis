import os
import warnings
import pandas as pd
import numpy as np
import datetime
from pandas.errors import SettingWithCopyWarning
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso, LassoCV, ElasticNet, LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

save_path = '/Users/home/Desktop/college/fa23/research/src/notebooks/data/brisbane'

timeframe_dict = {}

for file in os.listdir(save_path):
    if file.endswith('-qubits.csv'):
        qubit_data = pd.read_csv(os.path.join(save_path, file))
        timeframe = file.split('-q')[0]
        if timeframe in timeframe_dict:
            timeframe_dict[timeframe]['qubit_data'] = qubit_data
        else:
            timeframe_dict[timeframe] = {'qubit_data': qubit_data}
    elif file.endswith('-gates.csv'):
        gate_data = pd.read_csv(os.path.join(save_path, file))
        timeframe = file.split('-g')[0]
        if timeframe in timeframe_dict:
            timeframe_dict[timeframe]['gate_data'] = gate_data
        else:
            timeframe_dict[timeframe] = {'gate_data': gate_data}
    elif file.endswith('general.csv'):
        general_data = pd.read_csv(os.path.join(save_path, file))
        timeframe = file.split('-g')[0]
        if timeframe in timeframe_dict:
            timeframe_dict[timeframe]['general_data'] = general_data
        else:
            timeframe_dict[timeframe] = {'general_data': general_data}

flattened_input_data = []
flattened_output_data = []
num_of_days = 315
for i in range(num_of_days):
    date = datetime.datetime(2023, 1, 1) + datetime.timedelta(i)
    dateString = date.strftime('%Y-%m-%d')

    target_gate = 'cx0_1'
    # assoc_gate = 'cx1_0'
    qubit_allowed_values = [0,1]
    timeframe_data = timeframe_dict[dateString]
    qubit_data_for_timeframe = timeframe_data['qubit_data']
    gate_data_for_timeframe = timeframe_data['gate_data']
    general_data_for_timeframe = timeframe_data['general_data']

    qubit_data_for_timeframe = qubit_data_for_timeframe[qubit_data_for_timeframe['Qubit'].isin(qubit_allowed_values)]
    gate_data_for_timeframe = gate_data_for_timeframe[gate_data_for_timeframe['Qubit'].isin(qubit_allowed_values)]
    # tmp_df = gate_data_for_timeframe[(gate_data_for_timeframe['Parameter'] == 'gate_error') & ((gate_data_for_timeframe['Gate Name'] == target_gate) | (gate_data_for_timeframe['Gate Name'] == assoc_gate))].copy()
    tmp_df = gate_data_for_timeframe[(gate_data_for_timeframe['Parameter'] == 'gate_error') & (gate_data_for_timeframe['Gate Name'] == target_gate)].copy()
    flattened_output_data.append(tmp_df.iloc[0]['Value'])
    # gate_data_for_timeframe = gate_data_for_timeframe[~((gate_data_for_timeframe['Parameter'] == 'gate_error') & ((gate_data_for_timeframe['Gate Name'] == target_gate) | (gate_data_for_timeframe['Gate Name'] == assoc_gate)))]
    gate_data_for_timeframe = gate_data_for_timeframe[~((gate_data_for_timeframe['Parameter'] == 'gate_error') & (gate_data_for_timeframe['Gate Name'] == target_gate))]

    # qubit_data_for_timeframe.insert(2, 'Name_Unit', None, True)
    # qubit_data_for_timeframe['Name_Unit'] = qubit_data_for_timeframe['Name'] + '_value'
    qubit_data_for_timeframe['Name'] = qubit_data_for_timeframe['Name'] + '_' + qubit_data_for_timeframe['Qubit'].astype(str)
    # qubit_data_for_timeframe = qubit_data_for_timeframe[['Name', 'Name_Unit', 'Value', 'Unit']]
    #
    # gate_data_for_timeframe.insert(1, 'Gate_Unit', None, True)
    gate_data_for_timeframe['Name'] = gate_data_for_timeframe['Gate Name'] + '_' + gate_data_for_timeframe['Parameter'] + '_value'
    # gate_data_for_timeframe = gate_data_for_timeframe[['Gate Name', 'Gate_Unit', 'Value', 'Unit']]

    qubit_drop = ['Qubit', 'Unit']
    qubit_data_for_timeframe = qubit_data_for_timeframe.drop(columns=qubit_drop)
    gate_drop = ['Qubit', 'Gate', 'Parameter', 'Unit', 'Gate Name']
    gate_data_for_timeframe = gate_data_for_timeframe.drop(columns=gate_drop)
    gate_title = ['Name', 'Value']
    gate_data_for_timeframe = gate_data_for_timeframe.reindex(columns=gate_title)

    # full_data = pd.DataFrame()
    full_data = pd.concat([qubit_data_for_timeframe, gate_data_for_timeframe], ignore_index=False, sort=False)
    full_data = full_data.pivot_table(values='Value', columns=['Name'], aggfunc='first')
    flattened_input_data.append(full_data.values.flatten())

    # qubit_df1 = qubit_data_for_timeframe.iloc[:, :2]
    # qubit_df1 = pd.DataFrame(data=qubit_df1.values.reshape(qubit_df1.shape[0]*2,-1))
    # qubit_df2 = qubit_data_for_timeframe.iloc[:, 2:]
    # qubit_df2 = pd.DataFrame(data=qubit_df2.values.reshape(qubit_df2.shape[0]*2,-1))
    # qubit_data_for_timeframe = pd.concat([qubit_df1,qubit_df2], axis=1)
    # qubit_data_for_timeframe.columns = ['column1', 'column2']
    # qubit_data_for_timeframe = qubit_data_for_timeframe.pivot_table(values='column2', columns=['column1'], aggfunc='first')

    # gate_df1 = gate_data_for_timeframe.iloc[:, :2]
    # gate_df1 = pd.DataFrame(data=gate_df1.values.reshape(gate_df1.shape[0]*2,-1))
    # gate_df2 = gate_data_for_timeframe.iloc[:, 2:]
    # gate_df2 = pd.DataFrame(data=gate_df2.values.reshape(gate_df2.shape[0]*2,-1))
    # gate_data_for_timeframe = pd.concat([gate_df1,gate_df2], axis=1)
    # gate_data_for_timeframe.columns = ['column1', 'column2']
    # gate_data_for_timeframe = gate_data_for_timeframe.pivot_table(values='column2', columns=['column1'], aggfunc='first')

tscv = TimeSeriesSplit(n_splits=5)
scaler = StandardScaler()

l1_ratio = 0.1
X = np.array(flattened_input_data)
y = np.array(flattened_output_data)

# X = scaler.fit_transform(X)
# poly = PolynomialFeatures(degree=2)
# X = poly.fit_transform(X)

results = []
coeff = []
alphas = []
for i in range(1):
    # alphas.append(i/100)
    # for Lasso models
    model = Lasso(alpha=i/100)

    # for ElasticNet models
    # model = ElasticNet(alpha=(i/100), l1_ratio=l1_ratio)

    mse_scores = []
    r2_scores = []



    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.fit_transform(X_test)

        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mse_scores.append(mse)
        r2_scores.append(r2)

        plt.scatter(y_test, y_pred)



    avg_mse = np.mean(mse_scores)
    avg_r2 = np.mean(r2_scores)

    results.append({'Alpha': i/100, 'Avg_MSE': avg_mse, 'Avg_R2': avg_r2})
    coeff.append(model.coef_)

# model = LassoCV(alphas=alphas, cv=tscv)
# model.fit(X, y)
#
# coeff.append(model.coef_)
results_df = pd.DataFrame(results)
results_coeff = pd.DataFrame(coeff)

# results_df.to_csv('lasso_results1.csv', index=True)
# results_df.to_csv('elasticnet_results.csv', index=True)
# results_df.to_csv('poly_lasso_results.csv', index=True)
results_df.to_csv('/Users/home/Desktop/college/fa23/research/src/notebooks/data/brisbane_results/poly_elasticnet_results.csv', index=True)
# results_df.to_csv('linear_results.csv', index=True)
results_coeff.to_csv('/Users/home/Desktop/college/fa23/research/src/notebooks/data/brisbane_results/coefficients.csv', index=True)

plt.show()

print(f'done')
