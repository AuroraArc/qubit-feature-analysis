import os
import warnings
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LassoCV, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
import datetime
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=ConvergenceWarning)

data_path = '/Users/home/Desktop/college/fa23/research/src/notebooks/data/brisbane'

timeframe_dict = {}

for file in os.listdir(data_path):
    if file.endswith('-qubits.csv'):
        qubit_data = pd.read_csv(os.path.join(data_path, file))
        timeframe = file.split('-q')[0]
        if timeframe in timeframe_dict:
            timeframe_dict[timeframe]['qubit_data'] = qubit_data
        else:
            timeframe_dict[timeframe] = {'qubit_data': qubit_data}
    elif file.endswith('-gates.csv'):
        gate_data = pd.read_csv(os.path.join(data_path, file))
        timeframe = file.split('-g')[0]
        if timeframe in timeframe_dict:
            timeframe_dict[timeframe]['gate_data'] = gate_data
        else:
            timeframe_dict[timeframe] = {'gate_data': gate_data}
    elif file.endswith('general.csv'):
        general_data = pd.read_csv(os.path.join(data_path, file))
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

    timeframe_data = timeframe_dict[dateString]
    qubit_data_for_timeframe = timeframe_data['qubit_data']
    gate_data_for_timeframe = timeframe_data['gate_data']
    general_data_for_timeframe = timeframe_data['general_data']

    merged_data = pd.merge(qubit_data_for_timeframe, gate_data_for_timeframe, on='Qubit', how='outer')

    target_gate = 'sx24'
    assoc_gate = 'cx2_1'
    data_output = defaultdict(list)
    data_input = merged_data
    data_output0_found = False
    # data_output1_found = False
    i = 0
    while i < len(data_input):
        is_param_error = data_input['Parameter'][i] == "gate_error"
        # is_param_length = data_input['Parameter'][i] == "gate_length"
        is_gate = data_input['Gate Name'][i] == target_gate
        is_assoc_gate = data_input['Gate Name'][i] == assoc_gate
        if is_param_error and is_gate:
            if not data_output['gate_error']:
                data_output['gate_error'].append(data_input['Value_y'][i])
            data_output0_found = True
            data_input = data_input.drop([i])
            data_input = data_input.reset_index()
            data_input = data_input.drop(columns=['index'])
            i -= 1
        elif is_param_error and is_assoc_gate:
            data_input = data_input.drop([i])
            data_input = data_input.reset_index()
            data_input = data_input.drop(columns=['index'])
            i -= 1
        # elif is_param_length and is_gate:
        #     if not data_output['gate_length']:
        #         data_output['gate_length'].append(data_input['Value_y'][i])
        #     data_output1_found = True
        #     data_input = data_input.drop([i])
        #     data_input = data_input.reset_index()
        #     data_input = data_input.drop(columns=['index'])
        #     i -= 1
        i += 1
    if not data_output0_found:
        raise Exception(f"{target_gate} not found")

    # Columns to be one-hot encoded
    categorical_columns = ['Name', 'Unit_x', 'Gate', 'Parameter', 'Unit_y', 'Gate Name']

    # Columns that do not need to be encoded; numerical-valued ones
    numerical_columns = [col for col in data_input.columns if col not in categorical_columns]

    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    numerical_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical', numerical_transformer, numerical_columns),
            ('categorical', categorical_transformer, categorical_columns)
        ]
    )

    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    data_input_processed = pd.DataFrame(pipeline.fit_transform(data_input).toarray())
    data_output_df = pd.DataFrame.from_dict(dict(data_output), orient='index').T
    flattened_input_data.append(data_input_processed.values.flatten())
    flattened_output_data.append(data_output.get('gate_error'))

    # i want to cry
X = np.array(flattened_input_data)
y = np.array(flattened_output_data)

# poly = PolynomialFeatures(degree=2)
# X = poly.fit_transform(X)

l1_ratio = 0.1
tscv = TimeSeriesSplit()
scaler = StandardScaler()

results = []
coeff = []
count = 0
for i in range(1, 101):
    # model = Lasso(alpha=(i/100))
    model = ElasticNet(alpha=(i/100), l1_ratio=l1_ratio)
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

    count += 1
    print(count)

results_df = pd.DataFrame(results)
results_coeff = pd.DataFrame(coeff)

results_df.to_csv('/Users/home/Desktop/college/fa23/research/src/notebooks/data/brisbane_results/old_elasticnet_results.csv', index=True)
results_coeff.to_csv('/Users/home/Desktop/college/fa23/research/src/notebooks/data/brisbane_results/elasticnet_coefficients.csv', index=True)

plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Multiple Regression')
plt.legend()
plt.show()


print('done')