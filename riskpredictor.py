import pandas as pd  
import tensorflow as tf

data = pd.read_csv("D:\\DATA\Data\\99 _data.csv", low_memory =False)
# View the first few rows to understand it
pd.set_option('display.max_columns', None)
# print(data.head())
# print(data.columns.tolist())

# print(data['FEMALE'])
# print(filtered_data.isnull().sum())

dx_columns = ['I10_DX1', 'I10_DX2', 'I10_DX3' , 'I10_DX4', 'I10_DX5', 'I10_DX6', 'I10_DX7', 'I10_DX8', 'I10_DX9', 'I10_DX10', 'I10_DX11', 'I10_DX12', 'I10_DX13', 'I10_DX14', 'I10_DX15', 'I10_DX16', 'I10_DX17', 'I10_DX18', 'I10_DX19', 'I10_DX20', 'I10_DX21', 'I10_DX22', 'I10_DX23', 'I10_DX24', 'I10_DX25', 'I10_DX26', 'I10_DX27', 'I10_DX28', 'I10_DX29', 'I10_DX30', 'I10_DX31', 'I10_DX32', 'I10_DX33', 'I10_DX34', 'I10_DX35', 'I10_DX36', 'I10_DX37', 'I10_DX38', 'I10_DX39', 'I10_DX40' ]
filtered_data = data[data[dx_columns].astype(str).apply(lambda x: x.str.startswith('G3', na = False)).any(axis=1)]
print(filtered_data)
filtered_data = filtered_data[['AGE', 'FEMALE', 'RACE']]
print(filtered_data)




# print(filtered_data)

data_encoded = pd.get_dummies(filtered_data, columns=['AGE', 'FEMALE', 'RACE'])
print(data_encoded)

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# filtered_data['AGE'] = scaler.fit_transform(data_encoded[['AGE']])

# from sklearn.model_selection import train_test_split

# X = data_encoded.drop(columns=['APRDRG_Severity'])
# y = data_encoded['APRDRG_Severity']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


