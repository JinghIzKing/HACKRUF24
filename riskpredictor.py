import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

data = pd.read_csv('/ 97 _data.csv')
pd.set_option('display.max_columns', None)

dx_columns = ['I10_DX1', 'I10_DX2', 'I10_DX3' , 'I10_DX4', 'I10_DX5', 'I10_DX6', 'I10_DX7', 'I10_DX8', 'I10_DX9', 'I10_DX10', 'I10_DX11', 'I10_DX12', 'I10_DX13', 'I10_DX14', 'I10_DX15', 'I10_DX16', 'I10_DX17', 'I10_DX18', 'I10_DX19', 'I10_DX20', 'I10_DX21', 'I10_DX22', 'I10_DX23', 'I10_DX24', 'I10_DX25', 'I10_DX26', 'I10_DX27', 'I10_DX28', 'I10_DX29', 'I10_DX30', 'I10_DX31', 'I10_DX32', 'I10_DX33', 'I10_DX34', 'I10_DX35', 'I10_DX36', 'I10_DX37', 'I10_DX38', 'I10_DX39', 'I10_DX40' ]
filtered_data = data[data[dx_columns].astype(str).apply(lambda x: x.str.startswith('G3', na = False)).any(axis=1)]
##print(filtered_data)
filtered_data = filtered_data[['AGE', 'FEMALE', 'RACE', 'APRDRG_Severity']]
filtered_data = filtered_data[filtered_data['APRDRG_Severity'] != 'No class specified']
# print(filtered_data)

filtered_data['APRDRG_Severity'] = filtered_data['APRDRG_Severity'].str.replace(r'\s*\(includes? cases with no comorbidity or complications\)', '', regex=True)
## print(filtered_data.isna().sum())
print(filtered_data['APRDRG_Severity'].unique())
filtered_data = filtered_data.dropna()
## print(filtered_data.isna().sum())
filtered_data['AGE'] = (filtered_data['AGE'] - filtered_data['AGE'].mean()) / filtered_data['AGE'].std()
df_encoded = pd.get_dummies(filtered_data, columns=['FEMALE', 'RACE'])
## print(df_encoded.head())


risk_mapping = {
    "Minor loss of function": 0,
    "Moderate loss of function": 1,
    "Major loss of function": 2
}
df_encoded['APRDRG_Severity'] = df_encoded['APRDRG_Severity'].map(risk_mapping)
x = df_encoded.drop(columns=['APRDRG_Severity'])
print(x)
y = df_encoded['APRDRG_Severity']
print(df_encoded['APRDRG_Severity'])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
y_train = y_train.to_numpy(dtype=int)
# print("Shape of X_train:", X_train.shape)
y_train = y_train.astype(np.int32)
# print("Shape of y_train:", y_train.shape)
# print("Unique values in y_train:", np.unique(y_train))
# print("Unique values in y_test:", np.unique(y_test))


model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation = 'relu' ))
model.add(Dropout(0.5))
model.add(Dense(64, activation ='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation ='softmax'))


model.compile(loss='sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

print(y_test.shape)
print(X_test.shape)
early_stopping = EarlyStopping(monitor='val_loss', patience = 5)
history = model.fit(X_train, y_train, epochs=350, batch_size=5, validation_split=0.2, callbacks=[early_stopping])

# loss, accuracy = model.evaluate(X_test, y_test)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


predictions = model.predict(X_test)
print(X_test)
predicted_classes = np.argmax(predictions, axis=1)
inverse_risk_mapping = {0: 'Low', 1: 'Moderate', 2: 'High'}
predicted_labels = [inverse_risk_mapping[i] for i in predicted_classes]

print(predicted_labels)

new_data = pd.DataFrame({
    'Age': [64],  # Example age
    'Gender_Male': [0],  # 1 for Male, 0 for Female (one-hot encoding)
    'Gender_Female': [1],  # 1 for Hispanic, 0 for others (one-hot encoding)
    'Race_Asian': [1],  
    'Race_Black' : [1],
     'Race-Hispanic': [0],  # 1 for Male, 0 for Female (one-hot encoding)
    'Race_Other': [0],  # 1 for Hispanic, 0 for others (one-hot encoding)
    'Race_White': [1],  
    # 1 for Black, 0 for others (one-hot encoding)
})



# Predict the class
predicted_class = model.predict(new_data)
reverse_risk_mapping = {v: k for k, v in risk_mapping.items()}

predicted_classes = model.predict(new_data)  # Use the appropriate method to get predictions
predicted_classes = predicted_classes.argmax(axis=1)  # If using softmax for multi-class classification

# Now convert the predictions back to labels
# predicted_labels = [reverse_risk_mapping[cls] for cls in predicted_classes]

print("Predicted APRDRG Severity:", predicted_labels[0])


new_df = pd.read_csv('/ 42 _data.csv')
dx_columns = ['I10_DX1', 'I10_DX2', 'I10_DX3' , 'I10_DX4', 'I10_DX5', 'I10_DX6', 'I10_DX7', 'I10_DX8', 'I10_DX9', 'I10_DX10', 'I10_DX11', 'I10_DX12', 'I10_DX13', 'I10_DX14', 'I10_DX15', 'I10_DX16', 'I10_DX17', 'I10_DX18', 'I10_DX19', 'I10_DX20', 'I10_DX21', 'I10_DX22', 'I10_DX23', 'I10_DX24', 'I10_DX25', 'I10_DX26', 'I10_DX27', 'I10_DX28', 'I10_DX29', 'I10_DX30', 'I10_DX31', 'I10_DX32', 'I10_DX33', 'I10_DX34', 'I10_DX35', 'I10_DX36', 'I10_DX37', 'I10_DX38', 'I10_DX39', 'I10_DX40' ]
new_df = new_df[new_df[dx_columns].astype(str).apply(lambda x: x.str.startswith('G3', na = False)).any(axis=1)]
new_df = new_df[['AGE', 'FEMALE', 'RACE']]
new_df['AGE'] = (new_df['AGE'] - new_df['AGE'].mean()) / new_df['AGE'].std()
new_df = pd.get_dummies(new_df, columns=['FEMALE', 'RACE'])

new_df['RACE_Asian'] = False
print(new_df)

predictions = model.predict(new_df)
predicted_classes = np.argmax(predictions, axis=1)

inverse_risk_mapping = {0: 'Low', 1: 'Moderate', 2: 'High'}
predicted_labels = [inverse_risk_mapping[i] for i in predicted_classes]

print(predicted_labels)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
model.save('RiskLevelModel.h5')