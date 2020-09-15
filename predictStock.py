import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import RobustScaler

plt.style.use("bmh")
import ta
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

df = pd.read_csv("Apple_June15toJuly19.csv")  #Apple stock prices from January 2017 to July 2019

df['Date'] = pd.to_datetime(df.Date)
df.set_index('Date', inplace=True)  #setting an index
df.dropna(inplace=True) #dropping NaNs

df = ta.add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)   #adding all of the indicators
df.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1, inplace=True)   #dropping indicators except 'Close'

close_scaler = RobustScaler()
close_scaler.fit(df[['Close']])

scaler = RobustScaler()

df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

def split_sequence(seq, n_steps_in, n_steps_out):
    X, y = [], []
    for i in range(len(seq)):
        end = i + n_steps_in
        out_end = end + n_steps_out
        if out_end > len(seq):
            break
        seq_x, seq_y, = seq[i:end, :], seq[end:out_end, 0]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def visualize_training_results(results):
    history = results.history
    plt.figure(figsize=(16,5))
    plt.plot(history['val_loss'])
    plt.plot(history['loss'])
    plt.legend(['val_loss', 'loss'])
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    
    plt.figure(figsize=(16,5))
    plt.plot(history['val_accuracy'])
    plt.plot(history['accuracy'])
    plt.legend(['val_accuracy', 'accuracy'])
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

def layer_maker(n_layers, n_nodes, activation, drop=None, d_rate=.5):
    for x in range(1,n_layers+1):
        model.add(LSTM(n_nodes, activation=activation, return_sequences=True))
        try:
            if x % drop == 0:
                model.add(Dropout(d_rate))
        except:
            pass

def validater(n_per_in, n_per_out):
    predictions = pd.DataFrame(index=df.index, columns=[df.columns[0]])
    for i in range(n_per_in, len(df)-n_per_in, n_per_out):
        x = df[-i - n_per_in:-i]
        yhat = model.predict(np.array(x).reshape(1, n_per_in, n_features))
        yhat = close_scaler.inverse_transform(yhat)[0]
        pred_df = pd.DataFrame(yhat, 
                               index=pd.date_range(start=x.index[-1], 
                                                   periods=len(yhat), 
                                                   freq="B"),
                               columns=[x.columns[0]])
        predictions.update(pred_df)
    return predictions

def val_rmse(df1, df2):
    df = df1.copy()
    df['close2'] = df2.Close
    df.dropna(inplace=True)
    df['diff'] = df.Close - df.close2
    rms = (df[['diff']]**2).mean()
    return float(np.sqrt(rms))

n_per_in  = 90  #How many periods looking back to learn
n_per_out = 30  #How many periods to predict 
n_features = df.shape[1]    #Features
X, y = split_sequence(df.to_numpy(), n_per_in, n_per_out)   #Splitting the data into appropriate sequences

#Neural Network
model = Sequential()
model.add(LSTM(90, activation="tanh", return_sequences=True, input_shape=(n_per_in, n_features))) #input layer
layer_maker(n_layers=1, n_nodes=30, activation="tanh")  #hidden layers
model.add(LSTM(60, activation="tanh"))  #final hidden layer
model.add(Dense(n_per_out)) #output layer
model.summary()

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
res = model.fit(X, y, epochs=150, batch_size=128, validation_split=0.1)

#visualize_training_results(res)

#saving the NN results as .json and .h5
#model_json = model.to_json()
#with open("model.json", "w") as json_file:
    #json_file.write(model_json)
#model.save_weights("model.h5")

#saving the NN model
#import joblib
#filename = 'predictor101.sav'
#joblib.dump(model, filename)
#load_model = joblib.load(filename)
yhat = model.predict(np.array(df.tail(n_per_in)).reshape(1, n_per_in, n_features))

# Transforming the predicted values back to their original format
yhat = close_scaler.inverse_transform(yhat)[0]

# Creating a DF of the predicted prices
preds = pd.DataFrame(yhat, 
                     index=pd.date_range(start=df.index[-1]+timedelta(days=1), 
                                         periods=len(yhat), 
                                         freq="B"), 
                     columns=[df.columns[0]])

# Number of periods back to plot the actual values
pers = n_per_in

# Transforming the actual values to their original price
actual = pd.DataFrame(close_scaler.inverse_transform(df[["Close"]].tail(pers)), 
                      index=df.Close.tail(pers).index, 
                      columns=[df.columns[0]]).append(preds.head(1))

# Printing the predicted prices
print(preds)
# Plotting
plt.figure(figsize=(16,6))
plt.plot(actual, label="Actual Prices")
plt.plot(preds, label="Predicted Prices")
plt.ylabel("Price")
plt.xlabel("Dates")
plt.title(f"Forecasting the next {len(yhat)} days")
plt.legend()
plt.show()
#plt.savefig('Predicted.png')
#plt.close
