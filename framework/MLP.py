from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

def split_sequence(sequence, m_steps_in, n_steps_out):
    X, y = [], []
    for i in range(len(sequence)):

        end_element_index = i + n_steps_in
        out_end_index = end_element_index + n_steps_out
        if out_end_index > len(sequence): 
            break
        
        sequence_x, sequence_y = sequence[i:end_element_index], sequence[end_element_index:out_end_index]
        X.append(sequence_x)
        y.append(sequence_y)

    return np.array(X), np.array(y)

def vector_output_model(n_steps_in, n_steps_out, X, y, epochs_num):
    model = Sequential()
    model.add(Dense(100, activation='relu', input_dim=n_steps_in))
    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')
    
    model.fit(X, y, epochs=2000, verbose=0)
    return model

if __name__ == '__main__':
    epochs_num = 2000
    n_steps_in, n_steps_out = 3, 2
    
    raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
    
    print(X.shape, y.shape)
    for i in range(len(X)):
        print(X[i], y[i])
    
    model = vector_output_model(n_steps_in, n_steps_out, X, y, epochs_num)
    
    x_input = np.array([70, 80, 90])
    x_input = x_input.reshape((1, n_steps_in))
    yhat = model.predict(x_input, verbose=0)
    print(yhat)
