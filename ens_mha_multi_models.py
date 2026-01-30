# Silencing annoying warnings
import shutup
shutup.please()

# Main libraries
import numpy as np
import pandas as pd
import pywt  # For wavelet transformations
from scipy.stats import gamma
import numba as nb
import matplotlib.pyplot as plt
import seaborn as sns
import gc

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras import layers, models, optimizers, callbacks, losses, utils, config, initializers, regularizers
from keras_nlp.layers import TransformerEncoder, TransformerDecoder, SinePositionEncoding, FNetEncoder, PositionEmbedding

# Import libraries for environment variables
import os
import random
from tensorflow.random import set_seed
from tensorflow.random.experimental import get_global_generator
from tensorflow.keras.mixed_precision import Policy, set_global_policy

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore")

import logging
logging.disable(logging.CRITICAL)

# Environment variables
rnd = 5
set_seed(rnd)
random.seed(rnd)
np.random.seed(rnd)
os.environ['PYTHONHASHSEED'] = f'{rnd}'
get_global_generator().reset_from_seed(rnd)
pd.set_option('compute.use_numba', True)
config.enable_flash_attention()
set_global_policy(Policy('mixed_float16'))
    
   
# LSTM Transformer network
def lstm_basic(inputs, ff_dim, dropout, i):
    x = layers.MultiHeadAttention(key_dim=512, num_heads=4, dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout, name=f'lin_drop{i}')(x)
    x = layers.LayerNormalization(epsilon=1e-6, name=f'lin_norm{i}')(x)
    res = layers.Add(name=f'lin_add{i}')([x,inputs])
    # FFN
    x = layers.LSTM(ff_dim*4, return_sequences=True, name=f'ffn_lstm{i}')(res)
    x = layers.Dropout(dropout, name=f'lout_drop{i}')(x)
    x = layers.LSTM(inputs.shape[-1], return_sequences=True, name=f'out_lstm{i}')(x)
    x = layers.LayerNormalization(epsilon=1e-6, name=f'lout_norm{i}')(x)
    return layers.Add(name=f'lout_add{i}')([x,res])
    
    
def make_lstm_model(input_shape, dropout, dims, num_layers):
    # fencoder = FNetEncoder(dims, dropout=dropout, activation=activation)
    inputs = layers.Input(input_shape)
    x = norm(inputs)
    # pos_enc = SinePositionEncoding()(x)
    # x = layers.Add(name='model_add')([x, pos_enc])
    x = layers.LSTM(dims, return_sequences=True, name='model_lstm')(x)
    for i in range(num_layers):
        x = lstm_basic(x, dims, dropout, i)
    x = layers.Flatten(name='lmodel_flat')(x)
    # flat_dim = x.shape[1]
    # for dim in [128]:
    #     x = layers.Dense(dim, activation='relu')(x)
    #     x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(10)(x)
    return models.Model(inputs, outputs)    
    
    
# CNN Transformer network
def cnn_basic(inputs, ff_dim, dropout, i):
    x = layers.MultiHeadAttention(key_dim=512, num_heads=4, dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout, name=f'cin_drop{i}')(x)
    x = layers.LayerNormalization(epsilon=1e-6, name=f'cin_norm{i}')(x)
    res = layers.Add(name=f'cin_add{i}')([x,inputs])
    # FFN
    x = layers.Conv1D(ff_dim*4, kernel_size=3, padding='same', name=f'ffn_cnn{i}')(res)
    x = layers.Dropout(dropout, name=f'cout_drop{i}')(x)
    x = layers.Conv1D(inputs.shape[-1], kernel_size=3, padding='same', name=f'out_cnn{i}')(x)
    x = layers.LayerNormalization(epsilon=1e-6, name=f'cout_norm{i}')(x)
    return layers.Add(name=f'cout_add{i}')([x,res])
    
    
def make_cnn_model(input_shape, dropout, num_classes, dims, num_layers):
    # fencoder = FNetEncoder(dims, dropout=dropout, activation=activation)
    inputs = layers.Input(input_shape)
    x = norm(inputs)
    # pos_enc = SinePositionEncoding()(x)
    # x = layers.Add(name='model_add')([x, pos_enc])
    x = layers.Conv1D(dims, kernel_size=3, padding='same', name='model_cnn')(x)
    for i in range(num_layers):
        x = cnn_basic(x, dims, dropout, i)
    x = layers.Flatten(name='cmodel_flat')(x)
    # flat_dim = x.shape[1]
    # for dim in [128]:
    #     x = layers.Dense(dim, activation='relu')(x)
    #     x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(10)(x)
    return models.Model(inputs, outputs)
    
    
# GRU Transformer network
def gru_basic(inputs, ff_dim, dropout, i):
    x = layers.MultiHeadAttention(key_dim=512, num_heads=4, dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout, name=f'gin_drop{i}')(x)
    x = layers.LayerNormalization(epsilon=1e-6, name=f'gin_norm{i}')(x)
    res = layers.Add(name=f'gin_add{i}')([x,inputs])
    # FFN
    x = layers.GRU(ff_dim*4, return_sequences=True, name=f'ffn_gru{i}')(res)
    x = layers.Dropout(dropout, name=f'gout_drop{i}')(x)
    x = layers.GRU(inputs.shape[-1], return_sequences=True, name=f'out_gru{i}')(x)
    x = layers.LayerNormalization(epsilon=1e-6, name=f'gout_norm{i}')(x)
    return layers.Add(name=f'gout_add{i}')([x,res])
    
    
def make_gru_model(input_shape, dropout, num_classes, dims, num_layers):
    # fencoder = FNetEncoder(dims, dropout=dropout, activation=activation)
    inputs = layers.Input(input_shape)
    x = norm(inputs)
    # pos_enc = SinePositionEncoding()(x)
    # x = layers.Add(name='model_add')([x, pos_enc])
    x = layers.GRU(dims, return_sequences=True, name='model_gru')(x)
    for i in range(num_layers):
        x = gru_basic(x, dims, dropout, i)
    x = layers.Flatten(name='gmodel_flat')(x)
    # flat_dim = x.shape[1]
    # for dim in [128]:
    #     x = layers.Dense(dim, activation='relu')(x)
    #     x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation='linear')(x)
    return models.Model(inputs, outputs)     
    

def add_lag_features(data, column_name):
    """
    Adds lag-based features to count the number of lags since each value last appeared.
    Parameters:
        data (pd.DataFrame): Input DataFrame with one column of integers.
        column_name (str): Name of the column with integers ranging from 0 to 9.
    Returns:
        pd.DataFrame: DataFrame with an additional column for lags since last occurrence.
    """
    # print(f'\nAdding lag-based features to count the number of lags since each value last appeared.')
    # Initialize a dictionary to track the last seen index for each unique value
    unique_values = data[column_name].unique()
    last_seen = {val: -1 for val in unique_values}  # -1 indicates not seen yet
    lag_features = []
    for i, value in enumerate(data[column_name]):
        # Compute lag as the difference between current index and last seen index
        lag_features.append(i - last_seen[value] if last_seen[value] != -1 else 0)
        # Update last seen index for the current value
        last_seen[value] = i
    # Add the lag feature to the DataFrame
    data[f'{column_name}_lags_since_last'] = lag_features
    return data    


def rolling_count_fixed_window(data, column_name):
    """
    Calculates rolling count of value occurrences within a fixed window size.
    Parameters:
        data (pd.DataFrame): Input DataFrame with one column of integers.
        column_name (str): Name of the column containing values.
        window_size (int): Size of the rolling window.
    Returns:
        pd.DataFrame: DataFrame with the rolling count feature added.
    """
    # print(f'\nCalculating rolling count of value occurrences within a fixed window size...')
    rolling_counts = np.zeros(len(data), dtype=np.int8)  # Ensure dtype is int8
    value_positions = {val: [] for val in data[column_name].unique()}  # Track positions
    for i in range(len(data)):
        value = data[column_name][i]
        # Add current index to the tracking list for the current value
        value_positions[value].append(i)
        # Remove indices outside the fixed window
        # while value_positions[value] and value_positions[value][0] < i - window_size + 1:
        #     value_positions[value].pop(0)
        # Count the number of occurrences within the fixed window
        rolling_counts[i] = len(value_positions[value])
    # Add the feature to the DataFrame
    data[f'{column_name}_rolling_count'] = rolling_counts
    return data


def distance_to_next(data, column_name):
    """
    Calculate the distance (in rows) to the next occurrence of the same value.
    """
    data = data.reset_index(drop=True)
    # print(f'\nCalculating the distance (in rows) to the next occurrence of the same value....')
    next_seen = {}
    distance = [0] * len(data)
    for i in range(len(data) - 1, -1, -1):
        value = data[column_name][i]
        distance[i] = next_seen[value] - i if value in next_seen else 0
        next_seen[value] = i
    data[f'{column_name}_distance_to_next'] = distance
    return data


def create_windows(data, wl):
    print('\nGenerating sliding windows')
    windows = np.empty([len(data) - wl, wl, 4], dtype=np.int8)
    for i in range(len(data)-wl):
        new_data = data[i:i+wl]
        new_data = distance_to_next(new_data, '0')
        new_data = rolling_count_fixed_window(new_data, '0')
        new_data = add_lag_features(new_data, '0')
        windows[i] = new_data.values
    return windows    


def analyze_misclassifications(model, X, y_true, class_names=None):
    """
    Analyze misclassified samples for a classification model.

    Parameters:
    - model: Trained classification model.
    - X: Input features (NumPy array or DataFrame).
    - y_true: True labels (array-like).
    - class_names: List of class names for better interpretability (optional).

    Returns:
    - DataFrame summarizing misclassified samples.
    """
    
    # Predict and find misclassified samples
    y_pred = np.argmax(model.predict(X), axis=1)
    misclassified_indices = np.where(y_pred != y_true)[0]
    misclassified_X = X[misclassified_indices]
    misclassified_y_true = y_true[misclassified_indices]
    misclassified_y_pred = y_pred[misclassified_indices]

    # Summarize misclassifications
    misclassified_df = pd.DataFrame({
        'True Label': misclassified_y_true,
        'Predicted Label': misclassified_y_pred,
        'Sample Index': misclassified_indices
    })

    # Add class names if provided
    if class_names:
        misclassified_df['True Label'] = misclassified_df['True Label'].map(lambda x: class_names[x])
        misclassified_df['Predicted Label'] = misclassified_df['Predicted Label'].map(lambda x: class_names[x])
    return misclassified_df


def evaluate_test_set(model, X_test, y_test, class_names=None):
    """
    Evaluate the model's performance on the test set and visualize results.

    Parameters:
    - model: Trained classification model.
    - X_test: Test input features (NumPy array or DataFrame).
    - y_test: True labels for the test set.
    - class_names: List of class names for better interpretability (optional).
    """
    # Predict class probabilities and convert to predicted labels
    y_pred = np.argmax(model.predict(X_test), axis=1)

    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # # Confusion matrix
    # cm = confusion_matrix(y_test, y_pred)
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    # plt.title("Confusion Matrix")
    # plt.xlabel("Predicted")
    # plt.ylabel("True")
    # plt.show()


if __name__ == '__main__':
    # Parameters
    lotto = 'C4L'
    wl = 50
    dropout = 0.3
    dims = 64
    num_layers = 2
    activation = 'relu'
    
    batch_size = 16
    lr = 5e-4
    early_pat = 5
    reduc_pat = 2
    reduc_fac = 0.9

    # Target dataset
    target_df = pd.read_csv(f'datasets/{lotto}_Full.csv')
    target_df = (target_df[['A','B','C','D','E']].dropna().reset_index(drop=True).astype(np.int8))
    # Add preceding 0 for values between 1 and 9
    print('\nAdding preceding 0 for values from 1 to 9 in Target Dataset')
    for col in target_df.columns:
        target_df[col] = target_df[col].apply(lambda x: f'0{x}' if 1 <= x <= 9 else int(x))
    target_df = target_df.values.flatten()    
    # Flatten down to individual digits
    target_df = np.array([int(digit) for number in target_df for digit in str(number)], dtype=np.int8)    
    
    # Base dataset
    quick_df = pd.read_csv('datasets/Quick.csv', dtype=np.int8)
    quick_df = (quick_df.drop(columns=['Unnamed: 0']).dropna().reset_index(drop=True).astype(np.int8))
    # Add preceding 0 for values between 1 and 9
    print('\nAdding preceding 0 for values from 1 to 9 in Base Dataset')
    for col in quick_df.columns:
        quick_df[col] = quick_df[col].apply(lambda x: f'0{x}' if 1 <= x <= 9 else int(x))
    quick_df = quick_df.values.flatten()
    # Flatten down to individual digits
    print('\nFlattening down to individual digits')
    quick_df = np.array([int(digit) for number in quick_df for digit in str(number)], dtype=np.int8)
    quick_df = quick_df[int(len(quick_df)*0.95):int(len(quick_df)*0.953)]

    full_data = quick_df  #np.concatenate((quick_df, target_df))
    len_target = len(target_df)
    len_quick = len(quick_df)
    del quick_df, target_df
    
    # Creating new df from the digit data with targets
    full_data = pd.DataFrame(full_data, columns=['0'])

    # Shift target positions per iteration
    for t in range(1):
        data = full_data.copy()
        print(f'\nNow training for T{t} target...')
        data['targets'] = data['0'].shift(-(t+1))
        data.dropna(inplace=True)
        data['targets'] = data['targets'].astype(np.int8)
        
        # Creating new features
        print('\nCreating new features')
        col = '0' # Column to use for feature calculations
        
        # Splitting X and y values
        target = data['targets'].values
        data = data.drop(columns=['targets'])
        
        print(f'\nTarget Data Length: {len_target}\tQuick Data Length: {len_quick}')
        
        windows = create_windows(data, wl)
        
        enc_in = windows[1:]
        dec_out = target[wl:-1]

        # print(f'\nenc_in:\n{enc_in[-3:]}')
        # print(f'\ndec_out:\n{dec_out[-3:]}\n')
        
        split = len(enc_in)//10
        encoder_input = enc_in[:split*8]  #enc_in[:int(len_quick+(len_target*0.8))]
        decoder_output = dec_out[:split*8]  #dec_out[:int(len_quick+(len_target*0.8))]
        norm = layers.Normalization(name='data_norm')
        norm.adapt(encoder_input)
        
        enc_in_test = enc_in[split*8:split*9]  #enc_in[int(len_quick+(len_target*0.8)):int(len_quick+(len_target*0.95))]
        dec_out_test = dec_out[split*8:split*9]  #dec_out[int(len_quick+(len_target*0.8)):int(len_quick+(len_target*0.95))]
        
        enc_in_val = enc_in[split*9:]  #enc_in[-int(len_target*0.05):]
        dec_out_val = dec_out[split*9:]  #dec_out[-int(len_target*0.05):]
        
        input_shape = (encoder_input.shape[1],encoder_input.shape[2])
        
        model = make_lstm_model(input_shape, dropout, dims, num_layers)
        model_name = f'lstm_mha_t{t}_WL{wl}_digits_lin_v2'
        model_weights = f'model_weights/{model_name}.weights.h5'

        # model.load_weights(model_weights)
        
        model.compile(optimizer=optimizers.AdamW(learning_rate=lr), loss=losses.SparseCategoricalCrossentropy(use_logits=True), metrics=['accuracy'], jit_compile=True)
        model.summary()
        
        callback = [callbacks.EarlyStopping(monitor='val_mae', min_delta=0,
                                            patience=early_pat, verbose=1, mode='auto',
                                            baseline=None, restore_best_weights=True,
                                            start_from_epoch=0),
                   callbacks.ReduceLROnPlateau(monitor='val_loss', factor=reduc_fac,
                                               patience=reduc_pat, verbose=0, mode='auto',
                                               min_delta=0.0, cooldown=0, min_lr=0),
                    callbacks.ModelCheckpoint(f'checkpoint_models/{model_name}.keras',
                                              monitor = 'val_mae', save_best_only=True)
                   ]
        
        # Training the model
        #class_weights = compute_class_weight('balanced', classes=unique_classes, y=decoder_output)
        #class_weights_dict = dict(enumerate(class_weights))
        
        lstm_history = model.fit(encoder_input, decoder_output, batch_size=batch_size, epochs=100, verbose=1,
                           validation_data=(enc_in_test, dec_out_test), callbacks=callback,) # class_weight=class_weights_dict)
        
        model.save(f'models/{model_name}.keras', overwrite=True)
        model.save_weights(model_weights, overwrite=True)
        
        val_loss, val_mae, _ = model.evaluate(enc_in_val, dec_out_val)
        print(f'\nVal Loss: {val_loss:.4f}\tVal MAE: {val_mae:.4f}\n')
        
        # nums = 500
        #predictions = model.predict(enc_in_val)
        #preds = np.argmax(predictions, axis=1)
        #real_res = dec_out_val
        #hits = 0
        #for i, num in enumerate(preds):
        #    if num==real_res[i]:
        #        hits += 1
        #print(f'\nCorrect Predictions = {hits} out of {len(real_res)}')
        #print(f'\nenc_in_val:\n{enc_in_val[-3:]}')
        #print(f'\ndec_out_val:\n{dec_out_val[-3:]}\n')
        
        #evaluate_test_set(model, enc_in_val, dec_out_val, class_names=['Class 0', 'Class 1', 'Class 2','Class 3', 'Class 4', 'Class 5','Class 6', 'Class 7', 'Class 8','Class 9'])
        
        #misclassified_df = analyze_misclassifications(model, enc_in_val, dec_out_val, class_names=['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9'])
        #print(f'\nMisClassified Classes:\n{misclassified_df.head()}')
        del windows, model
        gc.collect()
