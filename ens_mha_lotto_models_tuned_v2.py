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
    

def create_windows(data, wl):
    print('\nGenerating sliding windows')
    windows = np.array([data[i:i + wl] for i in range(len(data) - wl)], dtype=np.int8)
    # np.random.shuffle(windows)
    return windows


def distance_to_next(data, column_name):
    """
    Calculate the distance (in rows) to the next occurrence of the same value.
    """
    print(f'\nCalculating the distance (in rows) to the next occurrence of the same value....')
    next_seen = {}
    distance = [0] * len(data)
    for i in range(len(data) - 1, -1, -1):
        value = data[column_name][i]
        distance[i] = next_seen[value] - i if value in next_seen else 0
        next_seen[value] = i
    data[f'{column_name}_distance_to_next'] = distance
    return data


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
    
    batch_size = 2
    lr = 1e-5
    early_pat = 10
    reduc_pat = 3
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
    quick_df = quick_df[int(len(quick_df)*0.96):int(len(quick_df)*0.97)]

    full_data = target_df  # np.concatenate((quick_df, target_df))
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
        data = distance_to_next(data, col)
        
        # Splitting X and y values
        target = data['targets'].values
        data = data.drop(columns=['targets']).values
        
        print(f'\nTarget Data Length: {len_target}\tQuick Data Length: {len_quick}')
        
        # Map classes to sequential range
        unique_classes = np.unique(target)
        num_classes = len(unique_classes)
        
        windows = create_windows(data, wl)
        del data
        
        enc_in = windows[1:]
        dec_out = target[wl:-1]
        
        split = len(enc_in)//10
        encoder_input = enc_in[:split*8]  #enc_in[:int(len_quick+(len_target*0.8))]
        decoder_output = dec_out[:split*8]  #dec_out[:int(len_quick+(len_target*0.8))]
        #norm = layers.Normalization(name='data_norm')
        #norm.adapt(encoder_input)
        
        enc_in_test = enc_in[split*8:split*9]  #enc_in[int(len_quick+(len_target*0.8)):int(len_quick+(len_target*0.95))]
        dec_out_test = dec_out[split*8:split*9]  #dec_out[int(len_quick+(len_target*0.8)):int(len_quick+(len_target*0.95))]
        
        enc_in_val = enc_in[split*9:]  #enc_in[-int(len_target*0.05):]
        dec_out_val = dec_out[split*9:]  #dec_out[-int(len_target*0.05):]
        
        layer_type = 'combo'
        model_name = f'{layer_type}_mha_t{t}_WL{wl}_digits_v1'
        model1 = 'lstm'
        model2 = 'gru'
        base_model1 = models.load_model(f'models/{model1}_mha_t{t}_WL{wl}_digits_v1.keras')
        base_model2 = models.load_model(f'models/{model2}_mha_t{t}_WL{wl}_digits_v1.keras')

        # Extract the output from the second-to-last layer
        intermediate_layer_model1 = models.Model(base_model1.input, base_model1.layers[-2].output)
        intermediate_layer_model2 = models.Model(base_model2.input, base_model2.layers[-2].output)
        
        # Freeze all layers of the intermediate model
        for layer1 in intermediate_layer_model1.layers:
            layer1.trainable = False
        for layer2 in intermediate_layer_model2.layers:
            layer2.trainable = False            
    
        # Add custom Dense and Dropout layers for the new task
        x1 = intermediate_layer_model1.output
        x2 = intermediate_layer_model2.output
        x = layers.Concatenate()([x1,x2])
        # x = layers.Flatten()(x)
        x = layers.Dropout(0.1)(x)
        flat_dim = x.shape[1]
        for i,dim in enumerate([flat_dim//4]):
            x = layers.Dense(dim, activation=activation,
                             kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                            bias_regularizer=regularizers.L2(1e-4),
                             activity_regularizer=regularizers.L2(1e-5),
                            name=f'dense_top_{i}')(x)
            x = layers.Dropout(0.1, name=f'dopout_top_{i}')(x)
        output = layers.Dense(num_classes, activation='softmax')(x)  # Output layer
    
        # Create the new model
        model = models.Model([intermediate_layer_model1.input,intermediate_layer_model2.input], output)
        
        # Parameters for the tuned version of the model
        model_name = f'{layer_type}_mha_{lotto}_WL{wl}_digits_tuned_v1'
        model_weights = f'model_weights/{model_name}.weights.h5'
        model.compile(optimizer=optimizers.AdamW(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'], jit_compile=True)
        model.summary()        
        
        callback = [callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0,
                                            patience=early_pat, verbose=1, mode='auto',
                                            baseline=None, restore_best_weights=True,
                                            start_from_epoch=0),
                   callbacks.ReduceLROnPlateau(monitor='val_loss', factor=reduc_fac,
                                               patience=reduc_pat, verbose=0, mode='auto',
                                               min_delta=0.0, cooldown=0, min_lr=0),
                    callbacks.ModelCheckpoint(f'checkpoint_models/{model_name}.keras',
                                              monitor = 'val_accuracy', save_best_only=True)
                   ]
        
        # Training the model
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=decoder_output)
        class_weights_dict = dict(enumerate(class_weights))
        
        history = model.fit([encoder_input,encoder_input,], decoder_output, batch_size=batch_size, epochs=100, verbose=1,
                           validation_data=([enc_in_test,enc_in_test], dec_out_test), callbacks=callback, class_weight=class_weights_dict)
        
        model.save(f'models/{model_name}.keras', overwrite=True)
        model.save_weights(model_weights, overwrite=True)
        
        val_loss, val_mae = model.evaluate([enc_in_val,enc_in_val], dec_out_val)
        print(f'\nVal Loss: {val_loss:.4f}\tVal ACCURACY: {val_mae:.4f}\n')
        
        # nums = 500
        predictions = model.predict(enc_in_val)
        preds = np.argmax(predictions, axis=1)
        real_res = dec_out_val
        hits = 0
        for i, num in enumerate(preds):
            if num==real_res[i]:
                hits += 1
        print(f'\nCorrect Predictions = {hits} out of {len(real_res)}\n')
        
        evaluate_test_set(model, enc_in_val, dec_out_val, class_names=['Class 0', 'Class 1', 'Class 2','Class 3', 'Class 4', 'Class 5','Class 6', 'Class 7', 'Class 8','Class 9'])
        
        misclassified_df = analyze_misclassifications(model, enc_in_val, dec_out_val, class_names=['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9'])
        print(f'\nMisClassified Classes:\n{misclassified_df}')
        del windows, model
        gc.collect()
