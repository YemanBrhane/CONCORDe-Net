import numpy as np
import tensorflow as tf
import os
import pandas as pd
import matplotlib.pyplot as plt
from GetDiceCellCountModel import GetDiceCellCountModel
import json


def plot_training_history(save_dir, log_file):

    train_hist_df = pd.read_csv(log_file)
    x = train_hist_df.columns.values[0]
    y = train_hist_df.columns.values[1:]
    plt.text(x=train_hist_df['epoch'].median(), y=train_hist_df['val_loss'].median(),
             s='min val loss = {:.3f}'.format(train_hist_df['val_loss'].min()))
    train_hist_df.plot(x=x, y=y)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'trainingHistory.pdf'))
    

def run(valid_image_dir,
        valid_label_dir,
        train_image_dir,
        train_label_dir,
        output_dir,
        learning_rate,
        batch_size,
        backend,
        input_shape, depth,
        cell_counter_model_dir,
        loss_type,
        optimizer_type,
        base=16,
        epochs=1000,
        pretrained=True,
        all_params=None):
    
    model_hash = [loss_type, optimizer_type, str(batch_size),
                  str(base), str(learning_rate), backend, str(depth)
                  ]
    model_hash = '_'.join(model_hash)
    output_dir = os.path.join(output_dir, model_hash)
    if os.path.exists(output_dir):
        print('output path {}, already exists'.format(output_dir))
        return 0
    else:
        print('output path {}'.format(output_dir))
        os.makedirs(output_dir, exist_ok=True)
    
    # save model configuration file
    if all_params is not None:
        with open(os.path.join(output_dir, 'config.json'), 'w') as j_file:
            json.dump(all_params, j_file)
    
    # load training data
    train_images = np.load(train_image_dir)
    train_label_data = np.load(train_label_dir)
    train_label = train_label_data['image']
    
    # uint8 to binary image
    train_label = (train_label > 200) * 1
    train_label = train_label.astype('float32')
    print('train_label:{}'.format(train_label.shape))
    
    # training cell count
    train_cell_count = train_label_data['numobj']
    train_cell_count = np.expand_dims(train_cell_count, axis=-1)
    print('train_cell_count:{}'.format(train_cell_count.shape))
    
    # load validation data
    valid_images = np.load(valid_image_dir)
    valid_label_data = np.load(valid_label_dir)
    valid_label = valid_label_data['image']

    # uint8 to binary image
    valid_label = (valid_label > 200)*1
    valid_label = valid_label.astype('float32')
    print('valid_label:{}'.format(valid_label.shape))
    
    # validation data cell count
    valid_cell_count = valid_label_data['numobj']
    valid_cell_count = np.expand_dims(valid_cell_count, axis=-1)
    print('valid_cell_count:{}'.format(valid_cell_count.shape))
    
    # image intensity normalization
    
    print('regular normalization applied')
    train_images = train_images*1.0/255
    valid_images = valid_images * 1.0 / 255
    
        
    valid_images = valid_images.astype('float32')
    train_images = train_images.astype('float32')
    valid_cell_count = valid_cell_count.astype('float32')
    train_cell_count = train_cell_count.astype('float32')

    params = {'optimizer_type': optimizer_type,
              'learning_rate': learning_rate,
              'backend': backend,
              'loss_type': loss_type,
              'base': base,
              'epochs': epochs,
              'depth': depth,
              'input_shape': input_shape,
              'cell_counter_model_dir': cell_counter_model_dir,
              'pretrained': pretrained
              }
    
    # cnn model
    obj = GetDiceCellCountModel(**params)
    model = obj.get_model()
    model.summary()

    # training log and callback configurations
    log_filename = os.path.join(output_dir, 'train_hist.csv')  # Callback that streams epoch results to a csv file.
    csv_log = tf.keras.callbacks.CSVLogger(log_filename, separator=',', append=True)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=0, mode='min')
    checkpoint_filepath = os.path.join(output_dir, 'best_model.h5')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                           mode='min')
    callbacks_list = [csv_log, early_stopping, checkpoint]
    
    # fit or train model
    if loss_type == 'dice':
        model.fit(x=train_images, y=train_label,
                  batch_size=batch_size, epochs=epochs, verbose=1, callbacks=callbacks_list,
                  validation_data=(valid_images, valid_label), shuffle=True)
    else:
        model.fit(x=train_images, y=[train_label, train_cell_count],
                  batch_size=batch_size, epochs=epochs, verbose=1, callbacks=callbacks_list,
                  validation_data=(valid_images, [valid_label, valid_cell_count]), shuffle=True)
    
    # plot training history
    plot_training_history(save_dir=output_dir, log_file=log_filename)
