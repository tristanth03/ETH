import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from scipy.stats import zscore

# get sequences
def sstack(ds):
    return tf.reshape(ds, [tf.shape(ds)[0], 3*lb]), tf.reshape(ds, [tf.shape(ds)[0], 3*lb])

class Autoencoder(tf.keras.models.Model):
  def __init__(self, num_inputs, num_hidden):
    super(Autoencoder, self).__init__()
    encoder_input = tf.keras.Input(shape=(num_inputs,), name="input")
    enc = tf.keras.layers.Dense(num_hidden, activation='sigmoid')(encoder_input)
    self.encoder = tf.keras.Model(inputs=encoder_input, outputs=enc)
    decoder_input = tf.keras.Input(num_hidden,)
    rec = tf.keras.layers.Dense(num_inputs, activation='linear')(decoder_input)
    self.decoder = tf.keras.Model(inputs=decoder_input, outputs=rec)

  def call(self, input):
    u = self.encoder(input)
    # local contractions
    w = self.encoder.layers[1].weights[0]
    b = self.encoder.layers[1].weights[1]
    tmp = tf.matmul(input, w) + tf.expand_dims(b,0)
    drv = tf.math.sigmoid(tmp) * (1 - tf.math.sigmoid(tmp))
    contractive_loss = tf.reduce_sum(tf.norm(tf.expand_dims(drv, 1) * tf.expand_dims(w, 0), ord='fro', axis=[1,2]))
    self.add_loss(0.0015 * contractive_loss)
    self.add_metric(contractive_loss, "contractive")
    decoded = self.decoder(u)
    return decoded

if __name__ == '__main__':
    lb = 5
    df = pd.read_excel('data.xlsx', sheet_name='US MKT', engine='openpyxl', index_col=0)
    df = df.apply(zscore)
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(df.values, None, lb, batch_size=100, shuffle=False)
    dss = ds.map(sstack)

    # Set up NN
    model = Autoencoder(num_inputs=lb*3, num_hidden=2)
    model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.SGD(learning_rate=0.1), metrics=[tf.metrics.MeanSquaredError(), ])
    model.run_eagerly = True
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50, mode='min')
    history = model.fit(dss, epochs=200, callbacks=[early_stopping])
    model.summary()

    fig, ax = plt.subplots(1,3)
    ax[0].plot(history.history['loss'])
    ax[1].plot(history.history['mean_squared_error'])
    ax[2].plot(history.history['contractive'])
    ax[2].set_title('Contractive Loss')

    fig, axs = plt.subplots(3)
    y_true = np.concatenate([x for x, y in dss], axis=0)
    y_true = tf.reshape(y_true, [tf.shape(y_true)[0], lb, 3])
    y_true = pd.DataFrame(y_true[:,-1,:], index=df.index[lb-1:], columns=df.columns)
    axs[0].plot(y_true)
    axs[0].set_title('True')
    axs[0].legend(df.columns) 

    y_pred = model.predict(dss)
    y_pred = tf.reshape(y_pred, [tf.shape(y_pred)[0], lb, 3])
    y_pred = pd.DataFrame(y_pred[:,-1,:], index=df.index[lb-1:], columns=df.columns)
    axs[1].set_title('Predicted')
    axs[1].plot(y_pred)

    model_0 = model.encoder
    y00 = model_0.predict(dss)
    y00 = pd.DataFrame(y00, index=df.index[lb-1:], columns=['x1', 'x2'])
    axs[2].set_title('Latent Representation')
    axs[2].plot(y00)

    plt.show()


