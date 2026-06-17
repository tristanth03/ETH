import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len, units):
        super().__init__()
        pos = tf.range(max_len, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(units, dtype=tf.float32)[tf.newaxis, :]
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(units, tf.float32))
        angle_rads = pos * angle_rates
        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        self.pos_encoding = pos_encoding[tf.newaxis, :, :]

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]

class WindowGenerator():
  def __init__(self, input_width, label_width, shift, input_columns=None, label_columns=None, all_columns=None):
    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
    self.train_label_indices = {name: i for i, name in enumerate(train_df.columns)}

    # ...and the input column indices
    self.input_columns = input_columns
    if input_columns is not None:
      self.input_columns_indices = {name: i for i, name in enumerate(input_columns)}
    self.train_input_indices = {name: i for i, name in enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def split_window(self, features):
      inputs = features[:, self.input_slice, :]
      labels = features[:, self.labels_slice, :]
      if self.input_columns is not None:
        inputs = tf.stack([inputs[:, :, self.train_input_indices[name]] for name in self.input_columns], axis=-1)
      if self.label_columns is not None:
        labels = tf.stack([labels[:, :, self.train_label_indices[name]] for name in self.label_columns], axis=-1)
      return inputs, labels

  def make_dataset(self, data, shuffle = False, batchsize = 500,):
      data = np.array(data, dtype=np.float32)
      ds = tf.keras.preprocessing.timeseries_dataset_from_array(data=data, targets=None, sequence_length=self.total_window_size,
                                                                sequence_stride=1, sampling_rate=1, shuffle=shuffle, batch_size=batchsize)
      ds = ds.map(self.split_window)
      return ds

class FeedForward(tf.keras.layers.Layer):
    def __init__(self, units, dff):
        super().__init__()
        self.ffn = tf.keras.Sequential([tf.keras.layers.Dense(dff, activation='relu'), tf.keras.layers.Dense(units),])
        self.add = tf.keras.layers.Add()
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        out = self.ffn(x)
        return self.norm(self.add([x, out]))

class TimeSeriesTransformerMultiStep(tf.keras.Model):
    def __init__(self, input_width, label_width, dff=128, units=64, num_heads=2):
        super().__init__()
        self.input_proj = tf.keras.layers.Dense(units)
        self.query_proj = tf.keras.layers.Dense(units)
        self.pos_enc_input = PositionalEncoding(input_width, units)
        self.pos_enc_query = PositionalEncoding(label_width, units)
        # === Encoder ===
        self.encoder_attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=units, dropout=0.0)
        # self.encoder_dropout1 = tf.keras.layers.Dropout(0.1)
        self.encoder_norm1 = tf.keras.layers.LayerNormalization()
        self.encoder_ffn = FeedForward(units, dff)
        # self.encoder_dropout2 = tf.keras.layers.Dropout(0.1)
        # === Decoder ===
        self.decoder_self_attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=units, dropout=0.0)
        self.decoder_norm1 = tf.keras.layers.LayerNormalization()
        self.cross_attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=units, dropout=0.5)
        self.decoder_norm2 = tf.keras.layers.LayerNormalization()
        self.cross_ffn = FeedForward(units, dff)
        self.decoder_norm3 = tf.keras.layers.LayerNormalization()
        # === Output ===
        self.out = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        past_input, future_query = inputs  # Shapes: (batch_size, input_width, num_inputs), (batch_size, label_width, num_inputs)
        # === Encoder ===
        x = self.input_proj(past_input)
        x = self.pos_enc_input(x)
        x = self.encoder_norm1(x + self.encoder_attn(x, x, x))
        x = self.encoder_ffn(x)
        # === Decoder ===
        y = self.query_proj(future_query)
        y = self.pos_enc_query(y)
        # Masked self-attention (prevent attending to future time steps)
        seq_len = tf.shape(y)[1]
        self_attn_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)        # lower triangle
        self_attn_mask = tf.reshape(self_attn_mask, (1, 1, seq_len, seq_len))           # for broadcasting
        y2 = self.decoder_self_attn(y, y, y, attention_mask=self_attn_mask)
        y = self.decoder_norm1(y + y2)
        y2, attn_scores = self.cross_attn(y, x, x, return_attention_scores=True)
        self.att_scr = attn_scores # Cross-attention: queries = y, keys/values = encoder output x
        # y2 = self.decoder_dropout1(y2)
        y = self.decoder_norm2(y + y2)
        y2 = self.cross_ffn(y)
        y = self.decoder_norm3(y + y2)
        return self.out(y)  # Shape: (batch_size, label_width, 1)

if __name__ == '__main__':
    # PREPARE DATA
    df = pd.read_excel(r'C:\Programming\ETH\SS26\ISQT26\market_transformer\market_data.xlsx',sheet_name='US', engine='openpyxl')
    df = df.set_index(df['Date'])
    df = df.drop(columns='Date')
    df['MKT'] = df['_MKT'].pct_change(52)
    df = df.dropna()

    # hold out test data (df2)
    n = len(df)
    df1 = df[0:int(0.8*n)]
    df2 = df[int(0.8*n):]
    mm_scaler = preprocessing.StandardScaler()
    df1m = mm_scaler.fit_transform(df1)
    df2m = mm_scaler.transform(df2)
    train_df = pd.DataFrame(df1m, index=df1.index, columns=df1.columns)
    test_df = pd.DataFrame(df2m, index=df2.index, columns=df2.columns)

    # define sliding window
    lf = 52      # shift forward
    lw = 52      # label width
    lb = 52

    # look back
    window = WindowGenerator(input_width=lb, label_width=lw, shift=lf, input_columns=['CF', 'CAPE'], label_columns=['MKT'])
    td = window.make_dataset(train_df, batchsize=150, shuffle=True)
    # cross-validation
    is_data = td.take(5)
    os_data = td.skip(5)

    model = TimeSeriesTransformerMultiStep(input_width=lb, label_width=lw, units=64)
    input = np.concatenate([x for x, y in is_data], axis=0)
    target = np.concatenate([y for x, y in is_data], axis=0)
    out = model([input, target])
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.001, decay_steps=50, decay_rate=0.99, staircase=True)
    model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.SGD(learning_rate=lr_schedule), metrics=[tf.metrics.MeanSquaredError(), ])
    model.run_eagerly = True
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50, mode='min')
    history = model.fit(x=[input, target], y=target[:,-1,:], epochs=25, callbacks=[early_stopping])
    model.summary()

    fig, axs = plt.subplots()
    axs.plot(history.history['loss'])
    axs.legend(['training loss', 'validation loss'])

    eval_train = window.make_dataset(train_df, batchsize=train_df.shape[0], shuffle=False)
    eval_test = window.make_dataset(test_df, batchsize=test_df.shape[0], shuffle=False)

    # CHECK IS and OS performance and P/L of a trading strategy
    plt.figure()
    plt.subplot(321)
    query = np.concatenate([x for x, y in eval_train], axis=0)
    value = np.concatenate([y for x, y in eval_train], axis=0)
    y_pred = model(([query, value]))
    y_true = value[:,-1,:]

    mse = tf.reduce_mean(tf.keras.losses.MSE(y_true, y_pred[:,-1,:]))
    plt.plot(train_df.index[lb+lf-1:], y_true)
    plt.plot(train_df.index[lb+lf-1:], y_pred[:,-1,:], '--')
    plt.title('in-sample mse =%1.2f' %mse )
    plt.legend(['y_true', 'y_pred'])

    plt.subplot(323)
    y_mkt = df1.iloc[lb+lf-1:,:].loc[:,'_MKT'].pct_change()
    # position taking: simple switch
    pos = np.sign(np.squeeze(y_pred[:,-1,:]))
    pos[pos == -1] = 0
    pnl = pos[1:] * y_mkt[:-1]
    # pnl2 = pos[2:] * y_mkt[:-2]
    plt.plot(y_mkt.index[:-1], np.cumsum(pnl))
    # plt.plot(y_mkt.index[:-2], np.cumsum(pnl2),'--')
    plt.plot(y_mkt.index[:-1], np.cumsum(y_mkt[:-1]))
    tmp = pnl - y_mkt[:-1]
    sr = tmp.mean()/tmp.std() * np.sqrt(52)
    plt.title('in-sample IR = %1.2f' %sr)
    plt.legend(['pnl [t+1]', 'underlying'])

    plt.subplot(325)
    plt.imshow(model.att_scr[:, 0, -1, :], aspect='auto')

    plt.subplot(322)
    query = np.concatenate([x for x, y in eval_test], axis=0)
    value = np.concatenate([y for x, y in eval_test], axis=0)
    y_pred = model(([query, value]))
    y_true = value[:,-1,:]
    mse = tf.reduce_mean(tf.keras.losses.MSE(y_true, y_pred[:,-1,:]))
    plt.plot(test_df.index[lb+lf-1:], y_true)
    plt.plot(test_df.index[lb+lf-1:], y_pred[:,-1,:], '--')
    plt.title('out-of-sample mse =%1.2f' %mse )
    plt.legend(['y_true', 'y_pred'])

    plt.subplot(324)
    y_mkt = df2.iloc[lb+lf-1:,:].loc[:,'_MKT'].pct_change()
    # position taking: simple switch
    pos = np.sign(np.squeeze(y_pred[:,-1,:]))
    pos[pos == -1] = 0
    pnl = pos[1:] * y_mkt[:-1]
    # pnl2 = pos[2:] * y_mkt[:-2]
    plt.plot(y_mkt.index[:-1], np.cumsum(pnl))
    # plt.plot(y_mkt.index[:-2], np.cumsum(pnl2),'--')
    plt.plot(y_mkt.index[:-1], np.cumsum(y_mkt[:-1]))
    tmp = pnl - y_mkt[:-1]
    sr = tmp.mean()/tmp.std() * np.sqrt(52)
    plt.title('out-of-sample IR = %1.2f' %sr)
    plt.legend(['pnl [t+1]', 'underlying'])

    plt.subplot(326)
    plt.imshow(model.att_scr[:, 0, -1, :], aspect='auto')

    plt.show()

