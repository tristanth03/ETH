import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def mse(y_pred, y_true):
    return (y_true - y_pred) ** 2

# ---------------------------------------------------------------------------
# Sliding-window dataset builder
# ---------------------------------------------------------------------------
class WindowGenerator():
    """Converts a time-series DataFrame into overlapping windows (x, y)
    where x is the input window and y is the label window (here identical,
    since this is an autoencoder)."""

    def __init__(self, input_width, label_width, shift,
                 input_columns=None, label_columns=None, all_columns=None):
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.train_label_indices = {name: i for i, name in enumerate(all_columns)}

        self.input_columns = input_columns
        if input_columns is not None:
            self.input_columns_indices = {name: i for i, name in enumerate(input_columns)}
        self.train_input_indices = {name: i for i, name in enumerate(all_columns)}

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
            inputs = tf.stack(
                [inputs[:, :, self.train_input_indices[name]] for name in self.input_columns], axis=-1)
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.train_label_indices[name]] for name in self.label_columns], axis=-1)
        return inputs, labels

    def make_dataset(self, data, shuffle=False, batchsize=500):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data, targets=None, sequence_length=self.total_window_size,
            sequence_stride=1, sampling_rate=1, shuffle=shuffle, batch_size=batchsize)
        ds = ds.map(self.split_window)
        return ds

# ---------------------------------------------------------------------------
# VQ Autoencoder
# ---------------------------------------------------------------------------
class Autoencoder(tf.keras.models.Model):
    """
    Vector-Quantized Autoencoder (VQ-AE) for time-series regime detection.
    Architecture (following the lecture notes):
      Encoder: causal Conv1D -> MaxPool -> Flatten -> Dense(K)
               produces K logits ell(x) for the categorical posterior q(c|x).
      Codebook: K learned embedding vectors {e_1,...,e_K} in R^d.
      Decoder:  Dense -> Reshape -> Conv1DTranspose stack.

    Training objective (MDL / rate-distortion):
      L = distortion + beta * rate
      distortion = MSE(x, x_hat)                         [reconstruction loss]
      rate       = KL(q(c|x) || Uniform(c))              [excess coding cost]
               = sum_k q(c=k|x) * (log q(c=k|x) - log(1/K))

    Rate-distortion frontier (elbow):
      As training progresses, distortion (MSE) decreases and rate (KL) increases.
      The "elbow" or "knee" of the rate-distortion curve is the sweet spot:
      good compression with acceptable reconstruction error.
      Increasing rate at lower distortion is EXPECTED and DESIRED behavior.
      The trade-off is controlled by beta: higher beta = more compression (higher rate).
    """

    def __init__(self, num_timesteps, num_inputs, num_hidden,
                 num_codes=8, gumbel_temp=0.5, mdl_beta=0.005):
        super(Autoencoder, self).__init__()
        self.pool = 4
        self.strides = 4
        self.filters = 32
        self.num_hidden = num_hidden
        self.num_codes = num_codes      # K: codebook size
        self.gumbel_temp = gumbel_temp  # tau: Gumbel-Softmax temperature
        self.mdl_beta = mdl_beta        # beta: rate penalty weight

        # --- Encoder: x -> logits ell(x) of shape (batch, K) ---
        enc_inp = tf.keras.Input(shape=(num_timesteps, num_inputs), name="input")
        x = tf.keras.layers.Conv1D(
            filters=self.filters, kernel_size=3, activation=None,
            use_bias=True, padding='causal')(enc_inp)
        x = tf.keras.layers.MaxPooling1D(
            pool_size=self.pool, strides=self.strides, padding='same')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(self.num_codes, activation=None)(x)  # logits
        self.encoder = tf.keras.Model(inputs=enc_inp, outputs=x)

        # --- Codebook: K embedding vectors in R^num_hidden ---
        self.codebook = tf.keras.layers.Embedding(
            input_dim=self.num_codes, output_dim=self.num_hidden)

        # --- Decoder: z_q (codeword) -> reconstructed segment ---
        dec_inp = tf.keras.Input(shape=(num_hidden,))
        y = tf.keras.layers.Dense(
            units=int(num_timesteps / self.strides) * self.filters,
            activation='relu')(dec_inp)
        y = tf.keras.layers.Reshape(
            target_shape=(int(num_timesteps / self.strides), self.filters))(y)
        y = tf.keras.layers.Conv1DTranspose(
            filters=32, kernel_size=3, strides=self.strides,
            activation='relu', use_bias=True, padding='same')(y)
        y = tf.keras.layers.Conv1DTranspose(
            filters=24, kernel_size=3, strides=1,
            activation='relu', use_bias=True, padding='same')(y)
        y = tf.keras.layers.Conv1DTranspose(
            filters=num_inputs, kernel_size=3, strides=1,
            activation=None, use_bias=True, padding='same')(y)
        self.decoder = tf.keras.Model(inputs=dec_inp, outputs=y)

    def encode(self, x):
        """Map input window x to logits ell(x)."""
        return self.encoder(x)

    def reparameterize(self, logits, training=True):
        """
        Gumbel-Softmax reparameterization (straight-through estimator).
        Training (training=True):
          1. Draw Gumbel noise: g_k = -log(-log(u_k)), u_k ~ U(0,1).
          2. Soft sample: y_soft = softmax((logits + g) / tau).
             As tau->0, y_soft -> one-hot draw (exact categorical sample).
          3. Straight-through: forward pass uses hard one-hot,
             backward pass uses soft gradients.
             c_hat = stopgrad(y_hard - y_soft) + y_soft

        Inference (training=False):
          Use hard argmax (no Gumbel noise); compute soft posterior
          q(c|x) = softmax(logits) separately for the rate term.

        Returns:
          z_q    : selected codeword sent to decoder, shape (batch, num_hidden)
          q_soft : soft posterior probabilities q(c|x), shape (batch, K)
        """
        eps = 1e-8
        if training:
            # Sample Gumbel noise via inverse-transform: g = -log(-log(u))
            u = tf.random.uniform(tf.shape(logits), minval=0.0, maxval=1.0)
            gumbel = -tf.math.log(-tf.math.log(u + eps) + eps)
            # Temperature-scaled softmax of perturbed logits
            y_soft = tf.nn.softmax((logits + gumbel) / self.gumbel_temp, axis=-1)
            # Hard one-hot in the forward pass
            y_hard = tf.one_hot(tf.argmax(y_soft, axis=-1), depth=self.num_codes)
            # Straight-through: gradients flow through y_soft, output is y_hard
            y = tf.stop_gradient(y_hard - y_soft) + y_soft
        else:
            # At inference: argmax of clean posterior (no Gumbel noise)
            y_soft = tf.nn.softmax(logits, axis=-1)
            y = tf.one_hot(tf.argmax(y_soft, axis=-1), depth=self.num_codes)

        # Retrieve all K codewords from the codebook
        codebook_vectors = self.codebook(tf.range(self.num_codes))
        # VQ bottleneck: z_q = sum_k c_hat_k * e_k  (weighted sum of codewords)
        z_q = tf.matmul(y, codebook_vectors)
        # Return z_q for decoding and y_soft for rate computation
        return z_q, y_soft

    def get_regime_probs(self, x):
        """Posterior probabilities q(c|x) = softmax(ell(x))."""
        logits = self.encode(x)
        return tf.nn.softmax(logits, axis=-1)

    def get_regime_ids(self, x):
        """Hard regime assignment: argmax_k q(c=k|x)."""
        return tf.argmax(self.get_regime_probs(x), axis=-1)

    def compute_loss(self, inp):
        """
        MDL objective: L = distortion + beta * rate
        distortion = E[||x - decoder(z_q)||^2]   (MSE reconstruction loss)
        rate       = E[KL(q(c|x) || Uniform(c))]  (excess coding cost)
                   = E[sum_k q_k * (log q_k - log(1/K))]
        """
        logits = self.encode(inp)
        z_q, qy = self.reparameterize(logits, training=True)
        x_hat = self.decoder(z_q)

        # Distortion: average MSE across batch, timesteps, and features
        distortion = tf.reduce_mean(mse(x_hat, inp))

        # Rate: KL(q(c|x) || Uniform), i.e. expected excess codelength
        # relative to a uniform code prior over K codewords
        eps = 1e-8
        log_uniform = tf.math.log(1.0 / float(self.num_codes))
        rate_nats = tf.reduce_mean(
            tf.reduce_sum(qy * (tf.math.log(qy + eps) - log_uniform), axis=1))

        return distortion + self.mdl_beta * rate_nats, distortion, rate_nats

    def call(self, x):
        logits = self.encode(x)
        z_q, _ = self.reparameterize(logits, training=False)
        return self.decoder(z_q)

if __name__ == '__main__':
    # -----------------------------------------------------------------------
    # 1. LOAD AND PREPARE DATA
    # -----------------------------------------------------------------------
    df = pd.read_excel('market_data.xlsx', sheet_name='US', engine='openpyxl')
    df = df.set_index(df['Date']).drop(columns='Date')
    df0 = df.copy()                        # keep raw prices for PnL later
    cols = ['_TY', 'ED', '_MKT']
    df = df[cols].pct_change().dropna()    # weekly returns

    # -----------------------------------------------------------------------
    # 2. TRAIN / TEST SPLIT (80 / 20, no shuffle -- respect time order)
    # -----------------------------------------------------------------------
    n = len(df)
    train_df_raw = df[:int(0.8 * n)]
    test_df_raw  = df[int(0.8 * n):]

    # Standardize using training statistics only (no leakage into test set)
    mm_scaler = preprocessing.StandardScaler()
    train_df = pd.DataFrame(
        mm_scaler.fit_transform(train_df_raw),
        index=train_df_raw.index, columns=cols)
    test_df = pd.DataFrame(
        mm_scaler.transform(test_df_raw),
        index=test_df_raw.index, columns=cols)

    # -----------------------------------------------------------------------
    # 3. SLIDING-WINDOW DATASETS
    #    Each sample is a window of lb consecutive returns (a "segment").
    #    The autoencoder maps each segment to a discrete code c in {1,...,K}.
    # -----------------------------------------------------------------------
    lb = 4   # lookback / window length L
    window = WindowGenerator(
        input_width=lb, label_width=lb, shift=0,
        input_columns=cols, label_columns=cols, all_columns=df.columns)

    # Training dataset: shuffled mini-batches
    train_data = window.make_dataset(train_df, shuffle=True, batchsize=256)
    # Test dataset: single batch in temporal order (for evaluation)
    test_data  = window.make_dataset(test_df, shuffle=False, batchsize=test_df.shape[0])

    # -----------------------------------------------------------------------
    # 4. MODEL
    #    num_codes  = K  (codebook size, i.e. number of regimes)
    #    num_hidden = d  (codeword dimension)
    #    gumbel_temp = tau  (Gumbel-Softmax temperature)
    #    mdl_beta   = beta  (rate penalty weight)
    # -----------------------------------------------------------------------
    model = Autoencoder(num_timesteps=lb, num_inputs=3, num_hidden=4, num_codes=12, gumbel_temp=0.6, mdl_beta=0.1)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Build model by running one forward pass before summary
    for x_sample, _ in train_data.take(1):
        model(x_sample)
        break
    model.summary()

    # -----------------------------------------------------------------------
    # 5. TRAINING LOOP
    #    Minimise L = distortion + beta * rate via gradient descent.
    #    We track both components separately to monitor the rate-distortion
    #    trade-off. Note: rate increases as distortion decreases (expected!).
    #    This traces out the rate-distortion frontier.
    # -----------------------------------------------------------------------
    epochs = 150
    loss_history, dist_history, rate_history = [], [], []

    print("Training VQ-AE (MDL objective)...")
    for epoch in range(epochs):
        batch_losses, batch_dists, batch_rates = [], [], []
        for x_batch, _ in train_data:
            with tf.GradientTape() as tape:
                total_loss, distortion, rate = model.compute_loss(x_batch)
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            batch_losses.append(float(total_loss))
            batch_dists.append(float(distortion))
            batch_rates.append(float(rate))

        avg_loss = np.mean(batch_losses)
        avg_dist = np.mean(batch_dists)
        avg_rate = np.mean(batch_rates)
        loss_history.append(avg_loss)
        dist_history.append(avg_dist)
        rate_history.append(avg_rate)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: loss={avg_loss:.5f}  "
                  f"dist={avg_dist:.5f}  rate={avg_rate:.5f}")

    # -----------------------------------------------------------------------
    # 6. PLOT TRAINING CURVES
    #    Show distortion vs rate (rate-distortion frontier).
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].plot(dist_history)
    axes[0].set_title('Distortion (train MSE)')
    axes[0].set_xlabel('epoch')
    axes[1].plot(rate_history)
    axes[1].set_title('Rate (KL from uniform)')
    axes[1].set_xlabel('epoch')
    plt.tight_layout()

    # -----------------------------------------------------------------------
    # 7. EVALUATION: RECONSTRUCTION AND STRATEGY PERFORMANCE
    # -----------------------------------------------------------------------
    eval_train = window.make_dataset(train_df, batchsize=train_df.shape[0], shuffle=False)
    eval_test  = window.make_dataset(test_df,  batchsize=test_df.shape[0],  shuffle=False)

    # Collect all batches into arrays
    def collect(dataset):
        xs, ys = [], []
        for x, y in dataset:
            xs.append(x.numpy()); ys.append(y.numpy())
        return np.concatenate(xs), np.concatenate(ys)

    x_train, y_train_true = collect(eval_train)
    x_test,  y_test_true  = collect(eval_test)
    y_train_pred = model.predict(x_train, verbose=0)
    y_test_pred  = model.predict(x_test,  verbose=0)

    # Regime assignments on the test set
    regime_probs = model.get_regime_probs(x_test).numpy()
    regime_ids   = np.argmax(regime_probs, axis=1)  # hard code assignment

    plt.figure(figsize=(12, 8))

    # --- Subplot 1: in-sample reconstruction (last timestep, _MKT feature) ---
    plt.subplot(221)
    mse_is = float(tf.reduce_mean(tf.keras.losses.MSE(y_train_true, y_train_pred)))
    plt.plot(train_df.index[lb-1:], np.cumsum(y_train_true[:, -1, -1]))
    plt.plot(train_df.index[lb-1:], np.cumsum(y_train_pred[:, -1, -1]), '--')
    plt.title(f'in-sample MSE = {mse_is:.2f}')
    plt.legend(['y_true', 'y_pred'])

    # --- Subplot 2: in-sample trading PnL ---
    # Strategy: go long MKT when model sees rising TY and ED (Example)
    plt.subplot(222)
    y_mkt_is = df.loc[train_df.index[lb-1:], '_MKT']
    pos_is = ((y_train_pred[:, -1, 0] > 0) & (y_train_pred[:, -1, 1] > 0)).astype(int)
    pnl_is  = pos_is[1:]  * y_mkt_is.values[:-1]
    pnl_is2 = pos_is[2:]  * y_mkt_is.values[:-2]
    plt.plot(y_mkt_is.index[:-1], np.cumsum(pnl_is))
    plt.plot(y_mkt_is.index[:-2], np.cumsum(pnl_is2), '--')
    plt.plot(y_mkt_is.index[:-1], np.cumsum(y_mkt_is.values[:-1]))
    ir_is = (pnl_is - y_mkt_is.values[:-1]).mean() / (pnl_is - y_mkt_is.values[:-1]).std() * np.sqrt(52)
    plt.title(f'in-sample IR = {ir_is:.2f}')
    plt.legend(['pnl [t+1]', 'pnl [t+2]', 'underlying'])

    # --- Subplot 3: out-of-sample reconstruction colored by regime ---
    plt.subplot(223)
    mse_os = float(tf.reduce_mean(tf.keras.losses.MSE(y_test_true, y_test_pred)))
    plt.plot(test_df.index[lb-1:], np.cumsum(y_test_true[:, -1, -1]))
    plt.plot(test_df.index[lb-1:], np.cumsum(y_test_pred[:, -1, -1]), '--')
    plt.scatter(test_df.index[lb-1:], np.cumsum(y_test_pred[:, -1, -1]),
                c=regime_ids, s=12, cmap='tab10')
    plt.title(f'out-of-sample MSE = {mse_os:.2f}')
    plt.legend(['y_true', 'y_pred', 'regime'])

    # --- Subplot 4: out-of-sample trading PnL ---
    plt.subplot(224)
    y_mkt_os = df.loc[test_df.index[lb-1:], '_MKT']
    pos_os = ((y_test_pred[:, -1, 0] > 0) & (y_test_pred[:, -1, 1] > 0)).astype(int)
    pnl_os  = pos_os[1:]  * y_mkt_os.values[:-1]
    pnl_os2 = pos_os[2:]  * y_mkt_os.values[:-2]
    plt.plot(y_mkt_os.index[:-1], np.cumsum(pnl_os))
    plt.plot(y_mkt_os.index[:-2], np.cumsum(pnl_os2), '--')
    plt.plot(y_mkt_os.index[:-1], np.cumsum(y_mkt_os.values[:-1]))
    ir_os = (pnl_os - y_mkt_os.values[:-1]).mean() / (pnl_os - y_mkt_os.values[:-1]).std() * np.sqrt(52)
    plt.title(f'out-of-sample IR = {ir_os:.2f}')
    plt.legend(['pnl [t+1]', 'pnl [t+2]', 'underlying'])
    plt.tight_layout()

    # -----------------------------------------------------------------------
    # 8. REGIME CHART
    #    Visualize inferred regime sequence and market path colored by regime.
    #    Near-uniform code usage (balanced occupancy) indicates efficient use
    #    of the codebook -- the discrete analogue of Gaussian prior matching.
    # -----------------------------------------------------------------------
    t_reg = test_df.index[lb-1:]
    fig2, axs2 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axs2[0].step(t_reg, regime_ids, where='post')
    axs2[0].set_title('Learned market regimes (code index)')
    axs2[0].set_ylabel('regime id')

    mkt_cum = np.cumsum(y_test_true[:, -1, -1])
    axs2[1].scatter(t_reg, mkt_cum, c=regime_ids, s=16, cmap='tab10')
    axs2[1].plot(t_reg, mkt_cum, color='gray', alpha=0.4)
    axs2[1].set_title('Market path colored by inferred regime')
    axs2[1].set_ylabel('cumulative return')
    axs2[1].set_xlabel('date')

    # -----------------------------------------------------------------------
    # 9. MDL DIAGNOSTICS
    #    Code entropy close to log2(K) bits -> near-uniform usage -> efficient
    #    Code entropy << log2(K)           -> codebook collapse
    # -----------------------------------------------------------------------
    counts = np.bincount(regime_ids, minlength=model.num_codes)
    probs  = counts / counts.sum()
    entropy_bits = -np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))
    print(f'\nActive codes : {int((counts > 0).sum())} / {model.num_codes}')
    print(f'Code entropy : {entropy_bits:.3f} bits  '
          f'(max = {np.log2(model.num_codes):.3f} bits for uniform usage)')
    print(f'Code counts  : {dict(zip(range(model.num_codes), counts.tolist()))}')

    plt.show()
