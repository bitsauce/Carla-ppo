import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os


def kl_divergence(mean, logstd_sq, name="kl_divergence"):
    with tf.variable_scope(name):
        return -0.5 * tf.reduce_sum(1.0 + logstd_sq - tf.square(mean) - tf.exp(logstd_sq), axis=1)

def bce_loss(labels, logits, targets):
    return tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels,
        logits=logits
    )

def bce_loss_v2(labels, logits, targets, epsilon=1e-10):
    with tf.variable_scope("bce"):
        return -(labels * tf.log(epsilon + targets) + (1 - labels) * tf.log(epsilon + 1 - targets))

def mse_loss(labels, logits, targets):
    return (labels - targets)**2

def verify_range(tensor, vmin, vmax):
    verify_op = tf.Assert(tf.reduce_all(tf.logical_and(tensor >= vmin, tensor <= vmax)),
                                ["min=", tf.reduce_min(tensor), "max=", tf.reduce_max(tensor)],
                                name="verify_range")
    with tf.control_dependencies([verify_op]):
        tensor = tf.multiply(tensor, 1, name="verify_tensor_identity")
    return tensor


class VAE():
    """
        Base variational autoencoder class.
    """

    def __init__(self, source_shape, target_shape, build_encoder_fn, build_decoder_fn,
                 z_dim=512, beta=1.0, learning_rate=1e-4, lr_decay=0.98, kl_tolerance=0.0,
                 model_dir=".", loss_fn=bce_loss, training=True, reuse=tf.AUTO_REUSE,
                 **kwargs):
        """
            Builds a VAE that passes source states through the encoding graph
            constructed by build_encoder_fn, then, after applying a normal distribution
            to the encoded state, it passes the latent vector through the
            decoding graph constructed by build_decoder_fn.

            source_shape (shape):
                Shape of source (input) tensors.
                E.g. if the VAE takes images of size 160x80x3,
                then source_shape=(160, 80, 3)
            target_shape (shape):
                Shape of target tensors.
            build_encoder_fn (function):
                A function taking an input tensors of shape [batch_size, source_shape],
                outputting a flattened tensor of [batch_size, ?].
            build_decoder_fn (function):
                A function taking an input tensor of shape [batch_size, z_dim],
                outputting a flattened tensor of size [batch_size, product(target_shape)].
            z_dim (int):
                Dimentionality of latent space at bottleneck.
            beta (float):
                KL-divergence loss strength.
            learning_rate (float):
                Learning rate to use with Adam when training.
            kl_tolerance (float):
                A value denoting how tolerant we are to KL-divergence before we
                apply KL-divergence loss.
            model_dir (sting):
                Directory of logs and checkpoints for this model. Used for saving and loading.
            loss_fn (function):
                A loss function taking labels, logits, and targets.
            training (bool):
                When True, defines the additional graph elements needed for training.
            reuse (bool):
                When True, reuse previously constructed graph
        """

        # Create vae
        self.source_shape = source_shape
        self.target_shape = target_shape
        self.z_dim = z_dim
        self.beta = beta
        self.kl_tolerance = kl_tolerance
        with tf.variable_scope("vae", reuse=reuse):
            # Get and verify source and target
            self.source_states = tf.placeholder(shape=(None, *self.source_shape), dtype=tf.float32, name="source_state_placeholder")
            self.target_states = tf.placeholder(shape=(None, *self.target_shape), dtype=tf.float32, name="target_state_placeholder")
            source_states = verify_range(self.source_states, vmin=0, vmax=1)
            target_states = verify_range(self.target_states, vmin=0, vmax=1)

            # Encode image
            with tf.variable_scope("encoder", reuse=False):
                encoded = build_encoder_fn(source_states)

            # Get encoded mean and std
            self.mean      = tf.layers.dense(encoded, z_dim, activation=None, name="mean")
            self.logstd_sq = tf.layers.dense(encoded, z_dim, activation=None, name="logstd_sqare")

            # Sample normal distribution
            self.normal = tfp.distributions.Normal(self.mean, tf.exp(0.5 * self.logstd_sq), validate_args=True)
            if training:
                self.sample = tf.squeeze(self.normal.sample(1), axis=0)
            else:
                self.sample = self.mean

            # Decode random sample
            with tf.variable_scope("decoder", reuse=False):
                decoded = build_decoder_fn(self.sample)

            # Reconstruct image
            self.reconstructed_logits = tf.layers.flatten(decoded, name="reconstructed_logits")
            self.reconstructed_states = tf.nn.sigmoid(self.reconstructed_logits, name="reconstructed_states")

            # Epoch variable
            self.step_idx = tf.Variable(0, name="step_idx", trainable=False)
            self.inc_step_idx = tf.assign(self.step_idx, self.step_idx + 1)

            # Create optimizer
            if training:
                # Reconstruction loss
                self.flattened_target = tf.layers.flatten(target_states, name="flattened_target")
                self.reconstruction_loss = tf.reduce_mean(
                    tf.reduce_sum(
                        loss_fn(labels=self.flattened_target, logits=self.reconstructed_logits, targets=self.reconstructed_states),
                        axis=1
                    )
                )

                # KL divergence loss
                self.kl_loss = kl_divergence(self.mean, self.logstd_sq, name="kl_divergence")
                if self.kl_tolerance > 0:
                    self.kl_loss = tf.maximum(self.kl_loss, self.kl_tolerance * self.z_dim)
                self.kl_loss = tf.reduce_mean(self.kl_loss)

                # Total loss
                self.loss = self.reconstruction_loss + self.beta * self.kl_loss

                # Create optimizer
                self.learning_rate = tf.train.exponential_decay(learning_rate, self.step_idx, 1, lr_decay, staircase=True)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                self.train_step = self.optimizer.minimize(self.loss)

                # Summary
                self.mean_kl_loss, self.update_mean_kl_loss = tf.metrics.mean(self.kl_loss)
                self.mean_reconstruction_loss, self.update_mean_reconstruction_loss = tf.metrics.mean(self.reconstruction_loss)
                self.merge_summary = tf.summary.merge([
                    tf.summary.scalar("kl_loss", self.mean_kl_loss),
                    tf.summary.scalar("reconstruction_loss", self.mean_reconstruction_loss),
                    tf.summary.scalar("learning_rate", self.learning_rate)
                ])

            # Setup model saver and dirs
            self.saver = tf.train.Saver()
            self.model_dir = model_dir
            self.checkpoint_dir = "{}/checkpoints/".format(self.model_dir)
            self.log_dir        = "{}/logs/".format(self.model_dir)
            self.dirs = [self.checkpoint_dir, self.log_dir]
            for d in self.dirs: os.makedirs(d, exist_ok=True)

    def init_session(self, sess=None, init_logging=True):
        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess

        if init_logging:
            self.train_writer = tf.summary.FileWriter(os.path.join(self.log_dir, "train"), self.sess.graph)
            self.val_writer = tf.summary.FileWriter(os.path.join(self.log_dir, "val"), self.sess.graph)

    def save(self):
        model_checkpoint = os.path.join(self.checkpoint_dir, "model.ckpt")
        self.saver.save(self.sess, model_checkpoint, global_step=self.step_idx)
        print("Model checkpoint saved to {}".format(model_checkpoint))

    def load_latest_checkpoint(self):
        model_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        if model_checkpoint:
            try:
                self.saver.restore(self.sess, model_checkpoint)
                print("Model checkpoint restored from {}".format(model_checkpoint))
                return True
            except Exception as e:
                print(e)
                return False

    def generate_from_latent(self, z):
        return self.sess.run(self.reconstructed_states, feed_dict={
                self.sample: z
            })

    def reconstruct(self, source_states):
        reconstructed_states = self.sess.run(self.reconstructed_states, feed_dict={
                self.source_states: source_states
            })
        return [s.reshape(self.source_shape) for s in reconstructed_states]

    def encode(self, source_states):
        return self.sess.run(self.mean, feed_dict={
                self.source_states: source_states
            })

    def get_step_idx(self):
        return tf.train.global_step(self.sess, self.step_idx)

    def train_one_epoch(self, train_source, train_target, batch_size):
        indices = np.arange(len(train_source))
        np.random.shuffle(indices)
        self.sess.run(tf.local_variables_initializer())
        for i in range(train_source.shape[0] // batch_size):
            mb_idx = indices[i*batch_size:(i+1)*batch_size]
            self.sess.run([self.train_step, self.update_mean_kl_loss, self.update_mean_reconstruction_loss], feed_dict={
                self.source_states: train_source[mb_idx],
                self.target_states: train_target[mb_idx]
            })
        self.train_writer.add_summary(self.sess.run(self.merge_summary), self.get_step_idx())
        self.sess.run(self.inc_step_idx)

    def evaluate(self, val_source, val_target, batch_size):
        indices = np.arange(len(val_source))
        np.random.shuffle(indices)
        self.sess.run(tf.local_variables_initializer())
        for i in range(val_source.shape[0] // batch_size):
            mb_idx = indices[i*batch_size:(i+1)*batch_size]
            self.sess.run([self.update_mean_kl_loss, self.update_mean_reconstruction_loss], feed_dict={
                self.source_states: val_source[mb_idx],
                self.target_states: val_target[mb_idx]
            })
        self.val_writer.add_summary(self.sess.run(self.merge_summary), self.get_step_idx())
        return self.sess.run([self.mean_reconstruction_loss, self.mean_kl_loss])

class ConvVAE(VAE):
    """
        Convolutional VAE class.
        Achitecture from: https://github.com/hardmaru/WorldModelsExperiments/blob/master/carracing/vae/vae.py
    """

    def __init__(self, source_shape, target_shape=None, **kwargs):
        """
            Define the encoder and decoder for the convolutional VAE,
            then initialize the base VAE.
            
            Note: This class is not general purpose with respect to source image shapes.
            Tested and adjusted to work with 160x80x3.
        """
        target_shape = source_shape if target_shape is None else target_shape

        def build_encoder(x):
            x = tf.layers.conv2d(x, filters=32,  kernel_size=4, strides=2, activation=tf.nn.relu, padding="valid", name="conv1")
            x = tf.layers.conv2d(x, filters=64,  kernel_size=4, strides=2, activation=tf.nn.relu, padding="valid", name="conv2")
            x = tf.layers.conv2d(x, filters=128, kernel_size=4, strides=2, activation=tf.nn.relu, padding="valid", name="conv3")
            x = tf.layers.conv2d(x, filters=256, kernel_size=4, strides=2, activation=tf.nn.relu, padding="valid", name="conv4")
            self.encoded_shape = x.shape[1:]
            x = tf.layers.flatten(x, name="flatten")
            return x

        def build_decoder(z):
            x = tf.layers.dense(z, np.prod(self.encoded_shape), activation=None, name="dense1")
            x = tf.reshape(x, (-1, *self.encoded_shape))
            x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=4, strides=2, activation=tf.nn.relu, padding="valid", name="deconv1")
            x = tf.layers.conv2d_transpose(x, filters=64,  kernel_size=4, strides=2, activation=tf.nn.relu, padding="valid", name="deconv2")
            x = tf.layers.conv2d_transpose(x, filters=32,  kernel_size=5, strides=2, activation=tf.nn.relu, padding="valid", name="deconv3")
            x = tf.layers.conv2d_transpose(x, filters=target_shape[-1], kernel_size=4, strides=2, activation=None, padding="valid", name="deconv4")
            assert x.shape[1:] == target_shape, f"{x.shape[1:]} != {target_shape}"
            return x
            
        super().__init__(source_shape, target_shape, build_encoder, build_decoder, **kwargs)


class MlpVAE(VAE):
    """
        Multi-layered perceptron VAE class.
    """

    def __init__(self, source_shape, target_shape=None,
                 encoder_sizes=(512, 256),
                 decoder_sizes=(256, 512),
                 **kwargs):
        """
            Define the encoder and decoder for the MLP VAE,
            then initialize the base VAE.
        """

        target_shape = source_shape if target_shape is None else target_shape

        def build_mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
            for h in hidden_sizes[:-1]:
                x = tf.layers.dense(x, units=h, activation=activation)
            return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

        def build_encoder(x):
            x = tf.layers.flatten(x, name="flattened_source")
            return build_mlp(x, hidden_sizes=encoder_sizes, activation=tf.nn.relu, output_activation=tf.nn.relu)

        def build_decoder(z):
            return build_mlp(z, hidden_sizes=list(decoder_sizes) + [np.prod(target_shape)], activation=tf.nn.relu, output_activation=None)

        super().__init__(source_shape, target_shape, build_encoder, build_decoder, **kwargs)
        