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


class VAE():
    def __init__(self, input_shape, build_encoder_fn, build_decoder_fn,
                 z_dim=512, beta=1.0, learning_rate=1e-4, kl_tolerance=0.5,
                 model_name="vae", models_dir=".", loss_fn=bce_loss,
                 training=True, reuse=tf.AUTO_REUSE,
                 **kwargs):
        # Create vae
        self.input_shape = input_shape
        self.z_dim = z_dim
        self.beta = beta
        self.kl_tolerance = kl_tolerance
        with tf.variable_scope("vae", reuse=reuse):
            # Get and verify input
            self.input_states = tf.placeholder(shape=(None, *input_shape), dtype=tf.float32, name="input_state_placeholder")
            verify_input_op = tf.Assert(tf.reduce_all(tf.logical_and(self.input_states >= 0, self.input_states <= 1)),
                                        ["min=", tf.reduce_min(self.input_states), "max=", tf.reduce_max(self.input_states)],
                                        name="verify_input")
            with tf.control_dependencies([verify_input_op]):
                if training:
                    self.input_states = tf.image.random_flip_left_right(self.input_states)
                else:
                    self.input_states = tf.multiply(self.input_states, 1, name="input_state_identity")

            # Encode image
            with tf.variable_scope("encoder", reuse=False):
                encoded = build_encoder_fn(self.input_states)

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
            self.saver = tf.train.Saver()
            if training:
                # Reconstruction loss
                self.flattened_input = tf.layers.flatten(self.input_states, name="flattened_input")
                self.reconstruction_loss = tf.reduce_mean(
                    tf.reduce_sum(
                        loss_fn(labels=self.flattened_input, logits=self.reconstructed_logits, targets=self.reconstructed_states),
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

                # Summary
                self.mean_kl_loss, self.update_mean_kl_loss = tf.metrics.mean(self.kl_loss)
                self.mean_reconstruction_loss, self.update_mean_reconstruction_loss = tf.metrics.mean(self.reconstruction_loss)
                self.merge_summary = tf.summary.merge([
                    tf.summary.scalar("kl_loss", self.mean_kl_loss),
                    tf.summary.scalar("reconstruction_loss", self.mean_reconstruction_loss)
                ])

                # Create optimizer
                self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                self.train_step = self.optimizer.minimize(self.loss)

            # Set model dirs
            self.model_name = model_name
            self.models_dir = os.path.join(models_dir, "models", model_name)
            self.log_dir = os.path.join(models_dir, "logs", model_name)
            self.dirs = [self.models_dir, self.log_dir]
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

    def generate_from_latent(self, z):
        return self.sess.run(self.reconstructed_states, feed_dict={
                self.sample: z
            })

    def reconstruct(self, input_states):
        return self.sess.run(self.reconstructed_states, feed_dict={
                self.input_states: input_states
            })

    def encode(self, input_states):
        return self.sess.run(self.mean, feed_dict={
                self.input_states: input_states
            })

    def save(self):
        model_checkpoint = os.path.join(self.models_dir, "model.ckpt")
        self.saver.save(self.sess, model_checkpoint, global_step=self.step_idx)
        print("Model checkpoint saved to {}".format(model_checkpoint))

    def load_latest_checkpoint(self):
        # Load checkpoint
        model_checkpoint = tf.train.latest_checkpoint(self.models_dir)
        if model_checkpoint:
            try:
                self.saver.restore(self.sess, model_checkpoint)
                print("Model checkpoint restored from {}".format(model_checkpoint))
                return True
            except Exception as e:
                print(e)
                return False

    def get_step_idx(self):
        return tf.train.global_step(self.sess, self.step_idx)

    def train_one_epoch(self, train_images, batch_size):
        np.random.shuffle(train_images)
        self.sess.run(tf.local_variables_initializer())
        for i in range(train_images.shape[0] // batch_size):
            self.sess.run([self.train_step, self.update_mean_kl_loss, self.update_mean_reconstruction_loss], feed_dict={
                self.input_states: train_images[i*batch_size:(i+1)*batch_size]
            })
        self.train_writer.add_summary(self.sess.run(self.merge_summary), self.get_step_idx())
        self.sess.run(self.inc_step_idx)

    def evaluate(self, val_images, batch_size):
        self.sess.run(tf.local_variables_initializer())
        for i in range(val_images.shape[0] // batch_size):
            self.sess.run([self.update_mean_kl_loss, self.update_mean_reconstruction_loss], feed_dict={
                self.input_states: val_images[i*batch_size:(i+1)*batch_size]
            })
        self.val_writer.add_summary(self.sess.run(self.merge_summary), self.get_step_idx())
        return self.sess.run([self.mean_reconstruction_loss, self.mean_kl_loss])

class ConvVAE(VAE):
    def __init__(self, input_shape, **kwargs):
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
            x = tf.layers.conv2d_transpose(x, filters=input_shape[-1], kernel_size=4, strides=2, activation=None, padding="valid", name="deconv4")
            assert x.shape[1:] == input_shape, f"{x.shape[1:]} != {input_shape}"
            return x
            
        super().__init__(input_shape, build_encoder, build_decoder, **kwargs)


class MlpVAE(VAE):
    def __init__(self, input_shape,
                 encoder_sizes=(512, 256),
                 decoder_sizes=(256, 512),
                 **kwargs):

        def build_mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
            for h in hidden_sizes[:-1]:
                x = tf.layers.dense(x, units=h, activation=activation)
            return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

        def build_encoder(x):
            x = tf.layers.flatten(x, name="flattened_input")
            return build_mlp(x, hidden_sizes=encoder_sizes, activation=tf.nn.relu, output_activation=tf.nn.relu)

        def build_decoder(z):
            return build_mlp(z, hidden_sizes=list(decoder_sizes) + [np.prod(input_shape)], activation=tf.nn.relu, output_activation=None)

        super().__init__(input_shape, build_encoder, build_decoder, **kwargs)
        