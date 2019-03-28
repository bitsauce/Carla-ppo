from collections import deque
import numpy as np
import tensorflow as tf
import types
import cv2
import scipy.signal

def preprocess_frame(frame):
    frame = frame[:-12, 6:-6] # Crop to 84x84
    frame = np.dot(frame[..., 0:3], [0.299, 0.587, 0.114])
    frame = frame / 255.0
    return np.expand_dims(frame, axis=-1)

class FrameStack():
    def __init__(self, initial_frame, stack_size=4, preprocess_fn=None):
        # Setup initial state
        self.frame_stack = deque(maxlen=stack_size)
        initial_frame = preprocess_fn(initial_frame) if preprocess_fn else initial_frame
        for _ in range(stack_size):
            self.frame_stack.append(initial_frame)
        self.state = np.stack(self.frame_stack, axis=-1)
        self.preprocess_fn = preprocess_fn
        
    def add_frame(self, frame):
        self.frame_stack.append(self.preprocess_fn(frame))
        self.state = np.stack(self.frame_stack, axis=-1)
        
    def get_state(self):
        return self.state

class VideoRecorder():
    def __init__(self, filename, frame_size):
        self.video_writer = cv2.VideoWriter(
            filename,
            cv2.VideoWriter_fourcc(*"MPEG"), 30,
            (frame_size[1], frame_size[0]))

    def add_frame(self, frame):
        self.video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def release(self):
        self.video_writer.release()

    def __del__(self):
        self.release()

def build_mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def create_polyak_update_ops(source_scope, target_scope, polyak=0.995):
    source_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=source_scope)
    target_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_scope)
    assert len(source_params) == len(target_params), "Source and target param count are different"
    for src, tgt in zip(source_params, target_params):
        assert src.shape == tgt.shape, "Source and target param shapes are different"
    polyak_update_op = tf.group([tgt.assign(polyak * tgt + (1.0 - polyak) * src) for src, tgt in zip(source_params, target_params)])
    assign_op = tf.group([tgt.assign(src) for src, tgt in zip(source_params, target_params)])
    return polyak_update_op, assign_op

def create_counter_variable(name):
    counter = types.SimpleNamespace()
    counter.var = tf.Variable(0, name=name, trainable=False)
    counter.inc_op = tf.assign(counter.var, counter.var + 1)
    return counter

def create_mean_metrics_from_dict(metrics):
    # Set up summaries for each metric
    update_metrics_ops = []
    summaries = []
    for name, (value, update_op) in metrics.items():
        summaries.append(tf.summary.scalar(name, value))
        update_metrics_ops.append(update_op)
    return tf.summary.merge(summaries), tf.group(update_metrics_ops)

def clip_grad(optimizer, params, loss, grad_clip):
    gvs = optimizer.compute_gradients(loss, var_list=params)
    capped_gvs = [(tf.clip_by_value(grad, -grad_clip, grad_clip), var) for grad, var in gvs]
    return optimizer.apply_gradients(capped_gvs)

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def compute_v_and_adv(rewards, values, bootstrapped_value, gamma, lam=1.0):
    rewards = np.array(rewards)
    values = np.array(list(values) + [bootstrapped_value])
    v = discount(np.array(list(rewards) + [bootstrapped_value]), gamma)[:-1]
    delta = rewards + gamma * values[1:] - values[:-1]
    adv = discount(delta, gamma * lam)
    return v, adv

def compute_returns(rewards, bootstrap_value, terminals, gamma):
    returns = []
    R = bootstrap_value
    for i in reversed(range(len(rewards))):
        R = rewards[i] + (1.0 - terminals[i]) * gamma * R
        returns.append(R)
    returns = reversed(returns)
    return np.array(list(returns))

def compute_gae(rewards, values, bootstrap_values, terminals, gamma, lam):
    rewards = np.array(rewards)
    values = np.array(list(values) + [bootstrap_values])
    terminals = np.array(terminals)
    deltas = rewards + (1.0 - terminals) * gamma * values[1:] - values[:-1]
    return scipy.signal.lfilter([1], [1, -gamma * lam], deltas[::-1], axis=0)[::-1]

def compute_gae_old(rewards, values, bootstrap_value, terminals, gamma, lam):
    values = np.array(list(values) + [bootstrap_value])
    last_gae_lam = 0
    advantages = np.zeros_like(rewards)
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * (1.0 - terminals[i]) - values[i]
        advantages[i] = last_gae_lam = delta + gamma * lam * (1.0 - terminals[i]) * last_gae_lam
    return advantages
    

# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
# for a new value newValue, compute the new count, new mean, the new M2.
# mean accumulates the mean of the entire dataset
# M2 aggregates the squared distance from the mean
# count aggregates the number of samples seen so far
def update(existingAggregate, newValue):
    (count, mean, M2) = existingAggregate
    count += 1 
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2

    return (count, mean, M2)

# retrieve the mean, variance and sample variance from an aggregate
def finalize(existingAggregate):
    (count, mean, M2) = existingAggregate
    (mean, variance, sampleVariance) = (mean, M2/count, M2/(count - 1)) 
    if count < 2:
        return float('nan')
    else:
        return (mean, variance, sampleVariance)