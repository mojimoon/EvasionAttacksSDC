import sys
sys.path.append('/home/jzhang2297/anomaly/malware/adv-self-driving')
import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import utilities as utils

def create_dataset(candidateX, candidatey, batch_size=128):
    dataset = tf.data.Dataset.from_generator(
        lambda: utils.val_generator(candidateX, candidatey, batch_size),
        # output_signature=(
        #     tf.TensorSpec(shape=(None, 128, 128, 3), dtype=tf.float32),
        #     tf.TensorSpec(shape=(None, 3), dtype=tf.float32)
        # )
        output_types=(tf.float32, tf.float32),
        output_shapes=((None, 128, 128, 3), (None, 3))
    )
    return dataset.prefetch(tf.data.experimental.AUTOTUNE)

def Random(sess, candidateX, candidatey, budget):
    selection_size = int(budget * candidateX.shape[0])
    select_idx = np.random.choice(np.arange(len(candidateX)), selection_size)
    selected_candidateX, selected_candidatey = candidateX[select_idx], candidatey[select_idx]

    return selected_candidateX, selected_candidatey

def deepgini(sess, candidateX, candidatey, model, budget):
    # dataset = tf.data.Dataset.from_tensor_slices((candidateX, candidatey)).batch(128)

    dataset = create_dataset(candidateX, candidatey, batch_size=128)

    @tf.function
    def compute_gini(probs):
        return 1 - tf.reduce_sum(probs ** 2, axis=1)
    
    scores = []

    for X_batch, y_batch in dataset:
        probs = model.prob(X_batch)
        gini = compute_gini(probs)
        scores.extend(gini.numpy())
    
    scores = np.array(scores)
    idx = np.argsort(scores)[::-1]
    selected_idx = idx[:int(len(idx) * budget)]
    return candidateX[selected_idx], candidatey[selected_idx], scores
