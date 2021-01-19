import tensorflow as tf

embedding = tf.constant(
    [
        [0.21, 0.41, 0.51, 0.11],
        [0.22, 0.42, 0.52, 0.12],
        [0.23, 0.43, 0.53, 0.13],
        [0.24, 0.44, 0.54, 0.14]
    ], dtype=tf.float32)

feature_batch = tf.constant([2, 3, 1, 0])
feature_batch_one_hot = tf.one_hot(feature_batch, depth=4)
get_embedding2 = tf.matmul(feature_batch_one_hot, embedding)

get_embedding1 = tf.nn.embedding_lookup(embedding, feature_batch)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    embedding1, embedding2 = sess.run([get_embedding1, get_embedding2])
    print(embedding1)
    print(embedding2)
