#test_resnet.py
import tensorflow as tf
import resnet_v1

def test_resnet_v1_50():
    # Define the input tensor
    input_tensor = tf.compat.v1.placeholder(tf.float32, [None, 224, 224, 3], name='input')

    # Build the ResNet model, use a variable scope with reuse set to tf.AUTO_REUSE
    with tf.compat.v1.variable_scope('resnet_v1_50', reuse=tf.compat.v1.AUTO_REUSE):
        logits, end_points = resnet_v1.resnet_v1_50(input_tensor, is_training=True)


    # Initialize global variables
    init = tf.compat.v1.global_variables_initializer()

    # Run the session
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        # Feed dummy data into the model and get the output
        dummy_input = tf.random.normal([1, 224, 224, 3])
        dummy_input_val = sess.run(dummy_input) # Evaluate the tensor to get a NumPy array

        logits_output = sess.run(logits, feed_dict={input_tensor: dummy_input_val})
        print("Logits Output:")
        print(logits_output)

if __name__ == "__main__":
    test_resnet_v1_50()
