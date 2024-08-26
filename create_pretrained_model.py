import tensorflow as tf
from tensorflow.keras.applications import ResNet50

# Load the pre-trained ResNet-50 model
model = ResNet50(weights='imagenet', include_top=True)

# Create a TensorFlow 1.x session and saver
with tf.compat.v1.Session() as sess:
    # Initialize model variables
    sess.run(tf.compat.v1.global_variables_initializer())
    # Create a Saver object
    saver = tf.compat.v1.train.Saver()
    # Save the model
    saver.save(sess, '/content/drive/MyDrive/DS_store/DS_train/Det_train/pretrained/resnet_v1_50.ckpt')

