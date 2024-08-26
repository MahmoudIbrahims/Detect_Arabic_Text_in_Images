#east_model
import resnet_v1
import tensorflow as tf
import numpy as np


class EAST:
    def __init__(self, training, max_len):
        self.__training = training
        self.__max_len = max_len

        self.data_format = 'channels_last'  # Or 'channels_first' if you're using that format

    def __upsample(self, input_tensor):
        return tf.image.resize(input_tensor, [tf.shape(input_tensor)[1] * 2, tf.shape(input_tensor)[2] * 2])

    def __subtract_mean(self, input_tensor, mean=[123.68, 116.78, 103.94]):

    # Get the number of channels from the shape of the tensor
      c = input_tensor.shape[-1]
    # Split the tensor into its individual channels
      channels = tf.split(input_tensor, num_or_size_splits=c, axis=-1)
    # Subtract the mean from each channel
      for i in range(c):
        channels[i] -= mean[i]
    # Concatenate the channels back together
        return tf.concat(channels, axis=-1)


    #def __conv_block(self, input_tensor, n_filter, k_size):
          # Remove 'inputs=' and pass input_tensor directly as the first argument
       # out = tf.keras.layers.Conv2D(
       #                        input_tensor,
       #                        filters=n_filter,
        #                       kernel_size=k_size,
        #                       padding='same',
         #                      use_bias=False,
         #                      kernel_regularizer=tf.keras.regularizers.L2(0.001))
        #out = tf.layers.batch_normalization(inputs=out, training=self.__training)
       # return tf.nn.leaky_relu(features=out)


    def __conv_block(self, input_tensor, n_filter, k_size):
    # Remove 'inputs=' and pass input_tensor directly as the first argument
         out = tf.keras.layers.Conv2D(
                                 
                                filters=int(n_filter),  # Convert n_filter to integer
                                kernel_size=k_size,
                                strides=1,
                                padding='SAME',
                                use_bias=False,
                                data_format=self.data_format)(input_tensor)  # Pass input_tensor here
         out = tf.keras.layers.BatchNormalization()(out)
         out = tf.nn.relu(out)
         return out

    def __feature_merge_block(self, f1, f2, n_filter):
        h = self.__upsample(f1)
        # Check if f2 is None and handle it appropriately
        if f2 is None:
            print("Warning: f2 is None. Skipping merge.") # Add a warning message
            return h  # Or handle it differently based on your model's logic
        
        
        # Ensure f2 is a Tensorflow tensor before concatenation
        f2 = tf.convert_to_tensor(f2, dtype=tf.float32)
        h = tf.concat([f2, h], axis=-1)
        h = self.__conv_block(h, n_filter, 1)
        return self.__conv_block(h, n_filter, 3)

    
    
    
    def __feature_extract(self, input_tensor):
        _, end_points = resnet_v1.resnet_v1_50(inputs=input_tensor,
                                               is_training=self.__training,
                                               scope='resnet_v1_50')
        return [end_points['pool5'], end_points['pool4'], end_points['pool3'], end_points['pool2']]

    def __feature_merge(self, f):
        h = f[0]
        for i in range(3):
            h = self.__feature_merge_block(h, f[i + 1], 128 / (2 ** i))
        return self.__conv_block(h, 32, 3)

    def build(self, input_tensor):
        net = self.__subtract_mean(input_tensor)
        net = self.__feature_extract(net)
        net = self.__feature_merge(net)

        score = tf.keras.layers.Conv2D(inputs=net,
                                 filters=1,
                                 kernel_size=1,
                                 padding='same',
                                 activation=tf.math.sigmoid,
                                 kernel_regularizer=tf.keras.regularizers.L2 (0.001))
        distance = tf.keras.layers.Conv2D(inputs=net,
                                    filters=4,
                                    kernel_size=1,
                                    padding='same',
                                    activation=tf.math.sigmoid,
                                    kernel_regularizer=tf.keras.regularizers.L2 (0.001))
        angle = tf.keras.layers.Conv2D(inputs=net,
                                 filters=1,
                                 kernel_size=1,
                                 padding='same',
                                 activation=tf.nn.sigmoid,
                                 kernel_regularizer=tf.keras.regularizers.L2 (0.001))
        return score, tf.concat([distance * self.__max_len, (angle - 0.5) * np.pi / 2], axis=-1)

    def dice_loss(self, gt_scr, pred_scr, eps=1e-4):
        i = tf.math.reduce_sum(gt_scr * pred_scr)
        u = tf.math.reduce_sum(tf.clip_by_value(gt_scr + pred_scr, 0, 1)) + eps
        return 1. - (2 * i / u)

    def loss(self, gt_score, pred_score, gt_gmt, pred_gmt, eps=1e-4):
        score_loss = self.dice_loss(gt_score, pred_score)

        gt_top, gt_right, gt_bottom, gt_left, gt_theta = tf.split(gt_gmt, 5, axis=3)
        pred_top, pred_right, pred_bottom, pred_left, pred_theta = tf.split(pred_gmt, 5, axis=3)
        gt_area = (gt_top + gt_bottom) * (gt_right + gt_left)
        pred_area = (pred_top + pred_bottom) * (pred_right + pred_left)
        i_w = tf.math.minimum(gt_right, pred_right) + tf.math.minimum(gt_left, pred_left)
        i_h = tf.math.minimum(gt_top, pred_top) + tf.math.minimum(gt_bottom, pred_bottom)
        i_area = i_w * i_h
        u_area = gt_area + pred_area - i_area
        aabb_loss = -tf.log((i_area + eps) / u_area)
        theta_loss = 1 - tf.cos(gt_theta - pred_theta)
        gmt_loss = aabb_loss + 10 * theta_loss

        total_loss = tf.math.reduce_mean(gmt_loss * gt_score) + score_loss
        return tf.math.add_n ([total_loss] + tf.compat.v1.get_collection (tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES))
