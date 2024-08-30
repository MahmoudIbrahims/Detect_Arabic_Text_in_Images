import tensorflow as tf
import tf_slim as slim
import east_model
import data_processor

# Set the logging level to suppress warnings
tf.get_logger().setLevel('ERROR')


BATCH_SIZE = 8
IMG_SIZE = 512
RESTORE = False
tf.compat.v1.disable_eager_execution()


def train(img_dir, gt_dir, train_list, pretrained_path):
      # Reset the default graph to avoid variable conflicts
    tf.compat.v1.reset_default_graph()

    img_input = tf.compat.v1.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3])
    gt_input = tf.compat.v1.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_SIZE // 4, IMG_SIZE // 4, 6])

    network = east_model.EAST(training=True, max_len=IMG_SIZE)

   # Use a variable scope to control variable reuse

    with tf.compat.v1.variable_scope('resnet_v1_50', reuse=tf.compat.v1.AUTO_REUSE):

        pred_score, pred_gmt = network.build(img_input)

    loss = network.loss(gt_input[:, :, :, 0:1], pred_score, gt_input[:, :, :, 1:6], pred_gmt)

    global_step = tf.Variable(0, trainable=False, name='global_step') # Use tf.Variable

    def get_learning_rate(global_step):
      learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.0001,
        decay_steps=10000,
        decay_rate=0.94,
        staircase=True
      )
      return learning_rate_fn(global_step)

    optimizer = tf.compat.v1.train.AdamOptimizer(
          learning_rate=get_learning_rate(global_step)).minimize(loss, global_step=global_step)
    
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer
  
    restore_op = slim.assign_from_checkpoint_fn(pretrained_path, slim.get_trainable_variables(),
                                                ignore_missing_vars=True)
    saver = tf.compat.v1.train.Saver()
    cfg = tf.compat.v1.ConfigProto()
    cfg.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=cfg) as sess:
        if RESTORE:
            saver.restore(sess, tf.train.latest_checkpoint('./checkpoint'))
        else:
            sess.run(tf.compat.v1.global_variables_initializer())
            restore_op(sess)

        data_list = data_processor.read_lines(train_list)
        for step in range(400):
            img_batch, gt_batch = data_processor.next_batch(img_dir, gt_dir, data_list, BATCH_SIZE)
            s, g, l, _ = sess.run([pred_score, pred_gmt, loss, train_op], 
                       feed_dict={img_input: img_batch, gt_input: gt_batch})

            if step % 100 == 0 and step > 0:
                saver.save(sess=sess, save_path='/content/drive/MyDrive/DS_store/DS_train/Det_train/pretrained/Checkpoint', global_step=step)
            if step % 10 == 0:
                print("the step is :",step, "and the loss is" , l)


if __name__ == '__main__':
    train('/content/drive/MyDrive/DS_store/DS_train/Det_train/input_img',
          '/content/drive/MyDrive/DS_store/DS_train/Det_train/input_gt',
          '/content/drive/MyDrive/DS_store/DS_train/Det_train/train.list',
          '/content/drive/MyDrive/DS_store/DS_train/Det_train/pretrained/resnet_v1_50.ckpt')
