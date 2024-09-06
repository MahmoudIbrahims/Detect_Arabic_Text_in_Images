import tensorflow as tf
import east_model
import data_processor
import tf_slim as slim
import cv2
import os

BATCH_SIZE = 8
IMG_SIZE = 512
RESTORE = False


#checkpoint_dir = '/content/drive/MyDrive/DS_store/DS_train/Det_train/pretrained/'
#print("Files in checkpoint directory:")
#for filename in os.listdir(checkpoint_dir):
#    print(filename)

def valid(img_dir, gt_dir, valid_list):
    data_list = data_processor.read_lines(valid_list)

    img_input = tf.compat.v1.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3])
    gt_input = tf.compat.v1.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_SIZE // 4, IMG_SIZE // 4, 6])

    network = east_model.EAST(training=False, max_len=IMG_SIZE)

    with tf.compat.v1.variable_scope('resnet_v1_50', reuse=tf.compat.v1.AUTO_REUSE):
        pred_score, pred_gmt = network.build(img_input)

    loss = network.loss(gt_input[:, :, :, 0:1], pred_score, gt_input[:, :, :, 1:6], pred_gmt)
    checkpoint_file ='/content/drive/MyDrive/DS_store/DS_train/Det_train/pretrained/Checkpoint-500'
    cfg = tf.compat.v1.ConfigProto()
    cfg.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=cfg) as sess:
        variables_to_restore = slim.get_variables_to_restore(exclude=['east/score_map', 'east/geo_map'])
        restorer = tf.compat.v1.train.Saver(variables_to_restore)
        restorer.restore(sess=sess, save_path=checkpoint_file)
     
        n_step = len(data_list) // BATCH_SIZE
        for n in range(n_step):
            img_batch, gt_batch = data_processor.next_batch(img_dir, gt_dir, data_list, BATCH_SIZE, n * BATCH_SIZE)
            s, g, l = sess.run([pred_score, pred_gmt, loss], feed_dict={img_input: img_batch, gt_input: gt_batch})

            print(n, l)
            for m in range(BATCH_SIZE):
                img_show = cv2.resize(img_batch[m], (IMG_SIZE // 4, IMG_SIZE // 4))
                img_show[:, :, 0] += s[m][:, :, 0] * 255
                img_show[:, :, 1] += s[m][:, :, 0] * 255
                img_show[:, :, 2] += s[m][:, :, 0] * 255
                cv2.imwrite('/content/drive/MyDrive/DS_store/DS_train/Det_train/Detection/Demo' + str(n) + '_' + str(m) + '_img.jpg', img_show)
                cv2.imwrite('/content/drive/MyDrive/DS_store/DS_train/Det_train/Detection/score' + str(n) + '_' + str(m) + '_scr.jpg', s[m] * 255)

if __name__ == '__main__':
    valid('/content/drive/MyDrive/DS_store/DS_train/Det_train/input_img',
          '/content/drive/MyDrive/DS_store/DS_train/Det_train/input_gt',
          '/content/drive/MyDrive/DS_store/DS_train/Det_train/valid.list')
