import tensorflow as tf
import east_model
import data_processor
import tf_slim as slim
import cv2
import os
import numpy as np
from east_utils import la_nms

BATCH_SIZE = 8
IMG_SIZE = 512
RESTORE = False



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

                score_map = s[m][:, :, 0]
                geo_map = g[m]

                threshold = 0.7  # Threshold to filter low-confidence predictions
                polys = []

                for i in range(score_map.shape[0]):
                    for j in range(score_map.shape[1]):
                        if score_map[i, j] > threshold:
                            top, right, bottom, left, theta = geo_map[i, j]

                            x_center = j * 4
                            y_center = i * 4
                            x1 = int(x_center - left)
                            y1 = int(y_center - top)
                            x2 = int(x_center + right)
                            y2 = int(y_center + bottom)

                            poly = [x1, y1, x2, y1, x2, y2, x1, y2, score_map[i, j]]
                            polys.append(poly)

                if len(polys) > 0:
                    
                    polys = np.array(polys)
                    final_polys = la_nms(polys)

                    for poly in final_polys:
                        x1, y1, x2, y2 = int(poly[0]), int(poly[1]), int(poly[2]), int(poly[5])
                        cv2.rectangle(img_show, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle for filtered box

                
                cv2.imwrite('/content/drive/MyDrive/DS_store/DS_train/Det_train/Detection/Demo' + str(n) + '_' + str(m) + '_img_with_nms.jpg', img_show)
                cv2.imwrite('/content/drive/MyDrive/DS_store/DS_train/Det_train/Detection/score' + str(n) + '_' + str(m) + '_scr.jpg', s[m] * 255)

if __name__ == '__main__':
    valid('/content/drive/MyDrive/DS_store/DS_train/Det_train/input_img',
          '/content/drive/MyDrive/DS_store/DS_train/Det_train/input_gt',
          '/content/drive/MyDrive/DS_store/DS_train/Det_train/valid.list')
