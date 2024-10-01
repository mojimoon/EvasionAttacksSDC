from model import SDC_model_epoch
import os
from utilities import Dave_data, val_generator
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, balanced_accuracy_score
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import metrics
import sys

IMAGE_FILE = '/home/jzhang2297/data/dave_test/driving_dataset/data.txt'
IMAGE_FOLDER = '/home/jzhang2297/data/dave_test/driving_dataset/'
pwd = os.getcwd()
model_path = os.path.join(pwd, 'models', 'sdc_epoch_dropout.h5')

data = Dave_data(IMAGE_FILE, IMAGE_FOLDER, 10000)

'''
    def test(self, testX=None, testy=None):
        test_generator = val_generator(testX, testy,
                                             batch_size=self.hp_params.batch_size)
        self.mode = 'test'

        # rebuild the graph
        tf.reset_default_graph()
        self.model_graph()
        cur_checkpoint = tf.train.latest_checkpoint(self.save_dir)
        print('****************In test_rpst. Loading saved parameters from', self.save_dir)
        if cur_checkpoint is None:
            print("No saved parameters")
            return
        # load parameters
        saver = tf.train.Saver()
        eval_dir = os.path.join(self.save_dir, 'eval')
        sess = tf.Session()
        with sess:
            saver.restore(sess, cur_checkpoint)
            with sess.as_default():
                pred = []
                for X_batch, y_batch in tqdm(test_generator):
                    test_dict = {
                        self.x_input: X_batch,
                        self.y_input: y_batch,
                        self.is_training: False
                    }
                    _y_pred = sess.run(self.y_pred, feed_dict=test_dict)
                    pred.append(_y_pred)
                y_pred = np.concatenate(pred)
                import pdb; pdb.set_trace()

            from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, balanced_accuracy_score
            accuracy = accuracy_score(testy, y_pred)
            b_accuracy = balanced_accuracy_score(testy, y_pred)

            MSG = "The accuracy on the test dataset is {:.5f}%"
            print(MSG.format(accuracy * 100))
            MSG = "The balanced accuracy on the test dataset is {:.5f}%"
            print(MSG.format(b_accuracy * 100))
            sess.close()

        return accuracy
'''

def test():
    test_gen = val_generator(data.test_X, data.test_y, batch_size=128)

    sess = tf.Session()
    with sess:
        model = SDC_model_epoch(model_path, session=sess)
        loss, acc = [], []
        for X_batch, y_batch in test_gen:
            _y_pred = model.evaluate(X_batch, y_batch)
            loss.append(_y_pred[0] * y_batch.shape[0])
            acc.append(_y_pred[1] * y_batch.shape[0])
        loss = np.sum(loss) / data.test_X.shape[0]
        accuracy = np.sum(acc) / data.test_X.shape[0]
        # print('Test loss:', loss)
        # print('Test accuracy:', accuracy)

    return accuracy

'''
    def retrain(self, candidateX=None, candidatey=None, testX=None, testy=None, epochs=None):
        self.hp_params.n_epochs = epochs
        print('retraining for {0} epochs'.format(self.hp_params.n_epochs))
        # use candidate set for retraining
        train_generator = data_generator(candidateX, candidatey,
                                     batch_size=self.hp_params.batch_size)

        # rebuild the graph
        tf.reset_default_graph()
        self.model_graph()
        cur_checkpoint = tf.train.latest_checkpoint(self.save_dir)
        print('***************In retrain function. Loading saved parameters from', self.save_dir)

        # load parameters
        saver = tf.train.Saver()
        sess = tf.Session()

        summary_writer = tf.summary.FileWriter(self.save_dir_retrain, sess.graph)
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('loss', self.cross_entropy)
        merged_summaries = tf.summary.merge_all()

        # optimizer
        global_train_step = tf.train.get_or_create_global_step()
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            optimizer = tf.train.AdamOptimizer(self.hp_params.learning_rate).minimize(self.cross_entropy,global_step=global_train_step)

        with sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, cur_checkpoint)

            training_time = 0.0
            train_input.reset_cursor()
            output_steps = 1
            best_f1_val, best_acc_val = 0., 0.
            for X_batch, y_batch in tqdm(train_generator):
                train_dict = {
                    self.x_input: X_batch,
                    self.y_input: y_batch,
                    self.is_training: True
                }

                if (step_idx) % output_steps == 0:
                    validation_generator = val_generator(testX, testy,
                                                         batch_size=self.hp_params.batch_size)
                    print('Step {}/{}:{}'.format(step_idx + 1, train_input.steps, datetime.now()))
                    val_input.reset_cursor()
                    val_res_list = [sess.run([self.accuracy, self.y_pred], feed_dict={self.x_input: valX_batch,
                                                                                      self.y_input: valy_batch,
                                                                                      self.is_training: False}) \
                                    for [valX_batch, valy_batch] in tqdm(validation_generator)
                                    ]
                    val_res = np.array(val_res_list, dtype=object)
                    _acc = np.mean(val_res[:, 0])
                    _pred_y = np.concatenate(val_res[:, 1])

                    if step_idx != 0:
                        #print('    {} samples per second'.format(
                        #    output_steps * self.hp_params.batch_size / training_time))
                        training_time = 0.

                    summary = sess.run(merged_summaries, feed_dict=train_dict)
                    summary_writer.add_summary(summary, global_train_step.eval(sess))

                    if best_acc < _acc:
                        best_acc = _acc
                        if not os.path.exists(self.save_dir_retrain):
                            os.makedirs(self.save_dir_retrain)
                        saver.save(sess,
                                   os.path.join(self.save_dir_retrain, 'checkpoint'),
                                   global_step=global_train_step)
                start = default_timer()
                step_idx += 1
                epoch = int((step_idx * self.hp_params.batch_size)/len(candidateX))
                if epoch > 10: # Udacity 1000*bs= number of samples used for training, 50000~ epoch=50
                    print('finish training')
                    break
                sess.run(optimizer, feed_dict=train_dict)
                end = default_timer()
                training_time = training_time + end - start
        sess.close()
        return best_acc_val
'''

def retrain():
    with tf.Session() as sess:
        model = SDC_model_epoch(model_path, session=sess)
        # canX, cany = metrics.Random(sess, data.train_X, data.train_y, 0.5)
        canX, cany = metrics.deepgini(sess, data.train_X, data.train_y, model, 0.5)
        testX, testy = data.test_X, data.test_y
        print('canX:', canX.shape, 'cany:', cany.shape)
        print('testX:', testX.shape, 'testy:', testy.shape)
        accuracy = model.retrain(canX, cany, testX, testy, epochs=10)
        print('Retraining accuracy:', accuracy)

if __name__ == '__main__':
    retrain()
