import numpy as np
import tensorflow as tf
import random
from dataloader import Gen_Data_loader, Dis_dataloader
from Discriminator import Discriminator
from target_lstm import TARGET_LSTM
from target_lstm20 import TARGET_LSTM20
from LeakGANModel import  LeakGAN
import cPickle
import os


#import numexpr as ne

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('restore', False, 'Training or testing a model')
flags.DEFINE_boolean('resD', False, 'Training or testing a D model')
flags.DEFINE_integer('length', 20, 'The length of toy data')
flags.DEFINE_string('model', "", 'Model NAME')
#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 32 # embedding dimension
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
SEQ_LENGTH = FLAGS.length # sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 80 # supervise (maximum likelihood estimation) epochs
SEED = 88
BATCH_SIZE = 64
LEARNING_RATE = 0.01
GOAL_SIZE = 16
STEP_SIZE = 4
#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 64


dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
if FLAGS.length == 20:
    dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
    dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
    LEARNING_RATE = 0.0015
    EMB_DIM = 32  # embedding dimension
    HIDDEN_DIM = 32  # hidden state dimension of lstm cell
elif FLAGS.length == 40:
    dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20,30,40]
    dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160,160]
    LEARNING_RATE = 0.0005
    EMB_DIM = 64
    HIDDEN_DIM = 64
else:
    exit(0)
print(SEQ_LENGTH)


GOAL_OUT_SIZE = sum(dis_num_filters)

dis_dropout_keep_prob = 1.0
dis_l2_reg_lambda = 0.2
dis_batch_size = 64

#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = 800
positive_file = 'save/real_data.txt'
negative_file = 'save/generator_sample.txt'
eval_file = 'save/eval_file.txt'
generated_num = 10000
model_path = './ckpts'

def generate_samples(sess, trainable_model, batch_size, generated_num, output_file,train = 1):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess,1.0,train))

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)

def target_loss(sess, target_lstm, data_loader):
    # target_loss means the oracle negative log-likelihood tested with the oracle model "target_lstm"
    # For more details, please see the Section 4 in https://arxiv.org/abs/1609.05473
    nll = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: batch})
        nll.append(g_loss)
    return np.mean(nll)

def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss,_,_ = trainable_model.pretrain_step(sess, batch,1.0)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)

def redistribution( idx, total, min_v):
    idx = (idx + 0.0) / (total + 0.0) * 16.0
    return (np.exp(idx - 8.0) / (1.0 + np.exp(idx - 8.0)))

def rescale( reward, rollout_num=1.0):
    reward = np.array(reward)
    x, y = reward.shape
    ret = np.zeros((x, y))
    for i in range(x):
        l = reward[i]
        rescalar = {}
        for s in l:
            rescalar[s] = s
        idxx = 1
        min_s = 1.0
        max_s = 0.0
        for s in rescalar:
            rescalar[s] = redistribution(idxx, len(l), min_s)
            idxx += 1
        for j in range(y):
            ret[i, j] = rescalar[reward[i, j]]
    return ret

def get_reward(model,dis, sess, input_x, rollout_num, dis_dropout_keep_prob):
    rewards = []
    for i in range(rollout_num):
        for given_num in range(1, model.sequence_length / model.step_size):
            real_given_num = given_num * model.step_size
            feed = {model.x: input_x, model.given_num: real_given_num, model.drop_out: 1.0}
            samples = sess.run(model.gen_for_reward, feed)
            # print samples.shape
            feed = {dis.D_input_x: samples, dis.dropout_keep_prob: 1.0}
            ypred_for_auc = sess.run(dis.ypred_for_auc, feed)
            ypred = np.array([item[1] for item in ypred_for_auc])
            if i == 0:
                rewards.append(ypred)
            else:
                rewards[given_num - 1] += ypred

        # the last token reward
        feed = {dis.D_input_x: input_x,  dis.dropout_keep_prob: 1.0}
        ypred_for_auc = sess.run(dis.ypred_for_auc, feed)
        ypred = np.array([item[1] for item in ypred_for_auc])
        if i == 0:
            rewards.append(ypred)
        else:
            rewards[model.sequence_length / model.step_size - 1] += ypred
    rewards = rescale(np.array(rewards), rollout_num)
    rewards = np.transpose(np.array(rewards)) / (1.0 * rollout_num)  # batch_size x seq_length
    return rewards

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0

    gen_data_loader = Gen_Data_loader(BATCH_SIZE,FLAGS.length)
    likelihood_data_loader = Gen_Data_loader(BATCH_SIZE,FLAGS.length) # For testing
    vocab_size = 5000
    file = open('save/target_params.pkl', 'rb')
    target_params = cPickle.load(file)
    
    dis_data_loader = Dis_dataloader(BATCH_SIZE,SEQ_LENGTH)
    discriminator = Discriminator(SEQ_LENGTH,num_classes=2,vocab_size=vocab_size,dis_emb_dim=dis_embedding_dim,filter_sizes=dis_filter_sizes,num_filters=dis_num_filters,
                        batch_size=BATCH_SIZE,hidden_dim=HIDDEN_DIM,start_token=START_TOKEN,goal_out_size=GOAL_OUT_SIZE,step_size=4)
    leakgan = LeakGAN(SEQ_LENGTH,num_classes=2,vocab_size=vocab_size,emb_dim=EMB_DIM,dis_emb_dim=dis_embedding_dim,filter_sizes=dis_filter_sizes,num_filters=dis_num_filters,
                        batch_size=BATCH_SIZE,hidden_dim=HIDDEN_DIM,start_token=START_TOKEN,goal_out_size=GOAL_OUT_SIZE,goal_size=GOAL_SIZE,step_size=4,D_model=discriminator,
                      learning_rate=LEARNING_RATE)
    if SEQ_LENGTH == 40:
        target_lstm = TARGET_LSTM(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)  # The oracle model
    else:
        target_lstm = TARGET_LSTM20(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN,target_params)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    generate_samples(sess, target_lstm, BATCH_SIZE, generated_num, positive_file, 0)
    for a in range(1):
        g = sess.run(leakgan.gen_x,feed_dict={leakgan.drop_out:0.8,leakgan.train:1})
        print(g)

        print("epoch:",a,"  ")

    log = open('save/experiment-log.txt', 'w')
    gen_data_loader.create_batches(positive_file)
    saver_variables = tf.global_variables()
    saver = tf.train.Saver(saver_variables)
    model = tf.train.latest_checkpoint(model_path)
    print(model)
    if FLAGS.restore and model:
        # model = tf.train.latest_checkpoint(model_path)
        # if model and FLAGS.restore:
        if model_path+'/' + FLAGS.model:
            print(model_path+'/' + FLAGS.model)
            saver.restore(sess, model_path+'/' + FLAGS.model)
        else:
            saver.restore(sess, model)
    else:
        if FLAGS.resD and model_path + '/' + FLAGS.model:
                print(model_path + '/' + FLAGS.model)
                saver.restore(sess, model_path + '/' + FLAGS.model)

                print('Start pre-training...')
                log.write('pre-training...\n')
                for epoch in range(PRE_EPOCH_NUM):
                    loss = pre_train_epoch(sess, leakgan, gen_data_loader)
                    if epoch % 5 == 0:
                        generate_samples(sess, leakgan, BATCH_SIZE, generated_num, eval_file, 0)
                        likelihood_data_loader.create_batches(eval_file)
                        test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
                        print('pre-train epoch ', epoch, 'test_loss ', test_loss)
                        buffer = 'epoch:\t' + str(epoch) + '\tnll:\t' + str(test_loss) + '\n'
                        log.write(buffer)
                        generate_samples(sess, target_lstm, BATCH_SIZE, generated_num, eval_file, 0)
                        likelihood_data_loader.create_batches(eval_file)
                        test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
                        print("Groud-Truth:", test_loss)
                saver.save(sess, model_path + '/leakgan_pre')
        else:
                print('Start pre-training discriminator...')
                # Train 3 epoch on the generated data and do this for 50 times
                for i in range(10):
                    for _ in range(5):
                        generate_samples(sess, leakgan, BATCH_SIZE, generated_num, negative_file,0)
                        generate_samples(sess, target_lstm, BATCH_SIZE, generated_num, positive_file,0)
                        # gen_data_loader.create_batches(positive_file)
                        dis_data_loader.load_train_data(positive_file, negative_file)
                        for _ in range(3):
                            dis_data_loader.reset_pointer()
                            for it in range(dis_data_loader.num_batch):
                                x_batch, y_batch = dis_data_loader.next_batch()
                                feed = {
                                    discriminator.D_input_x: x_batch,
                                    discriminator.D_input_y: y_batch,
                                    discriminator.dropout_keep_prob: dis_dropout_keep_prob
                                }
                                D_loss,_ = sess.run([discriminator.D_loss,discriminator.D_train_op], feed)
                                # # print 'D_loss ', D_loss
                                # buffer =  str(D_loss) + '\n'
                                # log.write(buffer)
                        leakgan.update_feature_function(discriminator)
                    saver.save(sess, model_path + '/leakgan_preD')

            # saver.save(sess, model_path + '/leakgan')
        #  pre-train generator
                    print('Start pre-training...')
                    log.write('pre-training...\n')
                    for epoch in range(PRE_EPOCH_NUM/10):
                        loss = pre_train_epoch(sess, leakgan, gen_data_loader)
                        if epoch % 5 == 0:
                            generate_samples(sess, leakgan, BATCH_SIZE, generated_num, eval_file,0)
                            likelihood_data_loader.create_batches(eval_file)
                            test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
                            print('pre-train epoch ', epoch, 'test_loss ', test_loss)
                            buffer = 'epoch:\t'+ str(epoch) + '\tnll:\t' + str(test_loss) + '\n'
                            log.write(buffer)
                            generate_samples(sess, target_lstm, BATCH_SIZE, generated_num, eval_file, 0)
                            likelihood_data_loader.create_batches(eval_file)
                            test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
                            print("Groud-Truth:", test_loss)
                saver.save(sess, model_path + '/leakgan_pre')

    gencircle = 1
    #
    print('#########################################################################')
    print('Start Adversarial Training...')
    log.write('adversarial training...\n')
    for total_batch in range(TOTAL_BATCH):
        # Train the generator for one step
        for it in range(1):

            for gi in range(gencircle):
                samples = leakgan.generate(sess,1.0,1)
                rewards = get_reward(leakgan, discriminator,sess, samples, 4, dis_dropout_keep_prob)
                feed = {leakgan.x: samples, leakgan.reward: rewards,leakgan.drop_out:1.0}
                _,_,g_loss,w_loss = sess.run([leakgan.manager_updates,leakgan.worker_updates,leakgan.goal_loss,leakgan.worker_loss], feed_dict=feed)
                print('total_batch: ', total_batch, "  ",g_loss,"  ", w_loss)

        # Test
        if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
            generate_samples(sess, leakgan, BATCH_SIZE, generated_num, eval_file,0)
            likelihood_data_loader.create_batches(eval_file)
            test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            buffer = 'epoch:\t' + str(total_batch) + '\tnll:\t' + str(test_loss) + '\n'
            print('total_batch: ', total_batch, 'test_loss: ', test_loss)
            log.write(buffer)
            generate_samples(sess, target_lstm, BATCH_SIZE, generated_num, eval_file, 0)
            likelihood_data_loader.create_batches(eval_file)
            test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            print("Groud-Truth:" ,test_loss)

        # Train the discriminator
        for _ in range(5):
            generate_samples(sess, leakgan, BATCH_SIZE, generated_num, negative_file,0)
            generate_samples(sess, target_lstm, BATCH_SIZE, generated_num, positive_file,0)
            dis_data_loader.load_train_data(positive_file, negative_file)

            for _ in range(3):
                dis_data_loader.reset_pointer()
                for it in range(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.D_input_x: x_batch,
                        discriminator.D_input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    D_loss, _ = sess.run([discriminator.D_loss, discriminator.D_train_op], feed)
                    # print 'D_loss ', D_loss
            leakgan.update_feature_function(discriminator)
    log.close()


if __name__ == '__main__':
    main()
