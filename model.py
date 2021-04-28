import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf_input = tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import data_processing

data = data_processing.load_data(download=True)
new_data = data_processing.convert2onehot(data)


# prepare training data
new_data = new_data.values.astype(np.float32)       # change to numpy array and float32
np.random.shuffle(new_data)                         # in chaos
sep = int(0.7*len(new_data))                        # 强转整型数字
train_data = new_data[:sep]                         # training data (70%)
test_data = new_data[sep:]                          # test data (30%)


# build network
tf_input = tf.placeholder(tf.float32, [None, 25], "input")  # length of data/class is 25
# tf_input = tf.disable_v2_behavior(tf.float32, [None, 25], "input")
tfx = tf_input[:, :21]  # features is 21
tfy = tf_input[:, 21:]

l1 = tf.layers.dense(tfx, 128, tf.nn.relu, name="l1")  # hidden layer 1
l2 = tf.layers.dense(l1, 128, tf.nn.relu, name="l2")  # hidden layer 2
out = tf.layers.dense(l2, 4, name="l3")   # output layer
prediction = tf.nn.softmax(out, name="pred")

loss = tf.losses.softmax_cross_entropy(onehot_labels=tfy, logits=out)  # output before softmax
accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(tfy, axis=1), predictions=tf.argmax(out, axis=1),)[1]  # not onehot_output, The [1] is output [0] is not
opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)  # select GSD optimizer
train_op = opt.minimize(loss)  # minimize

sess = tf.Session()
sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())) # initial all variables

# training
plt.ion() # No pause after plotting [plt.show()], change block-style to interactive-style
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
accuracies, steps = [], []
for t in range(4000):
    # training
    batch_index = np.random.randint(len(train_data), size=32)  #用于生成一个指定范围内的整数。生成的随机数n: a <= n <= b。
    sess.run(train_op, {tf_input: train_data[batch_index]})

    if t % 50 == 0:
        # testing
        acc_, pred_, loss_ = sess.run([accuracy, prediction, loss], {tf_input: test_data})  # training step
        accuracies.append(acc_)
        steps.append(t)
        print("Step: %i" % t,"| Accurate: %.2f" % acc_,"| Loss: %.2f" % loss_,)

        # visualize testing
        ax1.cla()  # clear axis1
        for c in range(4):
            bp = ax1.bar(c+0.1, height=sum((np.argmax(pred_, axis=1) == c)), width=0.2, color='red')
            bt = ax1.bar(c-0.1, height=sum((np.argmax(test_data[:, 21:], axis=1) == c)), width=0.2, color='blue')
        ax1.set_xticks(range(4), ["accepted", "good", "unaccepted", "very good"])
        ax1.legend(handles=[bp, bt], labels=["prediction", "target"])
        ax1.set_ylim((0, 400))
        ax2.cla()  # clear axis2
        ax2.plot(steps, accuracies, label="accuracy")  # plot figure2
        ax2.set_ylim(ymax=1)
        ax2.set_ylabel("accuracy")
        plt.pause(0.01)
# plt.clf()  # clear all axis in figure
plt.ioff()  # close interactive-style
plt.show()
# plt.close()  #  close window

# import pickle
# with open('save/clf.pickle','wb') as f:
#     pickle.dump(clf,f)
# with open('save/clf.pickle','rb') as f:
#     clf2 = pickle.load(f)
#     print(clf2.predict(X[0:1]))
#
# # method2:
# from sklearn.externals import joblib
# joblib.dump(clf,'save/clf.pkl')     #  command + /     cancel

# method 3:
# clf3 = joblib.load('save/clf.pkl')
# print(clf3.predict(X[0:1]))
