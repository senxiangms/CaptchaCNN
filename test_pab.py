# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import sys
import os

import random, math, time, string, os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# number
number = []
number.extend(string.ascii_lowercase)
number.extend(string.ascii_uppercase)
number.extend(string.digits)

# 图像大小
IMAGE_HEIGHT = 36
IMAGE_WIDTH = 80
MAX_CAPTCHA = 4

char_set = number
CHAR_SET_LEN = len(char_set)  #

image_filename_list = []

valid_path = './validationData'
model_path = "./model/PAB/PAB.ckpt-6100"

def get_image_file_name(imgFilePath):
	files = get_image_file_name_(imgFilePath)
	random.seed(time.time())
	random.shuffle(files)
	return files, len(files)
    
def get_image_file_name_(imgFilePath):
	fileName = []
	
	for filePath in os.listdir(imgFilePath):
		if os.path.isdir(os.path.join(imgFilePath, filePath)):
			fullsubdir = os.path.join(imgFilePath, filePath)
			subfiles = get_image_file_name_(fullsubdir)
			#print(fullsubdir + "\tD")
			for subfile in subfiles:
				fileName.append(subfile)
		else:
			filename = os.path.join(imgFilePath, filePath)
			if filename.endswith(".png"):
			    fileName.append(filename)
			#print(filePath +"\tF")
	return fileName

    # 向量转回文本
def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_at_pos = i  # c/63
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)

X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
# 声明dropout占位符变量
keep_prob = tf.placeholder(tf.float32)  # dropout

def gen_captcha_text_and_image(imageFilePath, image_filename_list, i):
    p = image_filename_list[i]
    #print("p:" + p)
    #os.path.join(imageFilePath, image_filename_list[num])
    #if p.endswith(".jpg"): # 平安的 gif -> png
    #    processImage(p)     
    #p=p.replace(".jpg", ".png")
    
    img = cv2.imread(p, 0)
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    img = np.float32(img)
    parts = p.split('.')
    text = parts[len(parts) - 2]
    #print("text" + text)
    text = text.replace('\\', '/')
    parts = text.split('/')
    
    #print(parts)
    text = parts[len(parts) - 1]
    if len(text) > MAX_CAPTCHA:
        print (text)
        print (parts)
    return text, img

def get_next_batch(imageFilePath, list_p, image_filename_list=None,  batch_size=128):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])
    #global list_pointer
    i = 0
    while (i < batch_size):
        if list_p[0] >= len(image_filename_list):
            print("all picture done : ", list_p[0])
            list_p[0] = 0
        if list_p[0] % 1000 == 0:
            print("--> : ", list_p[0])
        text, image = gen_captcha_text_and_image(imageFilePath, image_filename_list, list_p[0])
        #print("get_next_batch: ....")
        #print(text)
        #print(image.shape)
        list_p[0] += 1
        if image.shape == (IMAGE_HEIGHT, IMAGE_WIDTH):
            batch_x[i, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
            batch_y[i, :] = text2vec(text)
            i += 1
 
    return batch_x, batch_y

# 定义CNN
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
 
    # 第一层卷积层
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    # 图片和卷积核卷积 结果28x28x32
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)
    # 原图像HEIGHT = 60 WIDTH = 160，经过第一层卷积（图像尺寸不变、特征×32）、池化（图像尺寸缩小一半，特征不变）之后;
    # 输出大小为 30*80*32
 
    # 第二层卷积层
    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)
    # 原图像HEIGHT = 60 WIDTH = 160，经过第一层后输出大小为 30*80*32
    # 经过第二层运算后输出为 16*40*64 (30*80的图像经过2*2的卷积核池化，padding为SAME，输出维度是16*40)
 
    # 第三层卷积层
    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)
    # 原图像HEIGHT = 60 WIDTH = 160，经过神经网络第一层后输出大小为 30*80*32 经过第二层后输出为 16*40*64
    # 经过第二层运算后输出为 16*40*64 ; 经过第三层输出为 8*20*64
    # 24 80 ， 12 40 ， 6 20 ， 3 10
    x = math.ceil(IMAGE_WIDTH / 2 / 2 / 2)
    y = math.ceil(IMAGE_HEIGHT / 2 / 2 / 2)
    w_d = tf.Variable(w_alpha * tf.random_normal([x * y * 64, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)
    # w_out定义成一个形状为 [1024, 8 * 10] = [1024, 80]
    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    # out 的输出为 8*10 的向量， 8代表识别结果的位数，10是每一位上可能的结果（0到9）
    out = tf.add(tf.matmul(dense, w_out), b_out)
    # out = tf.nn.softmax(out)
    # 输出在当前参数下的预测值
    return out

def predict_captcha(captcha_image):
    output = crack_captcha_cnn()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver.restore(sess, model_path)
        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})
        print(text_list)
        #text = text_list[0].tolist()
        #vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
        #i = 0
        #for n in text:
            #vector[i * CHAR_SET_LEN + n] = 1
            #i += 1
        #return vec2text(vector)

def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        print(text)
        raise ValueError('验证码最长4个字符')
 
    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
 
    def char2pos(c):
        if c == '_':
            k = 62
            return k
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k
 
    for i, c in enumerate(text):
        #print(i, c)
        idx = i * CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    return vector


vlist_pointer = [0]
image_filename_list_valid, vtotal = get_image_file_name(valid_path)
print("validate total", vtotal)
#batch_x_test, batch_y_test = get_next_batch(valid_path, vlist_pointer, image_filename_list_valid, 1)
print(image_filename_list_valid[0])
img = cv2.imread(image_filename_list_valid[0], 0)
img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
img = np.float32(img)
input_image = img.flatten() / 255;
predict_captcha(input_image)

