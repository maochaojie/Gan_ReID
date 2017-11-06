import solve
import tensorflow as tf
import sys

train_list = './data/cuhk03/train.list'
val_list = './data/cuhk03/test.list'
train_batch_size = 100
val_batch_size = 50
learning_rate = 0.001
beta1 = 0.5
epoch = 24
model_path = './tfmodel/VGG_ILSVRC_16_layers.npy'
input_height = 160 
input_width = 80
resize_height = 160
resize_width = 80
crop = False
grayscale = False
sp = ' '
img_root = './data/cuhk03/data/'
checkpoint_dir = './checkpoint/'
model_prefix = './tfmodel_snapshot/cuhk03/'
model_name = 'cuhk03_siamse'
debug = True
run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth=True

phase = sys.argv[1]


with tf.Session(config=run_config) as sess:
	solve_ = solve.solver(sess, model_name, phase, train_list, val_list, train_batch_size, val_batch_size, \
		learning_rate, beta1,epoch, \
		model_path = model_path, input_height = input_height, input_width = input_width, resize_height = resize_height, \
		resize_width = resize_width, crop = crop, grayscale = grayscale, sp = sp, img_root = img_root, \
 checkpoint_dir=checkpoint_dir, model_prefix=model_prefix, debug = debug)
	if phase == 'train':
		solve_.train()
	elif phase == 'test':
		solve_.test()
