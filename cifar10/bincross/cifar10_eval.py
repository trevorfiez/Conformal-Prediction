# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/scratch/tfiez/conformal/cifar10/bincross/weight_100_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/scratch/tfiez/conformal/cifar10/bincross/equal',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")


def eval_once(saver, summary_writer, top_k_op, summary_op):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

def classes_predicted(preds):

	subbed = preds - 0.5

	
def find_all_threshold(saver, summary_writer, summary_op, logits, labels, epsilon):
	threshold = 0.0
	with tf.Session() as sess:
		ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			# Restores from checkpoint
			saver.restore(sess, ckpt.model_checkpoint_path)
			# Assuming model_checkpoint_path looks something like:
			#   /my-favorite-path/cifar10_train/model.ckpt-0,
			# extract global_step from it.
			global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
		coord = tf.train.Coordinator()	

		labels_dense = tf.one_hot(indices=labels, depth=10)

		#preds_sparse = tf.multiply(logits, labels_dense)

		#gt_max = tf.reduce_max(preds_sparse, axis=1)

		try:	
			
			threads = []
			for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
				threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
							 start=True))
			num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
		
			step = 0
			all_gt_scores = None
			while step < num_iter and not coord.should_stop():
				scores, batch_labels = sess.run([logits, labels_dense])

				pos_scores = []
				for i in range(scores.shape[0]):
					pos_scores.append(scores[i, np.argmax(batch_labels[i,:])])

				if all_gt_scores is not None:
					all_gt_scores = np.hstack((all_gt_scores, np.array(pos_scores)))
					#all_gt_scores = all_gt_scores + scores
				else:
					
					all_gt_scores = np.array(pos_scores)
				step += 1

			sort_arr = np.sort(all_gt_scores)
			max_index = int(sort_arr.shape[0] * (1 - epsilon))
			print(sort_arr[0:100])
			print("%d %d" % (num_iter, len(all_gt_scores)))

			threshold = sort_arr[max_index]
					
		except Exception as e:  # pylint: disable=broad-except
			coord.request_stop(e)
		coord.request_stop()
		coord.join(threads, stop_grace_period_secs=10)

	print("got eval %0.3f" % (threshold))

	return threshold

def find_class_threshold(saver, summary_writer, summary_op, logits, labels, epsilon):
	class_thresholds = []
	with tf.Session() as sess:
		ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			# Restores from checkpoint
			saver.restore(sess, ckpt.model_checkpoint_path)
			# Assuming model_checkpoint_path looks something like:
			#   /my-favorite-path/cifar10_train/model.ckpt-0,
			# extract global_step from it.
			global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
		coord = tf.train.Coordinator()	

		labels_dense = tf.one_hot(indices=labels, depth=10)

		preds_sparse = tf.multiply(logits, labels_dense)

		#gt_max = tf.reduce_max(preds_sparse, axis=1)
		
		try:	
			
			threads = []
			for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
				threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
							 start=True))
			num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
		
			step = 0
			all_gt_scores = None
			all_labels = None
			while step < num_iter and not coord.should_stop():
				scores, batch_labels = sess.run([preds_sparse, labels_dense])
				if all_gt_scores is not None:
					all_gt_scores = np.vstack((all_gt_scores, scores))
					all_labels = np.vstack((all_labels, batch_labels))
					#all_gt_scores = all_gt_scores + scores
				else:
					#print(scores)
					print(type(scores))
					all_labels = batch_labels
					all_gt_scores = scores
				step += 1
			classes = [[] for x in range(10)]
			print(all_gt_scores.shape)
			for i in xrange(all_gt_scores.shape[0]):
				max_in = np.argmax(all_labels[i,:])
				if (max_in > len(classes)):
					print(max_in)
				classes[max_in].append(all_gt_scores[i, max_in])
			
			#class_thresholds = []
			for c in classes:
				c.sort()
				t_in = int(len(c) * (1 - epsilon))
				class_thresholds.append(c[t_in])			
				
					
		except Exception as e:  # pylint: disable=broad-except
			coord.request_stop(e)
		coord.request_stop()
		coord.join(threads, stop_grace_period_secs=10)

	print("Class thresholds")
	print(class_thresholds)

	return class_thresholds	
			
		

def conformal_direct_eval(saver, summary_writer, summary_op, logits, labels, threshold):
	recall = 0.0
	precision = 0.0
	with tf.Session() as sess:
		ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			# Restores from checkpoint
			saver.restore(sess, ckpt.model_checkpoint_path)
			# Assuming model_checkpoint_path looks something like:
			#   /my-favorite-path/cifar10_train/model.ckpt-0,
			# extract global_step from it.
			global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
		coord = tf.train.Coordinator()		
		
		thresh = tf.constant(threshold, shape=logits.get_shape())
		labels_dense = tf.one_hot(indices=labels, depth=10)
		preds_sparse = tf.multiply(logits, labels_dense)

		num_correct = tf.greater_equal(preds_sparse, thresh)
		
		num_predicted = tf.greater_equal(logits, thresh)		
		
		try:
			threads = []
			for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
				threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
							 start=True))

			num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
			true_count = 0  # Counts the number of correct predictions.
			total_predicted = 0
			total_sample_count = num_iter * FLAGS.batch_size
			step = 0
			while step < num_iter and not coord.should_stop():
				predictions, batch_labels = sess.run([num_predicted, labels_dense])
				for i in range(batch_labels.shape[0]):
					if (predictions[i, np.argmax(batch_labels[i,:])]):
						true_count += 1
				
				total_predicted += np.sum(predictions)
				step += 1

		
			recall = true_count / total_sample_count
			precision = true_count / (total_predicted)
			print('Recall = %.3f' % (recall))
			print('Precision = %.3f' % (precision))

			summary = tf.Summary()
			summary.ParseFromString(sess.run(summary_op))
			summary.value.add(tag='Recall', simple_value=recall)
			summary.value.add(tag='Precision', simple_value=precision) 
			summary_writer.add_summary(summary, global_step)
		except Exception as e:  # pylint: disable=broad-except
			coord.request_stop(e)
		coord.request_stop()
		coord.join(threads, stop_grace_period_secs=10)

	return recall, precision

def conformal_class_eval(saver, summary_writer, summary_op, logits, labels, class_thresh):
	recall = 0.0
	precision = 0.0
	with tf.Session() as sess:
		ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			# Restores from checkpoint
			saver.restore(sess, ckpt.model_checkpoint_path)
			# Assuming model_checkpoint_path looks something like:
			#   /my-favorite-path/cifar10_train/model.ckpt-0,
			# extract global_step from it.
			global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
		coord = tf.train.Coordinator()
		
		print(logits.get_shape())
		temp_thresh = np.ones(shape=logits.get_shape(), dtype="float32")
		temp_thresh = temp_thresh * np.array(class_thresh)
		thresh = tf.constant(temp_thresh)
		labels_dense = tf.one_hot(indices=labels, depth=10)
		
		num_predicted = tf.greater_equal(logits, thresh)
		#num_correct = tf.multiply(num_predicted, labels_dense)		
		
		try:
			threads = []
			for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
				threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
							 start=True))

			num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
			true_count = 0  # Counts the number of correct predictions.
			total_predicted = 0
			total_sample_count = num_iter * FLAGS.batch_size
			step = 0
			while step < num_iter and not coord.should_stop():
				predictions, batch_labels = sess.run([num_predicted, labels_dense])
				for i in range(batch_labels.shape[0]):
					if (predictions[i, np.argmax(batch_labels[i,:])]):
						true_count += 1
				
				total_predicted += np.sum(predictions)
				step += 1
				
		
			recall = true_count / total_sample_count
			precision = true_count / (total_predicted)
			print('Recall = %.3f' % (recall))
			print('Precision = %.3f' % (precision))

			summary = tf.Summary()
			summary.ParseFromString(sess.run(summary_op))
			summary.value.add(tag='Recall', simple_value=recall)
			summary.value.add(tag='Precision', simple_value=precision) 
			summary_writer.add_summary(summary, global_step)
		except Exception as e:  # pylint: disable=broad-except
			coord.request_stop(e)
		coord.request_stop()
		coord.join(threads, stop_grace_period_secs=10)

	return recall, precision		
		

def evaluate_all_threshold(epsilon):		
  thresh = 0.5
  
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.

    validation_data = FLAGS.eval_data == "not test"
    valid_images, labels = cifar10.inputs(eval_data=validation_data)

    valid_scores = cifar10.inference(valid_images)

    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    thresh = find_all_threshold(saver, summary_writer, summary_op, valid_scores, labels, epsilon)
  
  #thresh = 0.5
  with tf.Graph().as_default() as g:
    eval_data = FLAGS.eval_data == "test"
    images, labels = cifar10.inputs(eval_data=eval_data)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images=images)

    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    # Calculate predictions.
    #top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Restore the moving average version of the learned variables for eval.
    
    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    #while :
      #eval_once(saver, summary_writer, top_k_op, summary_op)
    recall, precision = conformal_direct_eval(saver, summary_writer, summary_op, logits, labels, thresh)
    return recall, precision

def evaluate_class_threshold(epsilon):
  class_thresh = []
  
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.

    validation_data = FLAGS.eval_data == "not test"
    valid_images, labels = cifar10.inputs(eval_data=validation_data)

    valid_scores = cifar10.inference(valid_images)

    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    class_thresh = find_class_threshold(saver, summary_writer, summary_op, valid_scores, labels, epsilon)

  with tf.Graph().as_default() as g:
    eval_data = FLAGS.eval_data == "test"
    images, labels = cifar10.inputs(eval_data=eval_data)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images=images)

    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    # Calculate predictions.
    #top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Restore the moving average version of the learned variables for eval.
    
    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    #while :
      #eval_once(saver, summary_writer, top_k_op, summary_op)
    recall, precision = conformal_class_eval(saver, summary_writer, summary_op, logits, labels, class_thresh)
    return recall, precision

def conformal_diff(logits):
	values, indices = tf.nn.top_k(logits, 1)
	my_range = tf.expand_dims(tf.range(0, indices.get_shape()[0]), 1)
	full_indices = tf.concat([my_range, indices], axis=1)
	only_max = tf.sparse_to_dense(full_indices, logits.get_shape(), tf.reshape(values, [-1]), default_value=0.)
	no_max = logits - only_max
	max_vals = tf.reduce_max(only_max, axis=1, keep_dims=True)
	r = no_max - max_vals + only_max
	second_max = tf.reduce_max(no_max, axis=1, keep_dims=True)
	o_r = tf.nn.relu(only_max - second_max)
	final_result = o_r + r
	return final_result


def conformal_ratio(logits):
	values, indices = tf.nn.top_k(logits, 1)
	my_range = tf.expand_dims(tf.range(0, indices.get_shape()[0]), 1)
	full_indices = tf.concat([my_range, indices], axis=1)
	only_max = tf.sparse_to_dense(full_indices, logits.get_shape(), tf.reshape(values, [-1]), default_value=0.)
	only_max_ones = tf.sparse_to_dense(full_indices, logits.get_shape(), 1.0, default_value=0.)
	non_max_ones = tf.constant(np.ones(shape=logits.get_shape(), dtype="float32")) - only_max_ones
	no_max = logits - only_max
        non_max_ones_t = tf.transpose(non_max_ones)
	numerator_max = tf.transpose(non_max_ones_t * tf.reduce_max(only_max, axis=1))
	numerator_non_max = tf.transpose(tf.transpose(only_max_ones) * tf.reduce_max(no_max, axis=1))
	numerator = numerator_max + numerator_non_max
	ratio = tf.negative(tf.div(numerator, logits))
	return ratio

	

def evaluate_diff_threshold(epsilon, conf_method="diff"):
  class_thresh = []
  
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.

    validation_data = FLAGS.eval_data == "not test"
    valid_images, labels = cifar10.inputs(eval_data=validation_data)

    valid_scores = cifar10.inference(valid_images)

    if (conf_method== "diff"):
      valid_conf = conformal_diff(valid_scores)
    else:
      valid_conf = conformal_ratio(valid_scores)

    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    class_thresh = find_all_threshold(saver, summary_writer, summary_op, valid_conf, labels, epsilon)

  with tf.Graph().as_default() as g:
    eval_data = FLAGS.eval_data == "test"
    images, labels = cifar10.inputs(eval_data=eval_data)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images=images)
    
    if (conf_method == "diff"):
      conf_scores = conformal_diff(logits)
    else:
      conf_scores = conformal_ratio(logits)
    
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    # Calculate predictions.
    #top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Restore the moving average version of the learned variables for eval.
    
    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    #while :
      #eval_once(saver, summary_writer, top_k_op, summary_op)
    recall, precision = conformal_direct_eval(saver, summary_writer, summary_op, conf_scores, labels, class_thresh)
    return recall, precision


def evaluate(conformal_thresholds):
  """Eval CIFAR-10 for a number of steps."""
  
  results = []
  for thresh in conformal_thresholds:
  	results.append((evaluate_all_threshold(thresh)))

  for i in range(len(conformal_thresholds)):
    print("Recall threshold: %.2f, Recall: %.3f, Precision: %.3f" % (conformal_thresholds[i], results[i][0], results[i][1]))
  
  return results

def class_thresh_evaluate(conformal_thresholds):
  """Eval CIFAR-10 for a number of steps."""
  
  results = []
  for thresh in conformal_thresholds:
  	results.append((evaluate_class_threshold(thresh)))

  for i in range(len(conformal_thresholds)):
    print("Recall threshold: %.2f, Recall: %.3f, Precision: %.3f" % (conformal_thresholds[i], results[i][0], results[i][1]))
  
  return results

def conformal_diff_evaluate(conformal_thresholds, method="diff"):
  results = []
  for thresh in conformal_thresholds:
  	results.append((evaluate_diff_threshold(thresh, method)))

  for i in range(len(conformal_thresholds)):
    print("Recall threshold: %.2f, Recall: %.3f, Precision: %.3f" % (conformal_thresholds[i], results[i][0], results[i][1]))
  
  return results


def all_thresh_main():
  cifar10.maybe_download_and_extract()
  conformal_thresholds = [0.848, 0.941, 0.970, 0.985, 0.993, 0.996, 0.998]

  directory_bases = ['100', '075', '05', '025', '005', '001']
  all_results = []
  for base in directory_bases:
    FLAGS.eval_dir = '/scratch/tfiez/conformal/cifar10/bincross/weight_' + base + '_eval'
    FLAGS.checkpoint_dir = '/scratch/tfiez/conformal/cifar10/bincross/weight_' + base
    if (base == '100'):
      FLAGS.checkpoint_dir = '/scratch/tfiez/conformal/cifar10/bincross/equal'

    print(FLAGS.eval_dir)
    print(FLAGS.checkpoint_dir)
    
    if tf.gfile.Exists(FLAGS.eval_dir):
      tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    r = evaluate(conformal_thresholds)
    all_results.append(r)
  for j in range(len(directory_bases)):
    print("------  " + directory_bases[j] + '  -------')
    results = all_results[j]
    for i in range(len(conformal_thresholds)):
      print("Recall threshold: %.2f, Recall: %.3f, Precision: %.3f" % (conformal_thresholds[i], results[i][0], results[i][1]))
  
def class_thresh_main():
  cifar10.maybe_download_and_extract()
  conformal_thresholds = [0.848, 0.941, 0.970, 0.985, 0.993, 0.996, 0.998]

  directory_bases = ['100', '075', '05', '025', '005', '001']
  all_results = []
  for base in directory_bases:
    FLAGS.eval_dir = '/scratch/tfiez/conformal/cifar10/bincross/weight_' + base + '_eval'
    FLAGS.checkpoint_dir = '/scratch/tfiez/conformal/cifar10/bincross/weight_' + base
    if (base == '100'):
      FLAGS.checkpoint_dir = '/scratch/tfiez/conformal/cifar10/bincross/equal'

    print(FLAGS.eval_dir)
    print(FLAGS.checkpoint_dir)
    
    if tf.gfile.Exists(FLAGS.eval_dir):
      tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    r = class_thresh_evaluate(conformal_thresholds)
    all_results.append(r)
  
  for j in range(len(directory_bases)):
    print("------  " + directory_bases[j] + '  -------')
    results = all_results[j]
    for i in range(len(conformal_thresholds)):
      print("Recall threshold: %.3f, Recall: %.3f, Precision: %.3f" % (conformal_thresholds[i], results[i][0], results[i][1]))

def conformal_diff_main():
  cifar10.maybe_download_and_extract()
  conformal_thresholds = [0.848, 0.941, 0.970, 0.985, 0.993, 0.996, 0.998]

  directory_bases = ['100', '075', '05', '025', '005', '001']
  all_results = []
  for base in directory_bases:
    FLAGS.eval_dir = '/scratch/tfiez/conformal/cifar10/bincross/weight_' + base + '_eval'
    FLAGS.checkpoint_dir = '/scratch/tfiez/conformal/cifar10/bincross/weight_' + base
    if (base == '100'):
      FLAGS.checkpoint_dir = '/scratch/tfiez/conformal/cifar10/bincross/equal'

    print(FLAGS.eval_dir)
    print(FLAGS.checkpoint_dir)
    
    if tf.gfile.Exists(FLAGS.eval_dir):
      tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    r = conformal_diff_evaluate(conformal_thresholds)
    all_results.append(r)
  
  for j in range(len(directory_bases)):
    print("------  " + directory_bases[j] + '  -------')
    results = all_results[j]
    for i in range(len(conformal_thresholds)):
      print("Recall threshold: %.3f, Recall: %.3f, Precision: %.3f" % (conformal_thresholds[i], results[i][0], results[i][1]))

def conformal_ratio_main():
  cifar10.maybe_download_and_extract()
  conformal_thresholds = [0.848, 0.941, 0.970, 0.985, 0.993, 0.996, 0.998]

  directory_bases = ['100', '075', '05', '025', '005', '001']
  all_results = []
  for base in directory_bases:
    FLAGS.eval_dir = '/scratch/tfiez/conformal/cifar10/bincross/weight_' + base + '_eval'
    FLAGS.checkpoint_dir = '/scratch/tfiez/conformal/cifar10/bincross/weight_' + base
    if (base == '100'):
      FLAGS.checkpoint_dir = '/scratch/tfiez/conformal/cifar10/bincross/equal'

    print(FLAGS.eval_dir)
    print(FLAGS.checkpoint_dir)
    
    if tf.gfile.Exists(FLAGS.eval_dir):
      tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    r = conformal_diff_evaluate(conformal_thresholds, "ratio")
    all_results.append(r)
  
  for j in range(len(directory_bases)):
    print("------  " + directory_bases[j] + '  -------')
    results = all_results[j]
    for i in range(len(conformal_thresholds)):
      print("Recall threshold: %.3f, Recall: %.3f, Precision: %.3f" % (conformal_thresholds[i], results[i][0], results[i][1]))

  #all_results.append(r)
  
def main(argv=None):  # pylint: disable=unused-argument
  conformal_ratio_main()


if __name__ == '__main__':
  tf.app.run()
