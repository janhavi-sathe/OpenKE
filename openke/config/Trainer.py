# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
import copy
from tqdm import tqdm
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

class Trainer(object):

	def __init__(self, 
				 model = None,
				 data_loader = None,
				 train_times = 30,
				 alpha = 0.5,
				 use_gpu = True,
				 opt_method = "sgd",
				 save_steps = None,
				 checkpoint_dir = None):

		self.work_threads = 8
		self.train_times = train_times

		self.opt_method = opt_method
		self.optimizer = None
		self.lr_decay = 0
		self.weight_decay = 0
		self.alpha = alpha

		self.model = model
		self.data_loader = data_loader
		self.use_gpu = use_gpu
		self.save_steps = save_steps
		self.checkpoint_dir = checkpoint_dir

	def train_one_step(self, data):
		self.optimizer.zero_grad()
		loss = self.model({
			'batch_h': self.to_var(data['batch_h'], self.use_gpu),
			'batch_t': self.to_var(data['batch_t'], self.use_gpu),
			'batch_r': self.to_var(data['batch_r'], self.use_gpu),
			'batch_y': self.to_var(data['batch_y'], self.use_gpu),
			'mode': data['mode']
		})
		loss.backward()
		self.optimizer.step()		 
		return loss.item()

	def run(self):
		if self.use_gpu:
			self.model.cuda()

		if self.optimizer != None:
			pass
		elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
			self.optimizer = optim.Adagrad(
				self.model.parameters(),
				lr=self.alpha,
				lr_decay=self.lr_decay,
				weight_decay=self.weight_decay,
			)
		elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
			self.optimizer = optim.Adadelta(
				self.model.parameters(),
				lr=self.alpha,
				weight_decay=self.weight_decay,
			)
		elif self.opt_method == "Adam" or self.opt_method == "adam":
			self.optimizer = optim.Adam(
				self.model.parameters(),
				lr=self.alpha,
				weight_decay=self.weight_decay,
			)
		else:
			self.optimizer = optim.SGD(
				self.model.parameters(),
				lr = self.alpha,
				weight_decay=self.weight_decay,
			)
		print("Finish initializing...")
		
		training_range = tqdm(range(self.train_times), desc="Epoch Progress")
		epoch_losses = []
		for epoch in training_range:
			res = 0.0
			batch_loss_data = []
			batch_range = tqdm(self.data_loader, desc="Batch Progress", leave=False)
			# for data in self.data_loader:
			# 	loss = self.train_one_step(data)
			# 	res += loss
			# training_range.set_description("Epoch %d | loss: %f" % (epoch, res))
			for data in batch_range:
				loss = self.train_one_step(data)
				res += loss
				batch_loss_data.append(loss)
				batch_range.set_postfix(loss=loss)
			
			avg_epoch_loss = res / len(self.data_loader)
			epoch_losses.append(avg_epoch_loss)
			training_range.set_description(f"Epoch {epoch} | Average loss: {avg_epoch_loss:.4f}")
			self.plot_loss_trend(batch_loss_data, epoch)


			if self.save_steps and self.checkpoint_dir and (epoch + 1) % self.save_steps == 0:
				print("Epoch %d has finished, saving..." % (epoch))
				self.model.save_checkpoint(os.path.join(self.checkpoint_dir + "-" + str(epoch) + ".ckpt"))

		self.plot_epoch_losses(epoch_losses)
	
	def plot_epoch_losses(self, losses):
		plt.figure()
		plt.plot(losses, marker='o', linestyle='-')
		plt.title('Average Loss Per Epoch')
		plt.xlabel('Epoch')
		plt.ylabel('Average Loss')
		plt.grid(True)
		plt.show()
	
	def plot_loss_trend(self, losses, epoch):
		plt.figure()
		plt.plot(losses)
		plt.title(f'Loss Trend for Epoch {epoch}')
		plt.xlabel('Batch Number')
		plt.ylabel('Loss')
		plt.show()
	
	def set_model(self, model):
		self.model = model

	def to_var(self, x, use_gpu):
		if use_gpu:
			return Variable(torch.from_numpy(x).cuda())
		else:
			return Variable(torch.from_numpy(x))

	def set_use_gpu(self, use_gpu):
		self.use_gpu = use_gpu

	def set_alpha(self, alpha):
		self.alpha = alpha

	def set_lr_decay(self, lr_decay):
		self.lr_decay = lr_decay

	def set_weight_decay(self, weight_decay):
		self.weight_decay = weight_decay

	def set_opt_method(self, opt_method):
		self.opt_method = opt_method

	def set_train_times(self, train_times):
		self.train_times = train_times

	def set_save_steps(self, save_steps, checkpoint_dir = None):
		self.save_steps = save_steps
		if not self.checkpoint_dir:
			self.set_checkpoint_dir(checkpoint_dir)

	def set_checkpoint_dir(self, checkpoint_dir):
		self.checkpoint_dir = checkpoint_dir