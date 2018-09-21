import sys
import os.path
import argparse
import math
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import config
import data
import model_transformer
import utils
from PIL import Image


def run(net, loader, item):
	""" Run an epoch over the given loader """
	net.train()
	
	mapping = loader.dataset.answer_to_index
	index_to_answer = {mapping[key]:key for key in mapping.keys()}


	coco_id = loader.dataset.coco_ids[item]
	image_path = "/home/user/data/mscoco/images/train2017/" + str(coco_id).zfill(12) + ".jpg"
	if not os.path.isfile(image_path):
		image_path = "/home/user/data/mscoco/images/val2017/" + str(coco_id).zfill(12) + ".jpg"
	image = Image.open(image_path)
	image.show()
	print(image_path)
	question = input("What do you seek?")

	#net.eval()
	  
	v, q, b, idx, q_len = loader.dataset._load_item_demo(item, question)
	
	v = torch.tensor(v)
	q = torch.tensor(q)
	v = Variable(v).cuda(async=True).unsqueeze(0)
	q = Variable(q).cuda(async=True).unsqueeze(0)
	
	#q_len = Variable(q_len).cuda(async=True)

	    #out = net(v, b, q, q_len)
	out = net(v,q).view(3000)

	out = out.data.cpu().numpy()
	max_idxs = np.argsort(out)


	print(index_to_answer[max_idxs[-1]])
	print(index_to_answer[max_idxs[-2]])
	print(index_to_answer[max_idxs[-3]])
	
	
        
  
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('name', nargs='*')
	parser.add_argument('--eval', dest='eval_only', action='store_true')
	parser.add_argument('--test', action='store_true')
	parser.add_argument('--resume', nargs='*')
	args = parser.parse_args()

	logs = torch.load("logs/2018-09-21_12:08:23.pth")
	# hacky way to tell the VQA classes that they should use the vocab without passing more params around
	data.preloaded_vocab = logs['vocab']

	cudnn.benchmark = True

	if not args.eval_only:
	    train_loader = data.get_loader(train=True)
	if not args.test:
	    val_loader = data.get_loader(val=True)
	else:
	    val_loader = data.get_loader(test=True)

	net = model_transformer.make_model(val_loader.dataset.num_tokens, 3000).cuda()

	#net = model.Net(val_loader.dataset.num_tokens).cuda()

	net.load_state_dict(logs['weights'])


	while(True):
		try:
			idx = int(input("Enter image idx: "))
		except:
			print("Invalid!")
			idx = int(input("Enter image idx: "))
		r = run(net, val_loader, idx)


if __name__ == '__main__':
    main()
