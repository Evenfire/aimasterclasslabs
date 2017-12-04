import time
import json

class Saver():
	def __init__(self, session, id, hypers, server=False):
		self.id = id
		self.session = session
		self.hypers = hypers
		self.losses = []
		self.lr = []
		self.acc = []
		self.server = server

	def append_lr(self, lr, timing):
		self.lr.append((lr, timing))

	def append_losses(self, loss, timing):
		self.losses.append((loss, timing))

	def append_acc(self, acc, timing):
		self.acc.append((acc, timing))

	def gen_filename(self):
		date = time.strftime("%d-%m-%Y_%H:%M:%S_%Z")
		prefix = '/output/' if self.server else './output/'
		return "{}{}_{:02d}_{}.saver".format(prefix, self.session, self.id, date)

	def prep_data(self):
		dic = {}
		dic['id'] = self.id
		dic['session'] = self.session
		dic['hypers'] = self.hypers
		dic['losses'] = self.losses
		dic['lr'] = self.lr
		dic['acc'] = self.acc
		return dic

	def export(self):
		filename = self.gen_filename()
		data = self.prep_data()
		print(data)
		with open(filename, 'w') as f:
			json.dump(data, f)

	# def export_model(self, model):

if __name__ == '__main__':
	foo = Saver("bar", 01, ['a', 'b', 'c'])
	for i in range(29):
		foo.append_losses(i/100, i)
	print(foo.gen_filename())
	foo.export()

	with open(foo.gen_filename()) as json_data:
	    d = json.load(json_data)
	    print(type(d))
	    print(d)
	    print(type(d['losses'][0][0]))

