import time
import json

class Saver():
	def __init__(self, session, id, hypers, server=False):
		self.id = id
		self.session = session
		self.hypers = hypers
		self.losses = []
		self.lr = []
		self.server = server
		#model

	def append_lr(self, lr, timing):
		self.lr.append((lr, timing))

	def append_losses(self, loss, timing):
		self.losses.append((loss, timing))

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
		return dic

	def export(self):
		filename = self.gen_filename()
		data = self.prep_data()
		print(data)
		with open(filename, 'w') as f:
			json.dump(data, f)


if __name__ == '__main__':
	toto = Saver("gomt", 01, ['a', 'b', 'c'])
	for i in range(29):
		toto.append_losses(i/100, i)
	print(toto.gen_filename())
	toto.export()

	with open(toto.gen_filename()) as json_data:
	    d = json.load(json_data)
	    print(type(d))
	    print(d)
	    print(type(d['losses'][0][0]))