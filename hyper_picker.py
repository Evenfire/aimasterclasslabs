
class Hppicker(object):
	def __init__(self, arg):
		self.arg = arg
		self.betas = []#0.85-0.95#0.990-1
		self.lr0 = []
		self.decay_rate = []#0.1-30
		self.dropouts = []#0.3-0.6
		self.bns = []
		#self.layers

