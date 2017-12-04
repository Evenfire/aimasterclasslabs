import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Hppicker(object):
	def __init__(self, arg):
		self.arg = arg
		self.betas = [self.gen_dist_1minus([0.85, 0.999], 1000),
						0.999]#0.85-0.999(#0.990-1)
		self.lr0 = self.gen_dist([0.001 - 0.001 * 0.1, 0.001 + 0.001 * 0.1], 1000)#[self.gen_dist_log([0.0001, 0.01], 1000)]
		self.decay_rate = self.gen_dist_1minus([0.1/100, 30/100], 1000c) #0.1-30
		self.dropouts = []#0.3-0.6
		self.bns = []
		#self.layers


	def to_logint(self, interval):
		res = np.log10([interval[0], interval[1]]).tolist()
		res.sort()
		return (res)

	def to_1minusbeta_log(self, interval):
		res = np.log10([1-interval[0], 1-interval[1]]).tolist()
		res.sort()
		return (res)

	def gen_dist_1minus(self, interval, nbr):
		logs = self.to_1minusbeta_log(interval)
		lst = map(lambda x: 1 + -1 * 10**x, np.random.uniform(logs[0],logs[1],nbr).tolist())
		lst.sort()
		return lst

	def gen_dist_log(self, interval, nbr):
		logs = self.to_logint(interval)
		lst = map(lambda x: 10**x, np.random.uniform(logs[0],logs[1],nbr).tolist())
		lst.sort()
		return lst

	def gen_dist(self, interval, nbr):
		lst = np.random.uniform(interval[0],interval[1],nbr).tolist()
		lst.sort()
		return lst

if __name__ == '__main__':
	# res = np.random.uniform(-3,-0.83,100)
	# res = [1+-1*10**r for r in res]
	# res.sort()
	hp = Hppicker('a')

	plt.figure(1)
	plt.plot(hp.betas[0])
	plt.savefig('picker.png', bbox_inches='tight')

	plt.figure(2)
	plt.plot(hp.decay_rate)
	plt.savefig('picker.png', bbox_inches='tight')

	print(hp.to_1minusbeta_log([0.1/100, 30/100]))
	print(hp.to_logint((0.0001,1)))
	print(hp.to_1minusbeta_log((0.85,0.999)))
