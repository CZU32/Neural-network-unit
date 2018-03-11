'''
循环神经网络的基本实现

傅远昌 fyc@bupt.edu.cn
2017-10-11
'''

import numpy as np


class ReluActivator(object):
	'''激活函数Relu'''
	def forward(self, weighted_input):
		#实现前向计算
		return max(0, weighted_input)

	def backward(self, output):
		#计算导数
		return 1 if output > 0 else 0


def element_wise_op(array, op):
	#实现对numpy数组进行安元素操作，并将返回值写回到数组中
	for i in np.nditer(array, op_flags = ['readwrite']):
		i[...] = op(i)


class RecurrentLayer(object):
	def __init__(self, input_width, state_width, activator, learning_rate):
		self.input_width = input_width
		self.state_width = state_width
		self.activator = activator
		self.learning_rate = learning_rate

		self.time = 0
		self.state_list = []
		self.state_list.append(np.zeros(state_width, 1))	#初始化s0
		self.U = np.random.uniform(-1e-4, 1e-4,  (state_width, input_width))	#初始化U
		self.W = np.random.uniform(-1e-4, 1e-4,  (state_width, input_width))	#初始化W

	def forward(self, input_array):
		self.time += 1
		state = (np.dot(self.U, input_array) + np.dot(self.W, self.state_list[-1]))
		element_wise_op(state, self.activator.forword)
		self.state_list.append(state)

	def backward(self, sensitivity_array, activator):
		'''
		实现BPTT算法
		'''
		self.calc_delta(sensitivity_array, activator)
		self.calc_gradient()

	def calc_delta(self, sensitivity_array, activator):
		self.delta_list = []	#用来保存各个时刻的数据项
		for i in range(self.times):
			self.delta_list.append(np.zeros(self.state_width, 1))
		self.delta_list.append(sensitivity_array)
		#迭代计算每个时刻的误差项
		for k in range(self.times - 1, 0, -1):
			self.calc_delta_k(k, activator)

	def calc_delta_k(self, k, activator):
		'''
		根据k+1时刻的delta值计算k时刻的delta值
		'''
		state = self.state_list[k+1].copy()
		element_wise_op(self.state_list[k+1], activator.backward)
		self.delta_list[k] = np.dot(np.dot(self.delta_list[k+1].T, self.W), np.diag(state[:,0])).T 

	def calc_gradient(self):
		self.gratient_list = []		#保存各个时刻的权重梯度
		for t in range(self.tiems + 1):
			self.gratient_list.append(np.zeros((self.state_width, self.state_width)))
		for t in range(self.times, 0, -1):
			self.calc_gradient_t(t)
		#实际梯度是各时刻梯度之和
		self.gratient =  reduce(lambda a, b: a + b, self.gratient_list, self.gratient_list[0])

	def calc_gradient_t(self, t):
		#计算每个时刻t权重的梯度
		gratient = np.dot(self.delta_list[t], self.state_list[t-1].T)
		self.gratient_list[t] = gratient

	def update(self):
		'''
		按照梯度下降，跟新权重
		'''
		self.W -= self.learning_rate * self.gradient

		#缺U的更新

	def reset_state(self):
		#重置循环层的内部状态
		self.times = 0
		self.state_list = []
		self.state_list.append(np.zeros((self.state_width, 1)))


def gradient_check():
	'''
	梯度检查
	'''
	#设计一个误差函数，取所有节点输出项之和
	error_function = lambda o: o.sum()

	rl = RecurrentLayer(3, 2, IdentityActivator(), 1e-3)

	#计算forward值
	x, d = data_set()
	rl.forward(x[0])
	rl.forward(x[1])

	sensitivity_array = np.ones(rl.state_list[-1].shape, dtype = np.float64)

	rl.backward(sensitivity_array, IdentityActivator())

	#检查梯度
	epsilon = 10e-4
	for i in range(rl.W.shape[0]):
		for  j in range(rl.W.shape[1]):
			rl.W[i, j] += epsilon
			rl.reset_state()
			rl.forward(x[0])
			rl.forward(x[1])
			err1 = error_function(rl.state_list[-1])

			rl.W[i, j] -= 2 * epsilon
			rl.reset_state()
			rl.forward(x[0])
			rl.forward(x[1])
			err2 = error_function(rl.state_list[-1])

			expect_grad = (err1 - err2) / (2 * epsilon)
			rl.W[i, j] += epsilon
			print('weights(%d,%d):expected - actural %f -%f'%(i, j ,expect_grad, rl.gratient[i, j]))


