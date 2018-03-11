'''
感知器
激活函数：阶跃函数

傅远昌 fyc@bupt.edu.cn
2017-09-06
'''
from functools import reduce

class Perceptron(object):

	def __init__(self, input_num, activator):
		"""
		初始化感知器，设置输入参数的个数，以及激活函数
		激活函数的类型为double -> double
		"""
		self.activator = activator
		#权重向量初始化为0
		self.weights = [0.0 for _ in range(input_num)]
		#偏置项初始化为0
		self.bias = 0.0

	def __str__ (self): 
		'''
		打印学习的权重、偏置项
		'''
		return "weights\t:%s\nbias\t:%f\n"%(self.weights, self.bias)

	def predict(self, input_vec):
		'''
		输入向量，输出感知器的计算结果
		'''
		prod = [a*b for a, b in zip(input_vec, self.weights)]
		return self.activator(reduce(lambda a, b: a + b,prod, 0.0) + self.bias)

	def train(self, input_vec, label, iteration, rate):
		'''
		输入训练数据：一组向量、与每个向量对应的label：以及训练轮数、学习率
		''' 
		for i in range(iteration):

			self._one_iteration(input_vec, label, rate)

	def _one_iteration(self, input_vec, label, rate):
		'''
		一次迭代，把所有的训练数据过一遍
		'''
		samples = zip(input_vec, label)	#训练样本
		#对于每个样本，按照感知器规则更新权重
		for (input_vec, label) in samples:
			output = self.predict(input_vec)	#计算当前权重下的输出
			self._update_weights(input_vec, output, label, rate)

	def _update_weights(self, input_vec, output, label, rate):
		'''
		按感知器规则更新权重
		'''
		delta = label - output

		self.weights = [b+rate*delta*a for a, b in zip(input_vec, self.weights)]
		#self.weights = map(lambda (x, w): w + rate * delta * x, zip(input_vec, self.weights))

		#更新bias
		self.bias += rate * delta





#用该感知器实现and（按位与）函数

def f(x):
	'''
	定义激活函数
	'''
	return 1 if x > 0 else 0

def get_training_dataset():
	'''
	基于and真值表构建训练数据
	'''
	input_vec = [[1,1], [0,0], [1,0], [0,1]]
	#期望输出
	label = [1, 0, 0, 0]
	return input_vec, label

def train_and_perceptron():
	'''
	使用and真值表训练感知器
	'''
	#创建感知器（参数个数，激活函数）
	p = Perceptron(2, f)
	#训练，迭代10轮，学习速率为0.1
	input_vec, label = get_training_dataset()
	p.train(input_vec, label, 10, 0.1)
	#返回训练好的感知器
	return p

if __name__ == '__main__':
	#训练好的and感知器
	and_perceptron = train_and_perceptron()
	print(and_perceptron)

	#测试
	print('1 and 1 = %d'%and_perceptron.predict([1, 1]))
	print('0 and 0 = %d'%and_perceptron.predict([0, 0]))
	print('1 and 0 = %d'%and_perceptron.predict([1, 0]))
	print('0 and 1 = %d'%and_perceptron.predict([0, 1]))




