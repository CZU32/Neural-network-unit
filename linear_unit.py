'''
线性单元
激活函数：线性函数
随机梯度下降算法SGD

傅远昌 fyc@bupt.edu.cn
2017-09-11
'''
from perceptron import Perceptron


#定义激活函数
f = lambda x: x

class LinearUnit(Perceptron):
	def __init__ (self, input_num):
		'''
		初始化线性单元，设置输入参数的个数
		'''
		Perceptron.__init__(self, input_num, f)

def get_training_dataset():
	'''
	初始化数据进行训练: 工作年限对应月薪
	'''
	input_vec = [[5], [3], [8], [1.4], [10.1]]
	label = [5500, 2300, 7600, 1800, 11400]
	return input_vec, label

def train_linear_unit():
	#创建感知器，输入参数的特征数为1（工作年限）,迭代训练
	lu = LinearUnit(1)
	input_vec, label = get_training_dataset()
	lu.train(input_vec, label, 10, 0.01)
	return lu

if __name__ == '__main__':
	#训练线性单元
	linear_unit = train_linear_unit()
	print(linear_unit)	#打印权重
	#测试
	print('Work 3.4 years, monthly salary = %.2f'%linear_unit.predict([3.4]))
	print('Work 15 years, monthly salary = %.2f'%linear_unit.predict([15]))
	print('Work 1.5 years, monthly salary = %.2f'%linear_unit.predict([1.5]))
	print('Work 6.3 years, monthly salary = %.2f'%linear_unit.predict([6.3]))