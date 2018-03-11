'''
全连接神经网络：
激活函数是sigmoid函数
平方和误差
全连接网络
随机梯度下降优化算法

傅远昌 fyc@bupt.edu.cn
2017-09-18
'''

import random
from numpy import *


#节点类，负责记录和维护节点自身信息以及与这个节点相关的上下游连接，实现输出值和误差项的计算
class Node(object):
	def __init__ (self, layer_index, node_index):
		'''
		构造节点对象
		1.节点所属层的编号
		2.节点的编号
		'''
		self.layer_index = layer_index
		self.node_index = node_index
		self.downstream = []
		self.upstream = []
		self.output = 0
		self.input = 0
		self.delta = 0

	def set_output(self, output):
		self.output = output

	def append_downstream_connection(self, conn):
		#添加一个到下游节点的连接
		self.downstream.append(conn)

	def append_upstream_connection(self, conn):
		#添加一个到上游节点的连接
		self.upstream.append(conn)

	def calculate_output(self):
		#计算节点的输出
		output = reduce(lambda ret, conn: ret + conn.upstream_node.output * conn.weight, self.upstream, 0)
		self.output = sigmoid(output)

	def calculate_hidden_layer_delta(self):
		#计算隐含层的偏差值（delta)
		downstream_delta = reduce(lambda ret, conn: ret + conn.downstream_node.delta * conn.weight, self.downstream, 0.0)
		self.delta = self.output * (1 - self.output) * downstream_delta

	def calculate_output_layer_delta(self, label):
		#计算输出层的偏差值（label:样本输出值）
		self.delta = self.output * (1 - self.output) * (label - self.output)

	def __str__ (self):
		#打印节点信息
		node_str = '%u-%u: output: %f delta: %f'%(self.layer_index, self.node_index, self.output, self.delta)
		downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
		upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
		return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str


#ConstNode对象，为了实现一个输出恒为一的节点（计算偏置项目时需要）
class ConstNode(object):
	def __init__(self, layer_index, node_index):
		self.layer_index = layer_index	#节点所属层的编号
		self.node_index = node_index
		self.downstream = []
		self.output = 1

	def append_downstream_connection(self, conn):
		self.downstream.append(conn)

	def calculate_hidden_layer_delta(self):
		downstream_delta = reduce(lambda ret, conn: ret + conn.downstream_node.delta * conn.weight, self.downstream, 0.0)
		self.delta = self.output * (1 - self.output) * downstream_delta

	def __str__(self):
		node_str = '%u-%u: output: 1' % (self.layer_index, self.node_index)
		downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
		return node_str + '\n\tdownstream:' + downstream_str

#Layer对象，负责初始化一层。此外，作为Node的集合对象，提供对Node集合的操作
class Layer(object):
	def __init__(self, layer_index, node_count):
		#初始化一层：层编号，层所包含的节点数
		self.layer_index = layer_index
		self.nodes = []
		for i in range(node_count):
			self.nodes.append(Node(layer_index, i))
		self.nodes.append(ConstNode(layer_index, node_count))

	def set_output(self, data):
		for i in range(len(data)):
			self.nodes[i].set_output(data[i])

	def calculate_output(self):
		for node in self.nodes[:-1]:
			node.calculate_output()

	def dump(self):
		#打印层信息
		for node in self.nodes:
			print (node)

#Connection对象，主要职责是记录连接的权重，以及这个连接所关联的上下游节点
class Connection(object):
	def __init__(self, upstream_node, downstream_node):
		#初始化连接，权重初始化为是一个很小的随机数
		self.upstream_node = upstream_node
		self.downstream_node = downstream_node
		self.weight = random.uniform(-0.1, 0.1)
		self.gradient = 0.0

	def calcilate_gradient(slf):
		#计算梯度
		self.gradient = self.downstream_node.delta * self.upstream_node.output

	def get_gradient(self):
		return self.gradient

	def update_weight(self, rate):
		#根据梯度下降算法更新权重
		self.calcilate_gradient()
		self.weight += rate * self.gradient

	def __str__(self):
		#打印连接信息
		return '(%u-%u) -> (%u-%u) = %f' %(
			self.upstream_node.layer_index,
			self.upstream_node.node_index,
			self.downstream_node.layer_index,
			self.downstream_node.node_index,
			self.weight)
#Connection对象，提供Connection集合操作
class Connections(object):
	def __init__(self):
		self.connections = []

	def add_connection(self, connection):
		self.connections.append(connection)

	def dump(self):
		for conn in self.connections:
			print (conn)

#Network对象，提供API
class Network(object):
	def __init__(self, layers):
		#初始化一个全连接网络：二维数组（描述神经网络每层节点数）
		self.connections = Connections()
		self.layers = []
		layer_count = len(layers)
		node_count = 0
		for i in range(layer_count):
			self.layers.append(Layer(i, layers[i]))
		for layer in range(layer_count - 1):
			connections = [Connection(upstream_node, downstream_node)
							for upstream_node in self.layers[layer].nodes
							for downstream_node in self.layers[layer + 1].nodes[:-1]]
			for conn in connections:
				self.connections.add_connection(conn)
				conn.downstream_node.append_upstream_connection(conn)
				conn.upstream_node.append_downstream_connection(conn)

	def train(self, labels, data_set, rate, iteration):
		#训练神经网络：数组（训练样本标签，每个元素是一个样本标签），二维数组（训练样本特征，每个元素是一个样本特征）
		for i in range(iteration):
			for d in range(len(data_set)):
				self.train_one_sample(labels[d], data_set[d], rate)

	def train_one_sample(self, label, sample, rate):
		#用一个样本训练网络
		self.predict(sample)
		self.calcilate_delta(label)
		self.update_weight(rate)

	def calculate_delta(self, label):
		#计算每个节点的delta
		output_nodes = self.layers[-1].nodes
		for i in range(len(label)):
			output_nodes[i].calculate_output_layer_delta(label[i])
		for layer in self.layers[-2::-1]:
			for node in layer.nodes:
				node.calculate_hidden_layer_delta()

	def update_weight(self, rate):
		#更新每个连接的权重
		for layer in self.layers[:-1]:
			for node in layer.nodes:
				for conn in node.downstream:
					conn.update_weight(rate)

	def calculate_gradient(self):
		#计算每个连接的梯度
		for layer in self.layers[:-1]:
			for node in layer.nodes:
				for conn in node.downstream:
					conn.calculate_gradient()

	def get_gratient(self, label, sample):
		#获得网络在一个样本下，每个连接上的梯度：样本标签，样本输入
		self.layers[0].ser_output(sample)
		for i in range(1, len(self.layers)):
			self.layers[i].calculate_output()
		return map(lambda node: node.output, self.layers[-1].nodes[:-1])

	def dump(self):
		#打印网络信息
		for layer in self.layers:
			layer.dump()

#通过梯度检查，验证代码是否正确
def gradient_check(network, sample_feature, sample_label):
	#梯度检查：神经网络，样本的特征， 样本标签

	#计算网络误差
	network_error = lambda vec1, vec2: \
		0.5 * reduce(lambda a, b: a + b, map(lambda v: (v[0] - v[1]) * (v[0] - v[1]), zip(vec1, vec2)))

	#获取网络在当前样本下的每个连接的梯度
	network.get_gratient(sample_feature, sample_label)

	#对每个权重做梯度检查
	for conn in network.connections.connectons:
		#获取指定连接的梯度
		actual_gradient = conn.get_gratient()

		#增加一个很小的值，计算网络的误差
		epsilon = 0.0001
		conn.weight +=epsilon
		error1 = network_error(network.predict(sample_feature), sample_label)

		#减去一个很小的值，计算网络的误差
		con.weight -= 2 * epsilon 
		error2 = network_error(network.predict(sample_feature), sample_label)

		#根据式6计算期望的梯度值
		expected_gradient = (error2 - error1) / (2 * epsilon)

		print('Expected gradient: \t%f\nactual gradient: \t%f' % (expected_gradient, actual_gradient))

