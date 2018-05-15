import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V

class LayerNormGRUCell(nn.GRUCell):
	# only apply LN to reset gate r and update gate z, not including output gate n
	"""
		weight_ih: the learnable input-hidden weights, of shape
            `(3*hidden_size x input_size)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(3*hidden_size x hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(3*hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(3*hidden_size)`
	"""
	def __init__(self, input_size, hidden_size, bias=True, layer_norm=False):
		super(LayerNormGRUCell, self).__init__(input_size, hidden_size, bias)
		if layer_norm:
			self.gamma_ih = nn.Parameter(torch.ones(2 * self.hidden_size))
			self.gamma_hh = nn.Parameter(torch.ones(2 * self.hidden_size))
		self.eps = 0
		self.layer_norm = layer_norm

	def _layer_norm(self, x, g, b):
		mean = x.mean(1).unsqueeze(-1).expand_as(x)
		std = x.std(1).unsqueeze(-1).expand_as(x)
		return g.unsqueeze(0).expand_as(x) * ((x - mean) / (std + self.eps)) + b.unsqueeze(0).expand_as(x)

	def forward(self, x, h):
		if self.layer_norm:
			ih_rz = self._layer_norm(
				torch.mm(x, self.weight_ih.narrow(0, 0, 2 * self.hidden_size).transpose(0, 1)),
				self.gamma_ih,
				self.bias_ih.narrow(0, 0, 2 * self.hidden_size))

			hh_rz = self._layer_norm(
				torch.mm(h, self.weight_hh.narrow(0, 0, 2 * self.hidden_size).transpose(0, 1)),
				self.gamma_hh,
				self.bias_hh.narrow(0, 0, 2 * self.hidden_size))
		else:
			ih_rz = torch.mm(x, self.weight_ih.narrow(0, 0, 2 * self.hidden_size).transpose(0, 1)) \
					+ self.bias_ih.narrow(0, 0, 2 * self.hidden_size)

			hh_rz = torch.mm(h, self.weight_hh.narrow(0, 0, 2 * self.hidden_size).transpose(0, 1)) \
					+ self.bias_hh.narrow(0, 0, 2 * self.hidden_size)

		rz = torch.sigmoid(ih_rz + hh_rz)
		r = rz.narrow(1, 0, self.hidden_size)
		z = rz.narrow(1, self.hidden_size, self.hidden_size)

		ih_n = torch.mm(x, self.weight_ih.narrow(0, 2 * self.hidden_size, self.hidden_size).transpose(0, 1))
		hh_n = torch.mm(h, self.weight_hh.narrow(0, 2 * self.hidden_size, self.hidden_size).transpose(0, 1))

		n = torch.tanh(ih_n + r * hh_n)
		h = (1 - z) * n + z * h
		return h

class LayerNormGRUCellModule(nn.Module):
	def __init__(self, input_size, hidden_size, bias=True, layer_norm=False):
		super(LayerNormGRUCellModule, self).__init__()
		self.LayerNormGRUCell = LayerNormGRUCell(input_size, hidden_size, bias, layer_norm)

	def forward(self, x, h):
		return self.LayerNormGRUCell(x,h)

class LayerNormGRU(nn.Module):
	def __init__(self, input_size, hidden_size, nlayers, dropout=0.0, bias=True, layer_norm=False):
		super(LayerNormGRU, self).__init__()
		self.cell = torch.nn.ModuleList()
		self.nlayers = nlayers
		for i in range(self.nlayers):
			self.cell.append(LayerNormGRUCellModule(input_size, hidden_size, bias, layer_norm))
		self.dropout = dropout

	def forward(self, xs, h):
		# xs: [bptt*batch_size*hidden_size]
		# h: [nlayers*batch_size*hidden_size]
		hs = h.new_zeros(h.size())
		ys = []
		for i in range(xs.size(0)):
			x = xs.narrow(0, i, 1).squeeze(0)
			for j in range(self.nlayers):
				hs[j] = self.cell[j](x, h.narrow(0, j, 1).squeeze(0))
				x = hs[j].clone()
				if (self.dropout > 0) and (j != self.nlayers-1):
					x = F.dropout(x, p=self.dropout, training=self.training, inplace=False)
			ys.append(x.unsqueeze(0))
			for i in range(self.nlayers):
				hs[i] = hs[i].unsqueeze(0)
			h = hs.clone()
		y = torch.cat(ys, 0)
		return y, h

def LayerNormGRUCellTest():
	torch.manual_seed(1234)
	ninp = 50
	nhid = 50
	nsep = 100
	nbatch = 2
	x = V(torch.rand(nsep, nbatch, ninp), requires_grad=True)
	hidden = V(torch.rand(1, nbatch, nhid), requires_grad=True)
	ref = nn.GRUCell(ninp, nhid, bias=True)
	# under_test = LayerNormGRUCell(5, 5, bias=True, layer_norm=False)
	under_test = LayerNormGRUCellModule(ninp, nhid, bias=True, layer_norm=False)

	# Make ref and cur same parameters:
	val = torch.rand(1)[0]
	for c in under_test.parameters():
		c.data.fill_(val)
	# val = torch.rand(1)[0]
	for r in ref.parameters():
		r.data.fill_(val)

	objective = V(torch.zeros(nbatch, nhid))

	i, j = x.clone(), hidden.clone().squeeze(0)
	g, h = x.clone(), hidden.clone().squeeze(0)
	for index in range(nsep):
		j = ref(i[index], j)
		h = under_test(g[index], h)
		err = torch.sum((j - h) ** 2)
		print(err.item())
		assert (err.item() < 1e-1)
		# assert (torch.equal(j.data, h.data))
		ref_loss = torch.sum((i - objective) ** 2)
		cus_loss = torch.sum((g - objective) ** 2)
		ref_loss.backward()
		cus_loss.backward()
	# print('LayerNormGRUCell Test Passed')
	print('LayerNormGRUCellModule Test Passed')


def LayerNormGRUTest():
	torch.manual_seed(1234)
	nlayers = 2
	ninp = 50
	nhid = 50
	time_step = 10
	nbatch = 20
	x = V(torch.rand(time_step, nbatch, ninp), requires_grad=True)
	hidden = V(torch.rand(nlayers, nbatch, nhid), requires_grad=True)
	ref = nn.GRU(ninp, nhid, nlayers, dropout=0.5, bias=True)
	under_test = LayerNormGRU(ninp, nhid, nlayers, dropout=0.5, bias=True, layer_norm=False)

	# Make ref and cur same parameters:
	val = torch.rand(1)[0]
	for c in under_test.parameters():
		c.data.fill_(val)
	# val = torch.rand(1)[0]
	for r in ref.parameters():
		r.data.fill_(val)

	objective = V(torch.zeros(nbatch, nhid))

	i, j = V(x.clone(), requires_grad=True), V(hidden.data, requires_grad=True)
	g, h = V(x.clone(), requires_grad=True), V(hidden.data, requires_grad=True)
	for index in range(time_step):
		i, j = ref(i, j)
		g, h = under_test(g, h)
		err = torch.sum((g - i) ** 2)
		print("output err:{}".format(err.item()))
		# assert (err.item() < 5e-1)
		for k in range(nlayers):
			err = torch.sum((j[k] - h[k]) ** 2)
			print("layer {} hidden err:{}".format(k, err.item()))
			# assert (err.item() < 1e-3)

		ref_loss = torch.sum((i - objective) ** 2)
		cus_loss = torch.sum((g - objective) ** 2)
		ref_loss.backward(retain_graph=True)
		cus_loss.backward(retain_graph=True)
	print('LayerNormGRU Test Passed')

if __name__ == '__main__':
	# LayerNormGRUCellTest()
	LayerNormGRUTest()

