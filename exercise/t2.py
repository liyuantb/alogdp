import torch

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(device)

x = torch.tensor(1.0, requires_grad=True).to(device)
y = torch.tensor([1.0, 2.0], requires_grad=True).to(device)

print(x.shape)
print(y.shape)

z = x * y
print(z.shape)
print(z.device)

zz = z.mean()
print(x.grad)
zz.backward()
#print(x.grad)
#print(y.grad)


print(torch.get_default_device())
print(torch.get_default_dtype())

zero = torch.zeros([2, 4], layout=torch.strided)
print(zero)
one = torch.ones_like(zero)
print(one)


empty = torch.empty([2, 2])
print(empty)

ra = torch.range(0, 10, step=1)
print(ra)
rb = torch.linspace(0, 10, 1000)
#print(rb)

rd = torch.rand([3, 5])
print(rd)
rn = torch.randn([3, 5])
print(rn)


ar = torch.arange(200).reshape([20, 10]).contiguous()
print(ar)
tar = ar[2]
print(tar)
ttar = ar[2, 2]
print(ttar, ttar.shape)


