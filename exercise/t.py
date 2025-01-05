import torch
 
# 检查CUDA是否可用
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available!  Training on GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available.  Training on CPU.")

torch.empty(10).to(device=device)
torch.get_default_device()
 
# 定义一些数据
#print(x)
 
# 定义模型和损失函数
class Model(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(Model, self).__init__()
        self.n_hidden = n_hidden
        self.linear1 = torch.nn.Linear(n_features, n_hidden)
        self.linear2 = torch.nn.Linear(n_hidden, n_hidden)
        self.linear3 = torch.nn.Linear(n_hidden, n_hidden)
        self.linear4 = torch.nn.Linear(n_hidden, n_hidden)
        self.linear5 = torch.nn.Linear(n_hidden, n_hidden)
        self.linear6 = torch.nn.Linear(n_hidden, n_hidden)
        self.linear7 = torch.nn.Linear(n_hidden, n_hidden)
        self.linear8 = torch.nn.Linear(n_hidden, n_hidden)
        self.linear9 = torch.nn.Linear(n_hidden, n_hidden)
        self.linear10 = torch.nn.Linear(n_hidden, n_output)
 
    def forward(self, x):
        print('xxxxxxxxxxxxxxxxxxxx')
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        x = torch.relu(self.linear4(x))
        x = torch.relu(self.linear5(x))
        x = torch.relu(self.linear6(x))
        x = torch.relu(self.linear7(x))
        x = torch.relu(self.linear8(x))
        x = torch.relu(self.linear9(x))
        x = self.linear10(x)
        return x
 
model = Model(1000, 100, 1).to(device)
criterion = torch.nn.MSELoss().to(device)
 
# 定义优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
 
# 训练模型
for t in range(3):
    x = torch.linspace(t, t+10, 1000).to(device)
    y = torch.sin(x).to(device)

    prediction = model(x)
    loss = criterion(prediction, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
 
    if t % 100 == 0:
        print(f"t={t}, loss={loss.item()}")
 
# 保存和加载模型
torch.save(model.state_dict(), "model.ckpt")
model.load_state_dict(torch.load("model.ckpt"))