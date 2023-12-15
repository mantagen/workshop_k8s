# ./mnist/train.py
import torch
import torchvision
import net


trn_ds = torchvision.datasets.MNIST(root="data", train=True,  download=True, transform=torchvision.transforms.ToTensor())
tst_ds = torchvision.datasets.MNIST(root="data", train=False, download=True, transform=torchvision.transforms.ToTensor())
trn_dl = torch.utils.data.DataLoader(trn_ds, batch_size=64)
tst_dl = torch.utils.data.DataLoader(tst_ds, batch_size=64)


myCnn = net.Net()
loss = torch.nn.CrossEntropyLoss()
opt = torch.torch.optim.Adam(myCnn.parameters(), lr=1e-5)


for t in range(6):
  for idx, (x, y) in enumerate(trn_dl):
    logits = myCnn(x)
    l = loss(logits, y)
    opt.zero_grad()
    l.backward()
    opt.step()
    if idx % 50 == 0: print(f"Epoch: {t}, Loss: {l.item()}")

  with torch.no_grad():
      correct = 0
      total = 0
      for x, y in tst_dl:
        logits = myCnn(x)
        _, pred = torch.max(logits, dim=1)
        total += y.size(0)
        correct += (pred == y).sum().item()
      print(f"Epoch: {t}, Accuracy: {correct/total}")
  torch.save(myCnn.state_dict(), f"./mnist.pt")