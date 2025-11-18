import torch

idx2char = ['d', 'l', 'e', 'a', 'r', 'n']
char2idx = {char: idx for idx, char in enumerate(idx2char)}
input_str = "dlearn"
target_str = "lanrla"

x_data = [char2idx[char] for char in input_str]
y_data = [char2idx[char] for char in target_str]

input_size = len(idx2char)
hidden_size = 6
batch_size = 1
seq_len = len(input_str)

one_hot_lookup = torch.eye(input_size)
x_one_hot = one_hot_lookup[x_data]
inputs = x_one_hot.view(seq_len, batch_size, input_size)
labels = torch.LongTensor(y_data).view(-1)


class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnncell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)
        self.fc = torch.nn.Linear(hidden_size, input_size)

    def forward(self, input, hidden):
        hidden = self.rnncell(input, hidden)
        output = self.fc(hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

net = Model(input_size, hidden_size)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.08)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

train_log = []
for epoch in range(15):
    optimizer.zero_grad()
    hidden = net.init_hidden(batch_size)
    loss = 0.0
    predicted_str = []

    for t in range(seq_len):
        input_t = inputs[t]
        label_t = labels[t]

        output, hidden = net(input_t, hidden)

        loss += criterion(output, label_t.unsqueeze(0))

        _, idx = output.max(dim=1)
        predicted_str.append(idx2char[idx.item()])

    loss.backward()
    optimizer.step()
    scheduler.step()

    epoch_loss = loss.item()
    pred_str = ''.join(predicted_str)
    train_log.append({
        'epoch': epoch + 1,
        'loss': epoch_loss,
        'predicted': pred_str,
        'target': target_str
    })

    print('Epoch [%d/15], Loss: %.4f, Predicted: %s, Target: %s' %
          (epoch + 1, epoch_loss, pred_str, target_str))

print("\nRNNCell训练完成！")
print('最终预测结果:', train_log[-1]["predicted"])
print('目标结果:', target_str)