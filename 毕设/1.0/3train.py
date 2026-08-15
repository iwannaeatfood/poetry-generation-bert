from paddle.nn import CrossEntropyLoss 
from paddle.optimizer import Adam  
# 定义损失函数和优化器 
loss_fn = CrossEntropyLoss() 
optimizer = Adam(parameters=model.parameters())  
# 训练模型 
num_epochs = 10  # 你可以设置一个合适的epoch数 
for epoch in range(num_epochs):     
    for batch in train_dataloader:  # 假设你已经定义了数据加载器 
        train_dataloader         
        inputs, targets = batch         
        outputs = model(inputs)         
        loss = loss_fn(outputs, targets)          
        loss.backward()         
        optimizer.step()         
        optimizer.clear_grad()          
        print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, num_epochs, loss.item())) 