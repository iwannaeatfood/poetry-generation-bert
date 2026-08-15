# 评估模型
model.eval() 
correct = 0 
total = 0 
with paddle.no_grad(): 
    for batch in test_dataloader: 
        # 假设你已经定义了数据加载器 test_dataloader 
        inputs, targets = batch    
        outputs = model(inputs)      
        _, predicted = paddle.max(outputs.data, 1)
        total += targets.size(0)  
        correct += (predicted == targets).sum().item()  
print('Test Accuracy of the model on the test images: %d %%' % (100 * correct / total)) 