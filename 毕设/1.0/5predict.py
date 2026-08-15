# 使用模型进行预测
model.eval() 
start_text = '白日依山尽'  
# 你的诗的开头
generated_text = start_text 
with paddle.no_grad():    
    for _ in range(100): 
        # 生成100个字符         
        inputs = paddle.to_tensor([char_to_id[c] for c in start_text]) 
        # 假设你已经定义了字符到ID的映射  char_to_id        
        outputs = model(inputs)        
        _, predicted = paddle.max(outputs.data, 1)         
        generated_text += id_to_char[predicted[-1].item()] 
        
        # 假设你已经定义了ID到字符的映射  id_to_char     
        start_text = start_text[1:] + id_to_char[predicted[-1].item()]  
print('Generated text: ', generated_text) 