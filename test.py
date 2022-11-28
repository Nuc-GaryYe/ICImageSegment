import paddle
import numpy as np

input_data = paddle.uniform([1,2, 10 , 10], dtype="float64")
label = paddle.ones([1,1, 10 , 10], dtype="float64")

input =  paddle.to_tensor(input_data)
label = paddle.to_tensor(label)

print(input)
print(label)

ce_loss = paddle.nn.CrossEntropyLoss(axis=1)
output = ce_loss(input, label)
print(output)