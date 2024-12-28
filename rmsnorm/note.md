RMSNorm 比layerNormal更简单实现
只需要计算一次均方根值，然后对每个输入进行归一化即可
$$
output[i] = \gamma * \frac{input[i]}{RMS(x)} where RMS(x) = \sqrt{\frac{1}{N}\sum_{i=1}^{N}x_i^2 + \epsilon}
$$
