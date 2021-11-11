## Bert+GAT

1. 取连续batch_size个句子作为一个batch
2. 经过Bert得到句子表示，[batch_size, embedding_size]
3. 一个batch构建一个图，经过GAT仍然返回句子表示, [batch_size, embedding_size]
4. 最后接线性层，用回归算loss

注： 前batch_size=32个句子构建的图

haha, 没有31
![](https://cdn.jsdelivr.net/gh/growvv/image-bed//mac-m1/%E5%89%8D64%E4%B8%AA.png)
