## tensorflow模型转为caffe模型的步骤：

(1) 定义好和tensorflow网络结构一致的caffe的网络结构(prototxt)

(2) ckp2caffe.py中的view_last_conv_weights函数是将 针对最后一个卷积到第一个全连接层的权重从tensorflow变为caffe的格式，可能需要根据自己的网络在该层的名字的不同进行修改适配

(3) 运行ckp2caffe.py，将tensorflow的checkpoint模型变为caffemodel，

```
运行：
python ckp2caffe.py --model_path './model/tf_model/Model' --meta_path './model/tf_model/Model.meta' --save_path './model_npz/tf_model.npz' --deploy_file './caffe_net/caffe_net.prototxt' --caffemodel_file ./caffe_model/caffe_net.caffemodel

```

其中， weights的形状将依据caffe（N * C * H * W）的特性进行transpose， np.transpose(var, [3, 2, 0, 1]) 和 np.transpose(var, [1, 0])


注意点：

(1) caffe与tensorflow的pad方式不同

(2) tensorflow中的BN层在caffe中为BN+scale，注意两个框架下的这些层的参数的不同，需要统一