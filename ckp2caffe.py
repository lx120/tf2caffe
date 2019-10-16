import os
import sys
import tensorflow as tf
import numpy as np
from argparse import ArgumentParser
import tensorflow.contrib.image

caffe_root = '/home/lx/caffe/caffe-master'
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe

scale = 1


def ckp2npz(model_path, meta_path, var_len, scope_name=None, Sess=None):
    if Sess is None:
        Sess = tf.Session()

    Saver = tf.train.import_meta_graph(meta_path, import_scope=scope_name)
    Saver.restore(Sess, model_path)

    if scope_name is None or scope_name == '':
        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    else:
        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)

    # moving_mean = 'moving_mean'
    # moving_variance = 'moving_variance'

    if var_len < 1:
        var_len = len(all_vars)

    weights = []
    weights_name = []

    with Sess.as_default():
        for i in range(var_len):
            var = all_vars[i].eval()
            if len(var.shape) == 4:
                var = np.transpose(var, [3, 2, 0, 1])
            elif len(var.shape) == 2:
                var = np.transpose(var, [1, 0])
            weights.append(var)
            weights_name.append(all_vars[i].name)

    return np.array(weights), np.array(weights_name)


def view_last_conv_weights(weights, weights_name, save_path=None):
    flag = False

    w = []
    n = []
    for i in range(weights.shape[0]):
        if 'Adam' in weights_name[i] or '_pow' in weights_name[i]:
            continue
        w += [weights[i]]
        n += [weights_name[i]]

    weights_name = np.array(n)
    weights = np.array(w)

    for i in range(weights.shape[0]):
        '''
        if not flag and 'stage2' in weights_name[i]:
            weights_name[i] = 'stage1' + weights_name[i][6:]
            weights_name[i + 1] = 'stage1' + weights_name[i + 1][6:]

            weights_name[i + 2] = 'stage1' + weights_name[i + 2][6:]
            weights_name[i + 3] = 'stage1' + weights_name[i + 3][6:]
            weights_name[i + 4] = 'stage1' + weights_name[i + 4][6:]
            weights_name[i + 5] = 'stage1' + weights_name[i + 5][6:]

            flag = True
        '''

        if 'stage1/dense_2/kernel' in weights_name[i]:
            fc1_w = np.transpose(weights[i], [1, 0])
            fc1_w = fc1_w.reshape([7, 7, 512 // scale, 512])
            fc1_w = fc1_w.transpose([3, 2, 0, 1])
            fc1_w = fc1_w.reshape([512, -1])
            weights[i] = fc1_w

        if 'stage2/dense_1/kernel' in weights_name[i]:
            fc1_w = np.transpose(weights[i], [1, 0])
            fc1_w = fc1_w.reshape([7, 7, 512 // scale, 512])
            fc1_w = fc1_w.transpose([3, 2, 0, 1])
            fc1_w = fc1_w.reshape([512, -1])
            weights[i] = fc1_w

        print (weights_name[i] + '\t' + str(weights[i].shape))

    if save_path is not None:
        np.savez(save_path, weights=weights, name=weights_name)
        print('save model to ', save_path)

    return weights, weights_name


def load_weights_caffe(net, weights_in, weights_name_in, scope=None):
    # weights = weights_in
    # weights_name = weights_name_in

    weights = []
    weights_name = []
    if scope is None:
        weights = weights_in
        weights_name = weights_name_in
    else:
        for i in range(weights_name_in.shape[0]):
            if scope in weights_name_in[i]:
                weights.append(weights_in[i])
                weights_name.append(weights_name_in[i])
        weights = np.array(weights)
        weights_name = np.array(weights_name)

    tf_weight_idex = 0

    for key, value in net.params.items():
        if key.find('bn') >= 0:
            key_mean = key + '/moving_mean'
            net.params[key][0].data[...] = weights[tf_weight_idex + 2]
            print(weights_name[tf_weight_idex + 2], '   -->   ', key_mean, value[0].data.shape)

            key_variance = key + '/moving_variance'
            net.params[key][1].data[...] = weights[tf_weight_idex + 3]
            print(weights_name[tf_weight_idex + 3], '   -->   ', key_variance, value[1].data.shape)

            net.params[key][2].data[...] = 1.0

        elif key.find('scale') >= 0:
            key_gamma = key.replace('scale', 'bn') + '/gamma'
            net.params[key][0].data[...] = weights[tf_weight_idex]
            print(weights_name[tf_weight_idex], '   -->   ', key_gamma, value[0].data.shape)

            key_beta = key.replace('scale', 'bn') + '/beta'
            net.params[key][1].data[...] = weights[tf_weight_idex + 1]
            print(weights_name[tf_weight_idex + 1], '  -->  ', key_beta, value[1].data.shape)

            tf_weight_idex += 4

        elif key.find('conv') >= 0 or key.find('fc') >= 0:
            key_weights = key + '/weights'
            key_biases = key + '/biases'
            net.params[key][0].data[...] = weights[tf_weight_idex]
            print(weights_name[tf_weight_idex], '  -->  ', key_weights, value[0].data.shape)

            net.params[key][1].data[...] = weights[tf_weight_idex + 1]
            print(weights_name[tf_weight_idex + 1], '   -->   ', key_biases, value[1].data.shape)

            tf_weight_idex += 2

        else:
            raise Exception('Unknown layer')

    return net


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_path', default='./model/tf_model/Model'
                        , type=str)
    parser.add_argument('--meta_path', default='./modeltf_model/Model.meta'
                        , type=str)
    parser.add_argument('--save_path', default='./model_npz/model.npz'
                        , type=str)
    parser.add_argument("--deploy_file", type=str, default='./caffe_net/caffe_net.prototxt',
                        help="path to deployment file")
    parser.add_argument("--caffemodel_file", type=str, default='./caffe_model/caffe_net.caffemodel',
                        help="path to caffemodel file")
    parser.add_argument('--var_len', default=0, type=int)
    parser.add_argument('--scope', default=None, type=str)

    args = parser.parse_args()

    weights, weights_name = ckp2npz(args.model_path, args.meta_path, args.var_len)

    weights, weights_name = view_last_conv_weights(weights, weights_name, args.save_path)

    deploy_file = args.deploy_file
    caffemodel_file = args.caffemodel_file

    net = caffe.Net(deploy_file, caffe.TEST)

    net = load_weights_caffe(net, weights, weights_name, scope=args.scope)
    net.save(caffemodel_file)

    print('Model Saved!')
