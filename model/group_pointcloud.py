import os
import numpy as np
import tensorflow as tf
import time

from config import cfg


class VFELayer(object):

    def __init__(self, out_channels, name):
        super(VFELayer, self).__init__()
        self.units = int(out_channels / 2)
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            self.dense = tf.layers.Dense(
                self.units, tf.nn.relu, name='dense', _reuse=tf.AUTO_REUSE, _scope=scope)
            self.batch_norm = tf.layers.BatchNormalization(
                name='batch_norm', fused=True, _reuse=tf.AUTO_REUSE, _scope=scope)

    def apply(self, inputs, mask, training):
        # [K, T, 7] 
        pointwise = self.batch_norm.apply(self.dense.apply(inputs), training)

        #n [K, 1, units] 计算张量的各个维度上的元素的最大值
        aggregated = tf.reduce_max(pointwise, axis=1, keep_dims=True)

        # [K, T, units] 35 张量复制
        repeated = tf.tile(aggregated, [1, cfg.VOXEL_POINT_COUNT, 1])

        # [K, T, 2 * units] 连接
        concatenated = tf.concat([pointwise, repeated], axis=2)

        mask = tf.tile(mask, [1, 1, 2 * self.units])

        concatenated = tf.multiply(concatenated, tf.cast(mask, tf.float32))

        return concatenated


class FeatureNet(object):

    def __init__(self, training, batch_size, name=''):
        super(FeatureNet, self).__init__()
        self.training = training

       
        self.batch_size = batch_size
        # [ΣK, 35/45, 7] 非空 voxel 数量
        self.feature = tf.placeholder(  #35
            tf.float32, [None, cfg.VOXEL_POINT_COUNT, 7], name='feature')
        # [ΣK]
        self.number = tf.placeholder(tf.int64, [None], name='number')
        # [ΣK, 4]  (batch, d, h, w)
        self.coordinate = tf.placeholder(
            tf.int64, [None, 4], name='coordinate')

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            # 输出通道32/128
            self.vfe1 = VFELayer(32, 'VFE-1')
            self.vfe2 = VFELayer(128, 'VFE-2')

        #  [K, T, 2 * units]
        mask = tf.not_equal(tf.reduce_max(
            self.feature, axis=2, keep_dims=True), 0)
        x = self.vfe1.apply(self.feature, mask, self.training)
        x = self.vfe2.apply(x, mask, self.training)

        # [ΣK, 128]
        voxelwise = tf.reduce_max(x, axis=1)

        # car: [N * 10 * 400 * 352 * 128]
        # pedestrian/cyclist: [N * 10 * 200 * 240 * 128]
       
        """
        # 根据indices将updates散布到新的（初始为零）张量
        scatter_nd(indices,updates,shape,name=None)
        """
        self.outputs = tf.scatter_nd(
            self.coordinate, voxelwise, [self.batch_size, 10, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 128])


