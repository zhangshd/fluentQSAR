# coding: utf-8

# Author: Zhang Shengde <zhangshd@foxmail.com>

import numpy as np

__all__ = ["batchGenerator"]

class batchGenerator(object):
    """批量样本生成器
       示例：
       -----
       batch_gen = batchGenerator([x_data,y_data],batch_size=50)
       batch_x,batch_y = batch_gen.NextBatch()"""
    def __init__(self,data,batch_size,shuffle=True):
        """
        参数：
        -----
        data：由1个或多个\033[41;1mNumPy数组\033[0m构成的列表，如[x_data,y_data]或[x_data]，其中x_data、y_data都为数组(\033[41;1m不能为DataFrame\033[0m)
        batch_size：int型，每个批次的样本数量
        shuffle：bool型，控制是否在每次分批前打乱样本的顺序
        
        """
        self.batch_gen = self.BatchGenerator(data,batch_size,shuffle=True) # assign batch_gen as batch generator
        
    def __ShuffleAlignedList(self,data):
        """Shuffle arrays in a list by shuffling each array identically."""
        self.sample_num = len(data[0])
        self.shuffled_index = np.random.permutation(self.sample_num)
        return [d[self.shuffled_index] for d in data]

    def BatchGenerator(self,data,batch_size,shuffle=True):
        """Generate batches of data.

        Given a list of array-like objects, generate batches of a given
        size by yielding a list of array-like objects corresponding to the
        same slice of each input.
        """

        self.batch_count = 0
        while True:
            if self.batch_count == 0:
                if shuffle:
                    self.data = self.__ShuffleAlignedList(data)
            self.start = self.batch_count * batch_size
            self.end = self.start + batch_size
            if self.end > len(data[0]):
                self.batch_count = 0
                yield [d[self.start:] for d in data]
            else:
                self.batch_count += 1
                yield [d[self.start:self.end] for d in data]

    def NextBatch(self):
        """返回一个批次的样本"""
        return next(self.batch_gen)

if __name__ == '__main__':
    pass