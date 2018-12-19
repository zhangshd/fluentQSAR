# coding: utf-8

# Author: Zhang Shengde <zhangshd@foxmail.com>

from sklearn.metrics import accuracy_score,matthews_corrcoef,r2_score,mean_absolute_error,mean_squared_error

class modelEvaluator(object):
    """计算用于评价二分类模型或回归模型的统计量"""
    def __init__(self,y_true,y_pred,model_kind='clf'):
        """参数：
           -----
           y_true：array型(一维)，样本label真实值
           y_pred：array型(一维)，样本label预测值
           model_kind：string型，模型类型，'clf'(分类)或'rgr'(回归)"""
        
        if model_kind == 'clf':
            self.__clf_metrics(y_true,y_pred)
        if model_kind == 'rgr':
            self.__rgr_metrics(y_true,y_pred)
            
    def __clf_metrics(self,y_true,y_pred):
        """计算二分类模型预测结果的TP、TN、FP、FN以及accuracy、MCC、SE、SP"""
        
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        for i in range(len(y_true)):
            if y_true[i] == 1 and y_pred[i] == 1:
                self.tp += 1
            if y_true[i] == 1 and y_pred[i] == 0:
                self.fn += 1
            if y_true[i] == 0 and y_pred[i] == 1:
                self.fp += 1
            if y_true[i] == 0 and y_pred[i] == 0:
                self.tn += 1
        self.se = float(self.tp) / float(self.tp + self.fn)
        self.sp = float(self.tn) / float(self.tn + self.fp) 
        self.accuracy = accuracy_score(y_true, y_pred)
        self.MCC      = matthews_corrcoef(y_true, y_pred)
        
    def __rgr_metrics(self,y_true,y_pred):
        """计算回归模型预测结果的R2、RMSE、MAE"""
        self.R2 = r2_score(y_true, y_pred)
        self.RMSE = (mean_squared_error(y_true, y_pred))**0.5
        self.MAE = mean_absolute_error(y_true, y_pred)


if __name__ == '__main__':
    pass