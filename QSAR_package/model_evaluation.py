# coding: utf-8

# Author: Zhang Shengde <zhangshd@foxmail.com>

from sklearn.metrics import accuracy_score,matthews_corrcoef,r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import cross_val_predict,LeaveOneOut,KFold
import matplotlib.pyplot as plt
import pandas as pd
import copy

__all__ = ["modelEvaluator","modeling"]

class modelEvaluator(object):
    """计算用于评价二分类模型或回归模型的统计量"""
    def __init__(self,y_true,y_pred,model_kind='clf'):
        """参数：
           -----
           y_true：array型(一维)，样本label真实值
           y_pred：array型(一维)，样本label预测值
           model_kind：string型，模型类型，'clf'(分类)或'rgr'(回归)"""
        
        if model_kind == 'clf':
            self.__Clf_metrics(y_true,y_pred)
        if model_kind == 'rgr':
            self.__Rgr_metrics(y_true,y_pred)
            
    def __Clf_metrics(self,y_true,y_pred):
        """计算二分类模型预测结果的TP、TN、FP、FN以及accuracy、MCC、SE、SP"""
        
        self.accuracy = accuracy_score(y_true, y_pred)
        self.mcc = matthews_corrcoef(y_true, y_pred)
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
        
    def __Rgr_metrics(self,y_true,y_pred):
        """计算回归模型预测结果的R2、RMSE、MAE"""
        self.r2 = round(r2_score(y_true, y_pred),4)
        self.mse = round(mean_squared_error(y_true, y_pred),4)
        # self.mae = mean_absolute_error(y_true, y_pred)
    
class modeling(object):
    """拟合模型(训练集)及评价，预测样本(测试集)及评价，交互检验(训练集)及评价
    example：
    --------
    model = modeling(estimator,params=best_params)
    model.Fit(tr_scaled_x.loc[:,grid.best_features], tr_y)
    model.Predict(te_scaled_x.loc[:,grid.best_features],te_y)
    model.CrossVal(cv='LOO')
    model.SaveResults(res_path)
    model.ShowResults()"""
    def __init__(self,estimator,params=None):
        """参数：
           -----
           estimator：object型，传入学习器
           params：无或dict型，如果传入参数字典，则会重设学习器的超参数"""
        self.estimator = estimator
        if params is not None:
            self.params = params
            self.estimator.set_params(**params)
        else:
            self.params = self.estimator.get_params()
            
    def Fit(self,tr_scaled_x,tr_y):
        """拟合模型，并且评价训练集预测结果，预测结果评价存放于tr_metrics属性
           参数：
           -----
           tr_scaled_x：array型，训练样本的feature数据
           tr_y：array型(一维)，训练样本的label数据"""
        self.tr_scaled_x = tr_scaled_x
        self.tr_y = tr_y
        self.estimator.fit(tr_scaled_x,tr_y)
        self.tr_pred_y = self.estimator.predict(tr_scaled_x)
        self.tr_evaluator = modelEvaluator(self.tr_y,self.tr_pred_y,model_kind='rgr')
        self.tr_metrics = dict(self.tr_evaluator.__dict__.items())
    def Predict(self,te_scaled_x,te_y):
        """预测样本(测试集)，并且评价(测试集)预测结果，预测结果评价存放于te_metrics属性参数：
           参数：
           -----
           te_scaled_x：array型，测试样本的feature数据
           te_y：array型(一维)，测试样本的label数据"""
        self.te_y = te_y
        self.te_pred_y = self.estimator.predict(te_scaled_x)
        self.te_evaluator = modelEvaluator(self.te_y,self.te_pred_y,model_kind='rgr')
        self.te_metrics = dict(self.te_evaluator.__dict__.items())
    def CrossVal(self,cv):
        """进行交互检验预测(训练集)，并且评价预测结果，预测结果评价存放于cv_metrics属性，使用此方法前必须先fit
           参数：
           -----
           cv：int、string或object型，指定交互检验生成器如Kfold、LeaveOneOut等"""
        if type(cv) == int:
            self.cv = KFold(n_splits=cv, shuffle=True,random_state=0)
        elif cv == 'LOO':
            self.cv = LeaveOneOut()
        else:
            self.cv = cv
        self.cv_pred_y = cross_val_predict(self.estimator,self.tr_scaled_x,y=self.tr_y,n_jobs=-1,cv=self.cv)
        self.cv_evaluator = modelEvaluator(self.tr_y,self.cv_pred_y,model_kind='rgr')
        self.cv_metrics = dict(self.cv_evaluator.__dict__.items())
    def ShowResults(self):
        """打印模型所有评价结果(训练集预测结果、测试集预测结果、交互检验预测结果)，并且将训练集与测试集预测结果绘制散点图"""
        print('\033[1m{}\033[0m'.format(pd.DataFrame({'tr':self.tr_metrics,'te':self.te_metrics,'cv':self.cv_metrics})))
        axisMin = min(self.tr_y.min(),self.te_y.min(),self.tr_pred_y.min(),self.te_pred_y.min())-0.5
        axisMax = max(self.tr_y.max(),self.te_y.max(),self.tr_pred_y.max(),self.te_pred_y.max())+0.5
        plt.plot(self.tr_y,self.tr_pred_y,'xb')
        plt.plot(self.te_y,self.te_pred_y,'+r')
        plt.plot([axisMin,axisMax],[axisMin,axisMax],'k')
        plt.axis([axisMin,axisMax,axisMin,axisMax])
        plt.show()
    def SaveResults(self,res_path,notes=None):
        """将模型结果保存至CSV文件中
        参数：
        -----
        res_path：string型，结果文件的路径，如果是已存在的文件则追加行
        notes：string型，可以添加结果备注信息"""
        metrics = []
        for s in ['tr_','cv_','te_']:
            metrics_ = copy.deepcopy(eval('self.{}metrics'.format(s)))
            for k in copy.deepcopy(list(metrics_.keys())):
                metrics_[s+k]=metrics_.pop(k)
            metrics.append(metrics_)
        all_metrics = dict(metrics[0],**metrics[1],**metrics[2])
        self.results_df = pd.DataFrame(all_metrics,index=[0])
        self.estimatorName = str(self.estimator)[:str(self.estimator).find("(")]
        self.results_df.insert(0,'params',str(self.params))
        self.results_df.insert(0,'algorithm',str(self.estimatorName))
        self.results_df.insert(0,'n_features',self.tr_scaled_x.shape[1])
        if notes is not None:
            self.results_df.insert(-1,'notes',notes)

        try:
            with open(res_path) as testfile:
                pass
        except IOError:
            with open(res_path,mode='w') as fobj:
                fobj.write(','.join(self.results_df.columns)+'\n')
        self.results_df.to_csv(res_path,index=False,header=False,mode='a',float_format='%6.4f')


if __name__ == '__main__':
    pass