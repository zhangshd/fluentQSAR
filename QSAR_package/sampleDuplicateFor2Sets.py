
# coding: utf-8

# Author: Zhang Shengde

import numpy as np
import pandas as pd
from rdkit import Chem

__all__ = ["DuplicateFor2Sets"]

class DuplicateFor2Sets(object):
    """把在集合2中出现的属于集合1的分子去除
    Example:
    --------
    path1 = './patent_342.csv' 
    path2 = './pi3k_SGR_r1.csv'
    duplicater = DuplicateFor2Sets()
    duplicater.InputCSV(path1, path2, smi_label='smi')
    duplicater.Duplicate(save_csv=True)"""
    def __init__(self):
        pass
    def InputCSV(self,path1,path2,smi_label='smi'):
        """参数：
           -----
           path1: 集合1所在文件路径(对照文件)
           path2: 集合2所在文件路径(待去重文件)
           smi_label：smiles所在列的列名"""
        self.df1 = pd.read_csv(path1)
        self.df2 = pd.read_csv(path2)
        self.path2 = path2
        self.smi_label = smi_label
    def CanonicSmi(self,smi_list):
        """生成canonical smiles"""
        mols = [Chem.MolFromSmiles(m) for m in smi_list]
        canonic_smi = []
        invalid_id = []
        for i in range(len(smi_list)):
            if mols[i] is not None:
                canonic_smi.append(Chem.MolToSmiles(mols[i]))
            else:
                invalid_id.append(i)
                canonic_smi.append('invalid')
                print('The {}th smi is invalid.'.format(i))
        return canonic_smi,invalid_id
    def Duplicate(self,save_csv=True):
        self.canonic_smi1,self.invalid_id1 = self.CanonicSmi(self.df1.loc[:,self.smi_label])
        self.canonic_smi2,self.invalid_id2 = self.CanonicSmi(self.df2.loc[:,self.smi_label])
        self.df1.loc[:,self.smi_label] = self.canonic_smi1
        self.df2.loc[:,self.smi_label] = self.canonic_smi2
        self.df2.drop_duplicates(subset=self.smi_label,inplace=True) #内部去重
        
        self.duplicates_id = []
        for i in range(len(self.df2)):
            if self.df2.loc[:,self.smi_label][i] in set(self.df1.loc[:,self.smi_label]):
                self.duplicates_id.append(i)
                print('The {}th smi is a duplicate (with id starts from 0).'.format(i))
        self.df2_out = self.df2.drop(index=self.duplicates_id)
        if save_csv:
            path_out = self.path2[:-4]+'_duplicated_{}.csv'.format(len(self.df2_out))
            self.df2_out.to_csv(path_out,index=False)
            print('The duplicated result has saved in "{}".'.format(path_out))

if __name__ == '__main__':

    path1 = r"C:\OneDrive\Jupyter_notebook\myPackage\QSAR_package\patent_342.csv"  #对照文件路径
    path2 = r"C:\OneDrive\Jupyter_notebook\myPackage\QSAR_package\pi3k_SGR_r1.csv" #待去重文件路径
    duplicater = DuplicateFor2Sets()
    duplicater.InputCSV(path1, path2, smi_label='smi')
    duplicater.Duplicate(save_csv=True)

