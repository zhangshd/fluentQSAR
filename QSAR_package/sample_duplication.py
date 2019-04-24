# coding: utf-8

# Author: Qin Zijian <zijianqin@foxmail.com>

from rdkit import Chem
import numpy as np

class Duplicate(object):
    """
    """
    def __init__(self):
        self.unique_inchi = []
        self.repeat_inchi = []
        self.unique_index = []
        self.repeat_index = []
    
    def SetInputSmiles(self, smi_file=""):
        self.input_file = smi_file
        
        self.np_init_smi = np.loadtxt(self.input_file, dtype=np.str, comments=None)
        self.init_mols = [Chem.MolFromSmiles(m) for m in self.np_init_smi]
        self.init_inchi = [Chem.MolToInchiKey(m) for m in self.init_mols]
        
        self.init_num = len(self.init_inchi)
        print("Done. {} molecules have been converted to InChiKey."\
              .format(self.init_num))
    
    def SetInputSDF(self, sdf_file=""):
        self.input_file = sdf_file
        
        self.suppl = Chem.SDMolSupplier(self.input_file)
        self.np_init_smi = np.array([Chem.MolToSmiles(m) for m in self.suppl])
        self.init_mols = [Chem.MolFromSmiles(m) for m in self.np_init_smi]
        self.init_inchi = [Chem.MolToInchiKey(m) for m in self.init_mols]

        self.init_num = len(self.init_inchi)
        print("Done. {} molecules have been converted to InChiKey."\
              .format(self.init_num))
    
    def Duplicate(self):
        for i in range(0, len(self.init_inchi)):
            if self.init_inchi[i] not in self.unique_inchi:
                self.unique_inchi.append(self.init_inchi[i])
                self.unique_index.append(i)
            else:
                self.repeat_inchi.append(self.init_inchi[i])
                self.repeat_index.append(i)
        
        self.unique_num = len(self.unique_inchi)
        self.repeat_num = len(self.repeat_inchi)
        print("Done. init: {}, unique: {}, repeat: {}"\
              .format(self.init_num, self.unique_num, self.repeat_num))
    
    def SaveResults(self):
        # save unique smiles
        np.savetxt(self.input_file[:-4]+"_unique"+str(self.unique_num)+".smi",\
                   self.np_init_smi[self.unique_index], fmt="%s", comments=None)
        
        # save repeat smiles
        np.savetxt(self.input_file[:-4]+"_repeat"+str(self.repeat_num)+".smi",\
                   self.np_init_smi[self.repeat_index], fmt="%s", comments=None)
        
        # save log file
        self.np_init_inchi = np.array(self.init_inchi)
        with open(self.input_file[:-4]+"_duplicated.log", "w") as fobj:
            fobj.write("The index was started from 1 (not 0).\n\n")
            for repeat in set(self.repeat_inchi):
                fobj.write(str(np.where(self.np_init_inchi == repeat)[0]+1))
                fobj.write("\n")
        
        print("Done.")
        print("The unique smi file was saved at {}"\
              .format(self.input_file[:-4]+"_unique"+str(self.unique_num)+".smi"))
        print("The repeat smi file was saved at {}"\
              .format(self.input_file[:-4]+"_repeat"+str(self.repeat_num)+".smi"))
        print("The log file was saved at {}"\
              .format(self.input_file[:-4]+"_duplicated.log"))



if __name__ == '__main__':
    pass