# QSAR_package使用说明
## 1. 使用前准备：
下载所有脚本，把把所有文件解压后存放至一个目录，如```$/myPackage/```
![](https://github.com/zhangshd/test/blob/master/%E7%A4%BA%E4%BE%8B%E5%9B%BE%E7%89%87/Snipaste_2019-04-25_12-47-38.png)
![](https://github.com/zhangshd/test/blob/master/%E7%A4%BA%E4%BE%8B%E5%9B%BE%E7%89%87/Snipaste_2019-04-26_10-37-48.png)
新建一个文本文件，把上述目录的路径粘贴至这个文件内，然后把后缀改为```.pth```，如```myPackage.pth```
![](https://github.com/zhangshd/test/blob/master/%E7%A4%BA%E4%BE%8B%E5%9B%BE%E7%89%87/Snipaste_2019-04-26_10-47-04.png)
打开cmd，输入```python```进入Python交互界面  
```python
import sys
```  
```python
sys.path
```
  
找到一个类似```..\\lib\\site-packages```的路径  
![](https://github.com/zhangshd/test/blob/master/%E7%A4%BA%E4%BE%8B%E5%9B%BE%E7%89%87/Snipaste_2019-04-26_10-51-27.png)
![](https://github.com/zhangshd/test/blob/master/%E7%A4%BA%E4%BE%8B%E5%9B%BE%E7%89%87/Snipaste_2019-04-26_11-03-33.png)
然后进入这个文件夹，把刚才创建的```myPackage.pth```文件放入这个路径，
![](https://github.com/zhangshd/test/blob/master/%E7%A4%BA%E4%BE%8B%E5%9B%BE%E7%89%87/Snipaste_2019-04-26_11-08-25.png)
以上操作的目的是把自己的脚本库路径加入到Python的环境变量中
