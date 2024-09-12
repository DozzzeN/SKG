### 1 解析CSI数据

```matlab
% 解析csi数据
read_csi.m

% 去除txt文件中的特殊字符
remove_special_char.m

% 解析后的数据可视化
view_data_text_only.m


% 以秒为单位查看某一段CSI的数据
view_data_select.m

% 去掉噪声后地csi数据
view_data_nonnoise.m
```

### 2 long training seq的初步设计

``` matlab
select.m
select1.m
select2.m
```

### 3 long training seq的插值设计

```matlab
interpolation.m
% 输入为x、y两个数组，分别是已知的横坐标和纵坐标
% new_x是需要插值的横坐标点
```

### 4 拼接两个CSI中的有效数据

```matlab
joint.m
joint_circulation.m
```

### 5 获取ping的CSI数据，消去自己的信号，计算相关系数

```matlab
parameter.m

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%双向通信中，去除自己的信号，接收到的信号的csi数据，并计算通信双方csi数据的相关性
parameter_trx.m

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%双向通信中，物理层随机修改了long training seq的值，根据每个序列相对于原始序列的csi的商值，进行信号的还原，并计算通信双方的csi的相关性。
parameter_trx_modify.m

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 192.168.123.1发送的long training seq不变，192.168.123.2发送的变化
% 将192.168.123.2获取的原始数据经过处理后获得与192.168.123.1端相关性高的数据
parameter_trx_modify1.m

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 获取192.168.123.1接收到的变化的csi 相对于 192.168.123.2接收到的原始csi的商值
get_ralative_seq.m

```

### 6 消去自己的信号后，还需要过滤相关性太低的数据，归一化数据，拼接为一列，获得最终的testdata

```matlab
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 将csi_value_result中无用的相关性太低数据过滤掉（根据p_result中较低的值），拼接数据
% 已经获取了通信双方相关性高的数据（parameter_trx_modify1.m）
remove_low_correlation.m

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 数据归一化,获取最终的testdata
normalization.m
```



