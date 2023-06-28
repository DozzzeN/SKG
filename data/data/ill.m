clear all; close all; clc
carrier = 22;

filename = "1-12-4";
data1=read('data/' + filename + '.dat');
ind1=[1:length(data1)];
cell1=data1(ind1);
csi1=[];
time1=[];
for i=1:length(data1)
    if length(cell1{i}.csi)<=1
        csi1=[csi1,0];
    else
        csi1=[csi1,abs(cell1{i}.csi(carrier))];
    end
    time1=[time1,cell1{i}.timestamp];
end
figure(1)
plot(ind1, csi1.');

data2=read('data/' + filename + '-router.dat');
ind2=[1:length(data2)];
cell2=data2(ind2);
csi2=[];
time2=[];
for i=1:length(data2)
    if length(cell2{i}.csi)<=1
        csi2=[csi2,0];
    else
        csi2=[csi2,abs(cell2{i}.csi(carrier))];
    end
    time2=[time2,cell2{i}.timestamp];
end
figure(2)
plot(ind2, csi2.');

save(filename+'_1.mat', 'csi1');
save(filename+'_2.mat', 'csi2');