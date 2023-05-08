clear; close all; clc
segLen=1;
keyLen=2048*segLen;
times=0;
overhead=0;

load('./data_static_indoor_1.mat');
CSIa10rig=A(:,1);
CSIb10rig=A(:,2);
dataLen=length(CSIa10rig);

for staInd=1:keyLen:dataLen
    endInd=staInd+keyLen;
    % fprintf("range: %d %d",staInd,endInd);
    if endInd > length(CSIa10rig)
        break
    end
    times=times+1;
    CSIn10rig=normrnd(1,2,[dataLen,1]);
    seed=randi(10000);
    rng(seed);
    noise0rig=rand(dataLen,1).*3.*std(CSIa10rig);
    CSIa10rig=(CSIa10rig-mean(CSIa10rig))+noise0rig;
    CSIb10rig=(CSIb10rig-mean(CSIb10rig))+noise0rig;
    CSIn10rig=(CSIn10rig-mean(CSIn10rig))+noise0rig;

    tmpCSIa1=CSIa10rig(staInd:endInd-1);
    tmpCSIb1=CSIb10rig(staInd:endInd-1);
    [~,idx]=sort(tmpCSIa1);
    [~,tmpCSIa1Ind]=sort(idx);
    
    t1 = clock;
    [~,idx]=sort(tmpCSIb1);
    [~,tmpCSIb1Ind]=sort(idx);
    t2 = clock;

    minEpiIndClosenessLsb=zeros(keyLen/segLen);
    loop_time=keyLen/segLen;

    epiInda1_v=[];
    epiIndb1_h=[];
    a=[];
    b=[];
    epiInda1_v=reshape(tmpCSIa1Ind,[1,loop_time,segLen]);
    epiIndb1_h=reshape(tmpCSIb1Ind,[loop_time,1,segLen]);

    t3 = clock;
    a=repmat(epiInda1_v,loop_time,1,1);
    b=repmat(epiIndb1_h,1,loop_time,1);
    res=sum(abs(a-b),3);
    res_min=min(res);
    t4 = clock;

    % overhead = overhead + etime(t4,t1);
    overhead = overhead + etime(t2,t1) + etime(t4,t3);
end
overhead = overhead / times