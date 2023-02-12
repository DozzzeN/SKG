segLen=7;
keyLen=256*segLen;
times=0;
overhead=0;

load('./data_static_indoor_1.mat');
CSIa10rig=A(:,1);
CSIb10rig=A(:,2);
dataLen=length(CSIa10rig);

% poolSize=4;
% parpool("local",poolSize);

for staInd=1:keyLen:dataLen
    endInd=staInd+keyLen;
    fprintf("range: %d %d",staInd,endInd);
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

    tmpCSIa1=CSIa10rig(staInd:endInd);
    tmpCSIb1=CSIb10rig(staInd:endInd);
    [out,idx]=sort(tmpCSIa1);
    [out,tmpCSIa1Ind]=sort(idx);
%     tic;
    t1 = clock;
    [out,idx]=sort(tmpCSIb1);
    [out,tmpCSIb1Ind]=sort(idx);

    minEpiIndClosenessLsb=zeros(keyLen/segLen);
    loop_time=keyLen/segLen;
    for i=0:loop_time-1
        epiInda1=tmpCSIa1Ind(i*segLen+1:(i+1)*segLen);
        epiIndClosenessLsb=zeros(loop_time,1);
        for j=0:loop_time-1
            epiIndb1=tmpCSIb1Ind(j*segLen+1:(j+1)*segLen);
            epiIndClosenessLsb(j+1)=sum(abs(epiIndb1-epiInda1));
        end
        [~,argmin]=min(epiIndClosenessLsb);
        minEpiIndClosenessLsb(i+1)=argmin;
    end
%     toc
    t2 = clock;
    overhead = overhead + etime(t2,t1);
end

overhead = overhead / times;