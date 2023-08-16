function [ SR1 ] = success_rate(error1,times,num)
    S1=[];
    for iii=1:num
        if sum(error1((iii-1)*times+1:iii*times))==0
           S1=[S1,1]; 
        else
           S1=[S1,0]; 
        end
    end
    SR1=sum(S1)/(length(S1));
end