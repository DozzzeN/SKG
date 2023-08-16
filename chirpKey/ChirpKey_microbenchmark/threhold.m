function [new]= threhold(mismatch)  
    thre1=0.5;
    thre2=-0.5;    
    for i=1:length(mismatch)
        if mismatch(i)>thre1
            mismatch(i)=1;
        elseif mismatch(i)<thre2
            mismatch(i)=1;
        else
            mismatch(i)=0;
        end
    end
    new=mismatch;
end