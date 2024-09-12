function [SBR,BMR,Entropy,Alist,AlistKey] = quantification(alpha,testdata)
    p = 40;
%     alpha = 0.3;

    t1 = cputime;

    CSIa1Orig = [(1:length(testdata(:,1)))' testdata(:,1)];
    CSIb1Orig = [(1:length(testdata(:,2)))' testdata(:,2)];


    [PKS_a,LOCS_a] = findpeaks(CSIa1Orig(:,2),'MinPeakDistance',p);
    LOCS_a = [LOCS_a;length(CSIa1Orig(:,2))];
    [PKS_b,LOCS_b] = findpeaks(CSIb1Orig(:,2),'MinPeakDistance',p);
    LOCS_b = [LOCS_b;length(CSIb1Orig(:,2))];

    a_list_bit = [];
    b_list_bit = [];

    for i = 1:length(LOCS_a)-1
        Ha1 = CSIa1Orig(LOCS_a(i):LOCS_a(i+1),:);
        len_current = length(Ha1);

        while length(Ha1) > 0
            len_last = len_current;
            CSIa1 = Ha1;
            Ha1 = [];

            maxa1 = max(CSIa1(:,2));
            mina1 = min(CSIa1(:,2));
            qa1 = mean(CSIa1(:,2)) + alpha*(maxa1+mina1);
            qa0 = mean(CSIa1(:,2)) - alpha*(maxa1-mina1);

            for i = 1:size(CSIa1,1)
                if CSIa1(i,2) >= qa1
                    a_list_bit(1,CSIa1(i,1)) = CSIa1(i,1);
                    a_list_bit(2,CSIa1(i,1)) = 1;
                elseif CSIa1(i,2) <= qa0
                    a_list_bit(1,CSIa1(i,1)) = CSIa1(i,1);
                    a_list_bit(2,CSIa1(i,1)) = 0;
                else
                    Ha1 = [Ha1;CSIa1(i,1) CSIa1(i,2)];

                end
            end
            len_current = length(Ha1);

            if(len_current == len_last)
                break
            end
        end
    end
    drop_a = length(find(a_list_bit(1,:) == 0));

    for i = 1:length(LOCS_b)-1
        Hb1 = CSIb1Orig(LOCS_b(i):LOCS_b(i+1),:);
        len_current = length(Hb1);

        while length(Hb1) > 0
            len_last = len_current;
            CSIb1 = Hb1;
            Hb1 = [];

            maxb1 = max(CSIb1(:,2));
            minb1 = min(CSIb1(:,2));
            qb1 = mean(CSIb1(:,2)) + alpha*(maxb1+minb1);
            qb0 = mean(CSIb1(:,2)) - alpha*(maxb1-minb1);

            for i = 1:size(CSIb1,1)
                if CSIb1(i,2) >= qb1
                    b_list_bit(1,CSIb1(i,1)) = CSIb1(i,1);
                    b_list_bit(2,CSIb1(i,1)) = 1;
                elseif CSIb1(i,2) <= qb0
                    b_list_bit(1,CSIb1(i,1)) = CSIb1(i,1);
                    b_list_bit(2,CSIb1(i,1)) = 0;
                else
                    Hb1 = [Hb1;CSIb1(i,1) CSIb1(i,2)];

                end
            end
            len_current = length(Hb1);

            if(len_current == len_last)
                break
            end
        end
    end
    drop_b = length(find(b_list_bit(1,:) == 0));

    bitCorrect = 0;
    bitSum = 0;
    a_total_bit = [];
    b_total_bit = [];
    for i = 1:min(size(a_list_bit,2),size(b_list_bit,2))
        if(a_list_bit(1,i) ~= 0 && b_list_bit(1,i) ~= 0)
            bitSum = bitSum + 1;
            a_total_bit = [a_total_bit;a_list_bit(2,i)];
            b_total_bit = [b_total_bit;b_list_bit(2,i)];
            if(a_list_bit(2,i) == b_list_bit(2,i))
                bitCorrect = bitCorrect + 1;
            end
        end
    end

    t2 = cputime;

    totalpacket = size(a_list_bit,2);

    BMR = round(bitCorrect / bitSum, 10);
    disp(['比特匹配率   ', num2str(bitCorrect), '/', num2str(bitSum),...
            '=', num2str(BMR)]);
    SBR = bitCorrect/totalpacket;
    disp(['比特生成率   ', num2str(bitCorrect), '/', num2str(totalpacket),...
        '=', num2str(SBR)])
    
    a_list_bit = a_list_bit(2,:);
    b_list_bit = b_list_bit(2,:);
    len = min([length(a_list_bit),length(b_list_bit)]);
    keyLen = 16;  % 密钥比特数
    keyNum = fix(len/keyLen);
    a_list_bit(keyLen*keyNum+1:end)=[];
    b_list_bit(keyLen*keyNum+1:end)=[];
    a_list_key = reshape(a_list_bit, keyLen, keyNum);
    b_list_key = reshape(b_list_bit, keyLen, keyNum);
    keySum = 0;

    keyCorrect = 0;
    for i = 1:keyNum
        if isequal(a_list_key(:,i),b_list_key(:,i))
            keyCorrect = keyCorrect + 1;
            keySum = keySum + 1;
        else
            keySum = keySum + 1;
        end
    end

    KMR = round(keyCorrect / keySum, 10);
    disp(['密钥匹配率   ', num2str(keyCorrect), '/', num2str(keySum),...
            '=', num2str(KMR)]);
    % ---------------------------------------------    
    pa1 = length(find(a_total_bit==1))/length(a_total_bit);
    pa0 = length(find(a_total_bit==0))/length(a_total_bit);
    EntropyA = -(pa1*log(pa1)/log(2)+pa0*log(pa0)/log(2));

    pb1 = length(find(b_total_bit==1))/length(b_total_bit);
    pb0 = length(find(b_total_bit==0))/length(b_total_bit);
    EntropyB = -(pb1*log(pb1)/log(2)+pb0*log(pb0)/log(2));

    disp(['a熵:   ', num2str(EntropyA)]);
    disp(['b熵:   ', num2str(EntropyB)]);
    Entropy = (EntropyA + EntropyB) / 2;
    Alist = a_total_bit;
    AlistKey = a_list_key;
%     appEn = approx_entropy(5,0.5,a_list_bit);
%     disp(['近似熵:   ', num2str(appEn)]);

    disp(['运行时间   ', num2str((t2-t1) / 2 / keyNum * 1000)]);
end