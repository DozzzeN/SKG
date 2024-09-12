%% 数据归一化,获取最终的testdata
function [testdata] = normalization(csi_a,csi_b)
    csi_value1 = csi_a;
    csi_value2 = csi_b;

    min_csi = min(csi_value1);
    max_csi = max(csi_value1);
    csi_value1 = (csi_value1-min_csi)/(max_csi-min_csi);
    
    min_csi = min(csi_value2);
    max_csi = max(csi_value2);
    csi_value2 = (csi_value2-min_csi)/(max_csi-min_csi);

    testdata = [];
    testdata(:,1) = csi_value1;
    testdata(:,2) = csi_value2;
end