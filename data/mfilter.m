file_name = 'data_static_indoor_1';
load(strcat(file_name,'.mat'));
l = length(A(:,1));
for i = 2:l-1
    if abs(A(i,1)-A(i-1,1)) >= 3
        A(i,1) = (A(i,1)+A(i-1,1)+A(i+1,1))/3;
    end
    if abs(A(i,2)-A(i-1,2)) >= 3
        A(i,2) = (A(i,2)+A(i-1,2)+A(i+1,2))/3;
    end
end
A = A(1:l,:);
save(strcat(file_name,'_r.mat'),'A');
fprintf(strcat(file_name,' done'));
