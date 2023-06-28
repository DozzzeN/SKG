file_name = 'data_static_indoor_1';
load(strcat(file_name,'.mat'));
rem = 0;
l = length(A(:,1));
for i = 1:l
    if abs(A(i,1)-A(i,2)) >= 2
        for j = i:l-1
            A(j,1) = A(j+1,1);
            A(j,2) = A(j+1,2);
        end
        rem = rem + 1;
    end
end
A = A(1:l-rem-1,:);
save(strcat(file_name,'_r.mat'),'A');
fprintf(strcat(file_name,' done'));
