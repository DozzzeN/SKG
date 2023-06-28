file_name1 = 'CSIa';
load(strcat(file_name1,'.mat'));
file_name2 = 'CSIb';
load(strcat(file_name2,'.mat'));
rem = 0;
l = length(CSIa.');
for i = 1:l
    if abs(CSIa(i)-CSIb(i)) >= 2
        for j = i:l-1
            CSIa(j) = CSIa(j+1);
            CSIb(j) = CSIb(j+1);
        end
        rem = rem + 1;
    end
end
CSIa = CSIa(1:l-rem-1);
CSIb = CSIb(1:l-rem-1);
save(strcat(file_name1,'_r.mat'),'CSIa');
save(strcat(file_name2,'_r.mat'),'CSIb');
fprintf(strcat(file_name1,' done'));
