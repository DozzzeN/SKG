clear

%% Load data
% load the initial measurement matrix
load('A0h-gau.mat');
% load the initial channel measurements
rss = load('rss.csv');

%% Filter
order = 5;
framelen = 11;

rss = sgolayfilt(rss',order,framelen);

%% Key delivery
rss_normal=zscore(rss')./35;

M=20;
N=40;
error1=[];
error2=[];
error3=[];
error4=[];
bits_all_alice=[];
bits_all_bob=[];
bits_all_eve=[];
reuse_time=10;

for p=1:(floor(length(rss_normal(:,1:240))/N))
    

    T1=CirculantMtx(N,N,rss_normal(1,(p-1)*N+1:p*N));
    T2=CirculantMtx(N,N,rss_normal(2,(p-1)*N+1:p*N));
    T3=CirculantMtx(N,N,rss_normal(3,(p-1)*N+1:p*N));
    for pp=1:reuse_time
        h0=randi(2,8,1)-1;
        h_new=[];
        mask_h=[];
        for s=1:length(h0)
            h_new=[h_new,h0(s)];
            mask_h=[mask_h,1];
            h_new=[h_new,0];
            h_new=[h_new,0];
            h_new=[h_new,0];
            h_new=[h_new,0];
            mask_h=[mask_h,0];
            mask_h=[mask_h,0];
            mask_h=[mask_h,0];
            mask_h=[mask_h,0];
        end
        h=h_new';
        mask_h=mask_h';
        permut=randperm(40);
        T1 = T1(permut,:);
        T2 = T2(permut,:);
        permut1=randperm(20);
        A1=A0(permut1,:);
        K=10;                                  %support size
        sA=.1/M;                             %A noise variance
        sb=sA;                                %b noise variance
        lam=.02;                              %regularization parameter
        ni=30;                               %no. of iterations
        nr=30;                               %no. of Monte Carlo runs

        m21=zeros(ni,1);                      %mean-square error
        m22=zeros(ni,1);                      %mean-square error
        m21_eve=zeros(ni,1);                      %mean-square error
        m22_eve=zeros(ni,1);                      %mean-square error
        estimated_E={};
        noise_E={};
        x_solution1={};
        x_solution2={};
        x_solution1_eve={};
        x_solution2_eve={};
        for ii=1:nr
            ii;
            use_E=randn(M,N);
            noise_E=[noise_E,use_E];
            
            A=(A1*(T2+eye(N)))  +sqrt(sA)*use_E*0;       %noisy A matrix
            b=(A1*(T1+eye(N)))*h+sqrt(sb)*randn(M,1)*0;       %noisy b vector
            A_eve=(A0*(T3+eye(N)))  +sqrt(sA)*use_E*0; 

            [e21,~,~,solution1,E]=adm_cd_stls_f(A,b,M,N,K,lam,h,ni);
            [e23,~,~,solution2]=ass_pg_stls_f(A,b,N,K,lam,h,ni);
            x_solution1=[x_solution1,solution1];
            x_solution2=[x_solution2,solution2];
            estimated_E=[estimated_E,E];
            m21=m21+e21;
            m22=m22+e23;

            [e21,~,~,solution1_eve,E]=adm_cd_stls_f(A_eve,b,M,N,K,lam,h,ni);
            [e23,~,~,solution2_eve]=ass_pg_stls_f(A_eve,b,N,K,lam,h,ni);
            x_solution1_eve=[x_solution1_eve,solution1_eve];
            x_solution2_eve=[x_solution2_eve,solution2_eve];
            m21_eve=m21_eve+e21;
            m22_eve=m22_eve+e23;
        end
        m21=m21/nr;
        m22=m22/nr;
        m21_eve=m21_eve/nr;
        m22_eve=m22_eve/nr;

        temp1=threhold(solution1);
        temp1_eve=threhold(solution1_eve);
        temp2=threhold(solution2);
        temp2_eve=threhold(solution2_eve);
        error1=[error1,sum(abs((temp1-h).*mask_h))];
        error2=[error2,sum(abs((temp1_eve-h).*mask_h))];
        error3=[error3,sum(abs((temp2-h).*mask_h))];
        error4=[error4,sum(abs((temp2_eve-h).*mask_h))];
        bits_all_alice=[bits_all_alice,h'];
        bits_all_bob=[bits_all_bob,temp2'];
        bits_all_eve=[bits_all_eve,temp2_eve'];
    end
end
BAR1=1-sum(error1)/(8*length(error1));
BAR2=1-sum(error2)/(8*length(error2));
BAR3=1-sum(error3)/(8*length(error3));
BAR4=1-sum(error4)/(8*length(error4));
SR1 = success_rate(error1,reuse_time,6);
SR2 = success_rate(error2,reuse_time,6);
SR3 = success_rate(error3,reuse_time,6);
SR4 = success_rate(error4,reuse_time,6);
figure(1);
surf(T1)
figure(2);
surf(T2)
figure(3);
surf(T3)

%% Error correction
extract_key_Alice=[];
extract_key_Bob=[];
extract_key_Eve=[];
for i=1:length(bits_all_alice)
    if mod(i,5)==1
        extract_key_Alice=[extract_key_Alice,bits_all_alice(i)];
        extract_key_Bob=[extract_key_Bob,bits_all_bob(i)];
        extract_key_Eve=[extract_key_Eve,bits_all_eve(i)];
    end
end
mismatch_bob=double(xor(extract_key_Alice,extract_key_Bob));
mismatch_eve=double(xor(extract_key_Alice,extract_key_Eve));

bits_all_mismatch_alice_bob=[];
bits_all_mismatch_alice_eve=[];
bits_all_mismatch_bob=[];
bits_all_mismatch_eve=[];
for pp=1:12
    h_bob=mismatch_bob((pp-1)*40+1:pp*40)';
    h_eve=mismatch_eve((pp-1)*40+1:pp*40)';
    K=10;                                  %support size
    permut=randperm(40);
    T1 = T1(permut,:);
    T2 = T2(permut,:);
    permut1=randperm(20);
    A1=A0(permut1,:);
    sA=.1/M;                             %A noise variance
    sb=sA;                                %b noise variance
    lam=.02;                              %regularization parameter
    ni=30;                               %no. of iterations
    nr=30;                               %no. of Monte Carlo runs

    m21=zeros(ni,1);                      %mean-square error
    m22=zeros(ni,1);                      %mean-square error
    m21_eve=zeros(ni,1);                      %mean-square error
    m22_eve=zeros(ni,1);                      %mean-square error
    estimated_E={};
    noise_E={};
    x_solution1={};
    x_solution2={};
    x_solution1_eve={};
    x_solution2_eve={};
    for ii=1:nr
        ii;
        use_E=randn(M,N);
        noise_E=[noise_E,use_E];

        A=(A1*(T2+eye(N)))  +sqrt(sA)*use_E*0;       %noisy A matrix
        b=(A1*(T1+eye(N)))*h_bob+sqrt(sb)*randn(M,1)*0;       %noisy b vector
        A_eve=(A0*(T3+eye(N)))  +sqrt(sA)*use_E*0; 
        [e21,~,~,solution1,E]=adm_cd_stls_f(A,b,M,N,K,lam,h_bob,ni);
        [e23,~,~,solution2]=ass_pg_stls_f(A,b,N,K,lam,h_bob,ni);
        x_solution1=[x_solution1,solution1];
        x_solution2=[x_solution2,solution2];
        estimated_E=[estimated_E,E];
        m21=m21+e21;
        m22=m22+e23;
       
        [e21,~,~,solution1_eve,E]=adm_cd_stls_f(A_eve,b,M,N,K,lam,h_eve,ni);
        [e23,~,~,solution2_eve]=ass_pg_stls_f(A_eve,b,N,K,lam,h_eve,ni);
        x_solution1_eve=[x_solution1_eve,solution1_eve];
        x_solution2_eve=[x_solution2_eve,solution2_eve];
        m21_eve=m21_eve+e21;
        m22_eve=m22_eve+e23;
    end
    m21=m21/nr;
    m22=m22/nr;
    m21_eve=m21_eve/nr;
    m22_eve=m22_eve/nr;

    temp1=threhold(solution1);
    temp1_eve=threhold(solution1_eve);
    temp2=threhold(solution2);
    temp2_eve=threhold(solution2_eve);
    bits_all_mismatch_alice_bob=[bits_all_mismatch_alice_bob,h_bob'];
    bits_all_mismatch_alice_eve=[bits_all_mismatch_alice_eve,h_eve'];
    bits_all_mismatch_bob=[bits_all_mismatch_bob,temp2'];
    bits_all_mismatch_eve=[bits_all_mismatch_eve,temp2_eve'];
end

%% Calculate final mismatches
sum(xor(bits_all_mismatch_alice_bob,bits_all_mismatch_bob))
sum(xor(bits_all_mismatch_alice_eve,bits_all_mismatch_eve))