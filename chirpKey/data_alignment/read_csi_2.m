function [csi1, csi2, csi_time, Fs, start_time] = read_csi(filename)
    % Read txt line by line
    fid=fopen(filename);
    tline = fgetl(fid);
    
    csi_real1 = [];
    csi_image1 = [];
    csi_real2 = [];
    csi_image2 = [];
    
    csi_timestamp = [];

    ch = [];
    i = -2; 
    last = 0;
    while ischar(tline) 
        tline = fgetl(fid);

    %     % store the sof
        if mod(i+1,5)==0
            timestamp = str2num(char(tline));
            csi_timestamp = [csi_timestamp; timestamp];
        end

        % store the pre-equalized CSI
        if mod(i,5)==0
            % disp(tline)
            csi_entry = strsplit(tline,' ');

            % cache the current csi
            temp_csi_real = zeros(1, 64);
            temp_csi_image = zeros(1, 64);
            for subc = 1:64
                subc_entry = char(csi_entry(1,subc));
                subc_entry = strsplit(subc_entry,',');

                % Extract string of real and imaginary parts
                real_part = remove_special_char(char(subc_entry(1,1)));  
                image_part = remove_special_char(char(subc_entry(1,2)));

                % Conver the real and imaginary parts into string
                temp_csi_real(1, subc) = str2num(real_part);
                temp_csi_image(1, subc) = str2num(image_part);
            end
            csi_real1 = [csi_real1; temp_csi_real];
            csi_image1 = [csi_image1; temp_csi_image];
        end
        
        % store the pre-equalized CSI
        if mod(i,5)==1
            % disp(tline)
            csi_entry = strsplit(tline,' ');

            % cache the current csi
            temp_csi_real = zeros(1, 64);
            temp_csi_image = zeros(1, 64);
            for subc = 1:64
                subc_entry = char(csi_entry(1,subc));
                subc_entry = strsplit(subc_entry,',');

                % Extract string of real and imaginary parts
                real_part = remove_special_char(char(subc_entry(1,1)));  
                image_part = remove_special_char(char(subc_entry(1,2)));

                % Conver the real and imaginary parts into string
                temp_csi_real(1, subc) = str2num(real_part);
                temp_csi_image(1, subc) = str2num(image_part);
            end
            csi_real2 = [csi_real2; temp_csi_real];
            csi_image2 = [csi_image2; temp_csi_image];
        end

        i = i + 1; 
    end
    fclose(fid);

    % Ê±¼ä´Á
    dur = (max(csi_timestamp) - min(csi_timestamp))/1e6;
    start_time = min(csi_timestamp);
    % timestamp to relative second
    csi_time = csi_timestamp - start_time;
    % microseconds to second
    csi_time = csi_time/1e6;
    
    % CSI value
    csi1 = complex(csi_real1, csi_image1);
    csi2 = complex(csi_real2, csi_image2);
    Fs = size(csi1,1) / dur;
    
    %% Phase correction 
    % Flip the phase
    flip = [0,  0,  0,  0,  0,  0,  1,  1, -1, -1, 1,  1, -1,  1, -1,  1,  1,  1,  1,  1, 1, -1, -1,  1,  1, -1,  1, -1,  1,  1, 1,  1,  0,  1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1, -1, -1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1,  1,  1,  0, 0,  0,  0,  0];

    for i = 1:64
        if flip(i)==0
            csi1(:, i) = NaN;
            csi2(:, i) = NaN;
        elseif flip(i)==0
           continue
        else
            % CSI1
            amplitude = abs(csi1(:, i));
            phase1 = -angle(csi1(:, i)); 
            %phase = flip(i)*angle(csi(:, i)); 
            reconstruct_csi1 = complex (amplitude.*cos(phase1), amplitude.*sin(phase1));
            csi1(:, i) = reconstruct_csi1;
            
            %CSI2
            amplitude = abs(csi2(:, i));
            phase2 = -angle(csi2(:, i)); 
            %phase = flip(i)*angle(csi(:, i)); 
            reconstruct_csi2 = complex (amplitude.*cos(phase2), amplitude.*sin(phase2));
            csi2(:, i) = reconstruct_csi2;
        end 
    end
    csi1 = csi1(:, logical(abs(flip)));
    csi2 = csi2(:, logical(abs(flip)));
end