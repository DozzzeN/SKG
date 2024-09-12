function [csi, csi_time, Fs, start_time] = read_csi(filename)
    % Read txt line by line
    fid=fopen(filename);
    tline = fgetl(fid);
    csi_real = [];
    csi_image = [];
    csi_timestamp = [];

    ch = [];
    i = -2; 
    last = 0;
    while ischar(tline) 
        tline = fgetl(fid);

    %     % store the sof
        if mod(i+1,4)==0
            timestamp = str2num(char(tline));
        end

        % store the pre-equalized CSI
        if mod(i,4)==0 || i==0
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
             csi_real = [csi_real; temp_csi_real];
             csi_image = [csi_image; temp_csi_image];
             csi_timestamp = [csi_timestamp; timestamp];
            
        end

        i = i + 1; 
    end
    fclose(fid);

    csi = complex(csi_real, csi_image);
    dur = (max(csi_timestamp) - min(csi_timestamp))/1e6;
    Fs = size(csi,1) / dur;
    start_time = min(csi_timestamp);
    % timestamp to relative second
    csi_time = csi_timestamp - start_time;
    % microseconds to second
    csi_time = csi_time/1e6;
    
    %% Phase correction 
    % Flip the phase
    flip = [0,  0,  0,  0,  0,  0,  1,  1, -1, -1, 1,  1, -1,  1, -1,  1,  1,  1,  1,  1, 1, -1, -1,  1,  1, -1,  1, -1,  1,  1, 1,  1,  0,  1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1, -1, -1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1,  1,  1,  0, 0,  0,  0,  0];

    for i = 1:64
        if flip(i)==0
            csi(:, i) = NaN;
        elseif flip(i)==0
           continue
        else
            amplitude = abs(csi(:, i));
            phase = -angle(csi(:, i)); 
            %phase = flip(i)*angle(csi(:, i)); 
            reconstruct_csi = complex (amplitude.*cos(phase), amplitude.*sin(phase));
            csi(:, i) = reconstruct_csi;
        end 
    end
    csi = csi(:, logical(abs(flip)));
    
end