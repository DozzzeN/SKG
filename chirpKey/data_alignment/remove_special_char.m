function [output_str] = remove_special_char(input_str)

input_str = strrep(input_str, '(', ''); 
input_str = strrep(input_str, ')', ''); 
input_str = strrep(input_str, '#', ''); 
input_str = strrep(input_str, '{', ''); 
input_str = strrep(input_str, '}', ''); 
input_str = strrep(input_str, '[', ''); 
input_str = strrep(input_str, ']', ''); 

output_str = input_str;
end