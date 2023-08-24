function b = arr_to_str(a)

a = string(a);
b = "[ ";
for i=1:length(a)-1
    b = strcat(b,a(i),", ");
end

b = strcat(b, a(end),"]");


