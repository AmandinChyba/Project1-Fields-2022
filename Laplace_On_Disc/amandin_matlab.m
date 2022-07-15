x = 0;
y = 0;
z = 0;
for i=1:K+1
    x = [x, r(i)*cos(theta)];
    y = [y, r(i)*sin(theta)];
    z = [z, values_to_plot(1:K+1,i)'];
end

scatter3(x,y,z)