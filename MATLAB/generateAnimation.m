function [] = generateAnimation(P,dt, strideLength)
figure()
Px1 = [zeros(size(P,1),1) P(:,[1 3 5])];
Py1 = [zeros(size(P,1),1) P(:,[2 4 6])]; 

Px2 = P(:,[3 7 9]);
Py2 = P(:,[4 8 10]); 

Hl=line(Px1(1,:), Py1(1,:)); hold on
H2=line(Px2(1,:), Py2(1,:)); hold on
axis equal
Hl=handle(Hl);
xlim([-0.6 3.6])
ylim([-0.1 1.7])
xlabel('(m)')
ylabel('(m)')

for i = 1:7
for j=1:size(P,1)
    Hl.XData=Px1(j,:);
    Hl.YData=Py1(j,:);
    H2.XData=Px2(j,:);
    H2.YData=Py2(j,:);    
    pause(dt)
end
Px1 = Px1 + strideLength;
Px2 = Px2 + strideLength;

end
