MyXlim = [0 1e-7];
MyYlim = [-1e-4 2e-4];

%-------------------------%
%    Pre-error fixing     %
%-------------------------%
fileID = fopen('ForceCurveGPU2017_04_08__18_52_46.txt','r');

[A,count] = fscanf(fileID, ['t: %e   ',...
                            'Fx1: %e   ',...
                            'Fy1: %e   ',...
                            'Fz1: %e   ',...
                            'Fx2: %e   ',...
                            'Fy2: %e   ',...
                            'Fz2: %e   ',...
                            'Fx3: %e   ',...
                            'Fy3: %e   ',...
                            'Fz3: %e \n',...
                            ]);
                        

    
fclose(fileID);

nl = numel(A)/10;
Amat = reshape(A,10,nl)';
        
t = Amat(:,1);
FZ1 = Amat(:,4);
FZ2 = Amat(:,7);
FZ3 = Amat(:,10);

FZ1smooth = smooth(FZ1,30);
figure(11)
subplot(1,2,1)
plot(t,FZ3,t,FZ2,t,FZ1,t,FZ1smooth);
legend('FZ3: Pressure Gradient Force','FZ2: Seperated Accel Force','FZ1: Average Accel Force','FZ1: Average Accel Force (Smoothed)')
grid on
xlim(MyXlim)
ylim(MyYlim)
title('Before Error Fixing')

%-------------------------%
%   After error fixing    %
%-------------------------%
fileID = fopen('ForceCurveGPU2017_04_08__19_24_15.txt','r');

[A,count] = fscanf(fileID, ['t: %e   ',...
                            'Fx1: %e   ',...
                            'Fy1: %e   ',...
                            'Fz1: %e   ',...
                            'Fx2: %e   ',...
                            'Fy2: %e   ',...
                            'Fz2: %e   ',...
                            'Fx3: %e   ',...
                            'Fy3: %e   ',...
                            'Fz3: %e \n',...
                            ]);
                        

    
fclose(fileID);

nl = numel(A)/10;
Amat = reshape(A,10,nl)';
        
t = Amat(:,1);
FZ1 = Amat(:,4);
FZ2 = Amat(:,7);
FZ3 = Amat(:,10);

FZ1smooth = smooth(FZ1,30);
figure(11)
subplot(1,2,2)
plot(t,FZ3,t,FZ2,t,FZ1,t,FZ1smooth);
legend('FZ3: Pressure Gradient Force','FZ2: Seperated Accel Force','FZ1: Average Accel Force','FZ1: Average Accel Force (Smoothed)')
grid on
xlim(MyXlim)
ylim(MyYlim)
title('After Error Fixing')

set(gcf,'outerposition',[ 54  108  1187  512])
