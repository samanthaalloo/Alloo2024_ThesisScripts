%% This code will calculate the effective speckle size of a given input sample image. 
% Written by Samantha Jane Alloo (University of Canterbury, New Zealand)
% Contains ideas published in the doctoral thesis and Haibo Lin's Matlab code published in his Dissertation
% "Speckle Mechanism in Holographic Optical Coherence Imaging", 
% 2009, University of Missouri. 
close all
FF = 1%double(imread('')); % flat-feild image 
DC = 0%double(imread('')); % dark-current image
speckle = double(imread('2xP40_1xP80_1xP120.tif')); % reference-speckle image 
image = (speckle - DC )./(FF - DC);
image = image(1:750,1:750); % needs to be cropped to a square image

[rows, columns] = size(image);
pixel_size = 12.3 %12.3; % pixel size in microns 

figure(1) 
imshow(image,[]); % show the image 
colorbar()
%title('Speckle Image')


% Establishing an array to save the data in 
row_t = image(1,:); % select one array of data - test 
s_r = size(xcov(row_t)); % get the size of AC serial - test 
AC_r = double(zeros(s_r)); % generate a zero serial with the same size with AC serial with double character

% Calculating the Autocovariance (AC): Horizontal Direction <-> 
for i = 1:rows % iterating down all the rows  
r = image(i,:); % selecting row of 
AC_r = imadd(AC_r, xcov(r,'coeff')); % Do AC and add it to the zero serial: here, we add because eventually we divide by the number of rows to get average 
end; 
AC_hor = AC_r/rows; % Normalising the addition: i.e. averaging 


figure(2) 
plot(AC_hor); % Plot the normalized AC
%title('Autocovariance of Speckles in Horizontal Direction')
xlabel('Pixels')
ylabel('Autocovariance')
axis([0, 2*columns, -0.2, 1.1])

% Calculating the Autocovariance: Vertical Direction ^-
col_t = image(:,1); % Same process to calculate  normalized AC to column 
s_c = size(xcov(col_t)); 
AC_c = double(zeros(s_c)); 

for j = 1:columns 
c = image(:,j); 
AC_c = imadd(AC_c, xcov(c,'coeff')); 
end; 
AC_ver=AC_c/rows;

figure(3)
plot(AC_ver); 
%title('Autocovariance of Speckles in Vertical Direction')
xlabel('Pixels')
ylabel('Autocovariance')
axis([0, 2*rows, -0.2, 1.1])

% Fitting a Gaussian Function to the plots to get the FWHM 
gauss = @(p,x)p(1)*exp(-((x-p(2))/p(3)).^2); % the model 

% Horizontal Speckle Size Fitting
pix = 0:1:2*columns-2;
startingVals = [1,columns,10];
fitresult_h = fitnlm(pix,AC_hor,gauss,startingVals);
coeffs_h = fitresult_h.Coefficients(:,1);
coeffs_h = coeffs_h.(1);
p_1_h = coeffs_h(1);
p_2_h = coeffs_h(2);
p_3_h = coeffs_h(3);

fit_h =  p_1_h*exp(-((pix-p_2_h)/p_3_h).^2); 

figure(4)
hold on
plot(pix, fit_h,'b')
plot(pix, AC_hor,'--','MarkerEdgeColor','b')
legend('Fit','Raw Data')
%title('Fit: Autocovariance of Speckles in Horizontal Direction')
xlabel('Pixels')
ylabel('Autocovariance')
axis([0, 2*columns, -0.2, 1.1])
hold off

% Vertical Speckle Size Fitting
pix = 0:1:2*rows-2;
startingVals = [1,rows,10];
fitresult_v = fitnlm(pix,AC_ver,gauss,startingVals);
coeffs_v = fitresult_v.Coefficients(:,1);
coeffs_v = coeffs_v.(1);
p_1_v = coeffs_v(1);
p_2_v = coeffs_v(2);
p_3_v = coeffs_v(3);

fit_v =  p_1_v*exp(-((pix-p_2_v)/p_3_v).^2); 

figure(5)
hold on
plot(pix, fit_v,'b')
plot(pix, AC_ver,'--','MarkerEdgeColor','b')
legend('Fit','Raw Data')
%title('Fit: Autocovariance of Speckles in Vertical Direction')
xlabel('Pixels')
ylabel('Autocovariance')
axis([0, 2*rows, -0.2, 1.1])
hold off

% By definition of our Gaussian model, and using the relationship between
% standard deviation and FWHM the speckle size in each direction is FWHM =
% 2*sqrt(log(2)) * p_3
seff_h = 2*sqrt(log(2)) * p_3_h * pixel_size
seff_v = 2*sqrt(log(2)) * p_3_v * pixel_size

seff_microns = 0.5*(seff_v+seff_h)   % adding in square to get overall effective 



saveas(figure(1),'2xP40_1xP80_1xP120SpeckleImage.png')
saveas(figure(4),'2xP40_1xP80_1xP120HorizontalAutoCovar.png')
saveas(figure(5),'2xP40_1xP80_1xP120VerticallAutoCovar.png')