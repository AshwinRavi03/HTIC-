clc; clear; close all;

% Load your image
[file, path] = uigetfile({'*.jpg;*.png;*.bmp;*.tif'});
Img = imread(fullfile(path, file));
if size(Img,3) == 3
    Img = rgb2gray(Img);
end
Img = double(Img);
logI = log(Img + 1e-6);

% Ensure even dimensions
[X, Y] = size(logI);
if mod(X,2) ~= 0
    logI = logI(1:X-1,:); X = X-1;
end
if mod(Y,2) ~= 0
    logI = logI(:,1:Y-1); Y = Y-1;
end

% Step 1: Manual Haar DWT (row-wise)
temp = zeros(X, Y);
for i = 1:X
    k = 1;
    for j = 1:2:Y
        avg  = (logI(i,j) + logI(i,j+1))/2;
        diff = (logI(i,j) - logI(i,j+1))/2;
        temp(i,k) = avg;
        temp(i, k + Y/2) = diff;
        k = k + 1;
    end
end

% Step 2: Manual Haar DWT (col-wise)
coeffs = zeros(X, Y);
for j = 1:Y
    k = 1;
    for i = 1:2:X
        avg  = (temp(i,j) + temp(i+1,j))/2;
        diff = (temp(i,j) - temp(i+1,j))/2;
        coeffs(k,j) = avg;
        coeffs(k + X/2, j) = diff;
        k = k + 1;
    end
end

% Frequency separation
LL = coeffs(1:X/2, 1:Y/2);      % Low Frequency: Illumination
LH = coeffs(1:X/2, Y/2+1:end);  % Horizontal
HL = coeffs(X/2+1:end, 1:Y/2);  % Vertical
HH = coeffs(X/2+1:end, Y/2+1:end); % Diagonal

high_freq = zeros(size(coeffs));
high_freq(1:X/2, Y/2+1:end) = LH;
high_freq(X/2+1:end, 1:Y/2) = HL;
high_freq(X/2+1:end, Y/2+1:end) = HH;

% Wiener filtering for smoother illumination and reflectance
illumination_filtered = wiener2(LL, [5 5]); % smooth low frequency
reflectance_filtered  = wiener2(mat2gray(high_freq), [3 3]); % details

% Manual Inverse Haar transform (col-wise)
temp_rec = zeros(X, Y);
for j = 1:Y
    for k = 1:X/2
        avg = coeffs(k,j);
        diff = coeffs(k + X/2, j);
        temp_rec(2*k-1, j) = avg + diff;
        temp_rec(2*k, j)   = avg - diff;
    end
end

% Manual Inverse Haar transform (row-wise)
reconstructed = zeros(X, Y);
for i = 1:X
    for k = 1:Y/2
        avg = temp_rec(i,k);
        diff = temp_rec(i, k + Y/2);
        reconstructed(i, 2*k-1) = avg + diff;
        reconstructed(i, 2*k)   = avg - diff;
    end
end

logR = reconstructed;
R_hat = exp(logR);
R_hat = mat2gray(R_hat);

% Display results
figure('Name','Manual Haar DWT + Wiener Frequency Separation');
subplot(2,3,1); imshow(uint8(Img)); title('Original');
subplot(2,3,2); imshow(mat2gray(LL)); title('Low Freq (Illumination)');
subplot(2,3,3); imshow(mat2gray(illumination_filtered)); title('Wiener Illumination');
subplot(2,3,4); imshow(mat2gray(high_freq)); title('High Freq (Reflectance)');
subplot(2,3,5); imshow(mat2gray(reflectance_filtered)); title('Wiener Reflectance');
subplot(2,3,6); imshow(R_hat, []); title('Final Reflectance');
