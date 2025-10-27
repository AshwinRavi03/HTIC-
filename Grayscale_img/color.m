clc; clear; close all;

% Load color image 
[file, path] = uigetfile({'*.jpg;*.png;*.bmp;*.tif'}, 'Select Image');
color_img = imread(fullfile(path, file));
color_img = double(color_img);

% Ensure even dimensions for all channels
[X, Y, chans] = size(color_img);
if mod(X,2)~=0
    color_img = color_img(1:X-1,:,:); X=X-1;
end
if mod(Y,2)~=0
    color_img = color_img(:,1:Y-1,:); Y=Y-1;
end

% Output storage
R_hatColor = zeros(X, Y, 3);
illumination_out = zeros(X/2, Y/2, 3);

for c = 1:3
    img_chan = color_img(:,:,c);
    img_log = log(img_chan + 1e-6);

    % ---------------- Manual Haar DWT (row-wise) ------------------
    temp = zeros(X, Y);
    for i = 1:X
        k = 1;
        for j = 1:2:Y
            avg  = (img_log(i,j) + img_log(i,j+1))/2;
            diff = (img_log(i,j) - img_log(i,j+1))/2;
            temp(i,k) = avg;
            temp(i, k + Y/2) = diff;
            k = k + 1;
        end
    end

    % ---------------- Manual Haar DWT (col-wise) ------------------
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
    LL = coeffs(1:X/2,1:Y/2);      % Illumination
    LH = coeffs(1:X/2,Y/2+1:end);
    HL = coeffs(X/2+1:end,1:Y/2);
    HH = coeffs(X/2+1:end,Y/2+1:end);

    high_freq = zeros(size(coeffs));
    high_freq(1:X/2,Y/2+1:end) = LH;
    high_freq(X/2+1:end,1:Y/2) = HL;
    high_freq(X/2+1:end,Y/2+1:end) = HH;

    % ---------------- Wiener filtering ----------------
    illumination_filtered = wiener2(LL, [5 5]);
    reflectance_filtered  = wiener2(mat2gray(high_freq), [3 3]);

    illumination_out(:,:,c) = illumination_filtered;

    % ---------------- Inverse Haar Transform (col then row) ----------------
    temp_rec = zeros(X, Y);
    for j = 1:Y
        for k = 1:X/2
            avg = coeffs(k,j);
            diff = coeffs(k + X/2, j);
            temp_rec(2*k-1, j) = avg + diff;
            temp_rec(2*k, j)   = avg - diff;
        end
    end

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
    R_hatColor(:,:,c) = exp(logR);
end

% ---------------- Color Ratio Preservation ----------------
% Normalize to maintain color ratios in final reflectance
R_hatColor = R_hatColor ./ (repmat(sum(R_hatColor, 3), [1 1 3]) + 1e-6) .* repmat(sum(color_img, 3), [1 1 3]);
R_hatColor = mat2gray(R_hatColor);

% ---------------- Visualization ----------------
figure('Name','Color DWT, Spectral Illumination & Ratio-Preserved Reflectance');
subplot(2,3,1); imshow(uint8(color_img)); title('Original Color');
subplot(2,3,2); imshow(mat2gray(illumination_out(:,:,1))); title('Illumination - Red');
subplot(2,3,3); imshow(mat2gray(illumination_out(:,:,2))); title('Illumination - Green');
subplot(2,3,4); imshow(mat2gray(illumination_out(:,:,3))); title('Illumination - Blue');
subplot(2,3,5); imshow(R_hatColor); title('Reflectance (Ratio-Preserved)');
subplot(2,3,6); imshow(mean(R_hatColor,3)); title('Reflectance (Grayscale Composite)');
