% Project 1 - Annotation Transfer
% The code is to transfer the annotations from H&E images to the corresponding MUSE images
% Author: Tianling Niu; Date Revised: Oct 2026

clear all;
close all; clc;

% load fixed and moving images
moving = imread("moving.jpg");
fixed = imread("fixed.jpg");
% select control points
cpselect(fixed,moving);
% save the movingPoints and fixedPoints for registration

% load pre-selected point pairs
%load fixedPoints.mat; load movingPoints.mat; 

%% Registration
tform = fitgeotform2d(movingPoints,fixedPoints,"polynomial",2); 
Rfixed = imref2d(size(fixed));
registered1 = imwarp(moving,tform,FillValues=0,OutputView=Rfixed);
figure(1); imshow(registered1);

% keep the black background
rep = find(registered1(:,:,:) == 255);
registered1(rep) = 0;
figure(2); imshow(registered1);

%% outline extraction
HEw = imread("fixed_annotation.jpg");
HE = imread("fixed.jpg");
outline1 = rgb2gray(HE-HEw);
imagesc(outline1); colormap gray;
%figure; imshowpair(registered1,outline1,"blend");

% rep = find(outline1(:,:) >= 90);
% outline1(rep) = 255; 
% rep0 = find(outline1(:,:) ~= 255);
% outline1(rep0) = 1;
% zero = find(registered1(:)==0); registered1(zero) = 1;
% annotation = (outline1.*registered1);
% figure; imagesc(annotation);

%% Annotation refine: adjust the outline to tissue edges
annotation_mask = imfill(outline1, 'holes'); 
figure(3); imagesc(annotation_mask); colormap gray

object_mask = edge(rgb2gray(registered1), 'Canny'); 
se = strel('disk', 20);
object_mask = imclose(object_mask, se);
% Convert the logical matrix to a double matrix
double_ob = double(object_mask);
% Create the Gaussian filter
filter_size = 103; % Adjust the filter size as needed
sigma = 83.0; % Adjust the standard deviation as needed
gaussian_filter = fspecial('gaussian', filter_size, sigma);
% Apply the Gaussian filter
object_mask = imfilter(double_ob, gaussian_filter, 'same'); object_mask = imclose(object_mask, se);
% Threshold the smoothed matrix
threshold = 0.1; % Adjust the threshold as needed
object_mask = object_mask > threshold; 
figure(4); imagesc(object_mask);colormap gray; 

final_mask = annotation_mask & object_mask; 
figure(5); imagesc(final_mask); colormap gray;

% Define the structuring element with the desired thickness
se = strel('disk', 5); 
% Erode the logical matrix to get the inner boundary
eroded_mask = imerode(final_mask, se);
% Subtract the eroded matrix from the original matrix to get the annotation line
annotation_line = final_mask & ~eroded_mask; %figure; imagesc(annotation_line);colormap gray;
annotation_line = uint8(~annotation_line);
f = find(annotation_line == 0); annotation_line(f) = 255; 

tformInv = fitgeotform2d(fixedPoints,movingPoints,"polynomial",2);
R = imref2d(size(moving));
deformed = imwarp(annotation_line,tformInv,FillValues=0,OutputView=R);
% imagesc(deformed); colormap gray;

%% Outline Application
% avoid all black: change 0 to 1
re = find(moving(:,:,:)==0);
moving(re) = 1;
ori = deformed .* moving;
imshow(ori);
aMUSE = deformed .* moving(:,:,:); 
figure; imagesc(aMUSE); 
imwrite(aMUSE,'annotation.jpg');

%% Similarity Score Calculation
% load manual annotated images and extract annotation area
im1 = imread("manual.jpg");
im2 = imread("annotation.jpg");
[deepS] = compute_easysimilarity(im1, im2);
fprintf('CNN feature and similarity: %f\n', deepS);

img1 = im1;
img1 = img1(:,:,1);
a = find(img1(:,:)>=230);
b = find(img1(:,:)<230);
se = strel('disk', 10); 
img1(a) = 255; img1(b) = 0; img1 = imclose(img1,se);
img1 = bwareaopen(img1, 80000);  
img1 = imfill(img1,'holes'); img1 = logical(img1);
figure; imshow(img1);
% load annotation area using algorithm 
se = strel('disk', 200); 
img2 = deformed; img2 = imclose(img2,se);
se = strel('disk', 200); 
img2 = imfill(img2,'holes'); 
img2(img2 == 1) = 0;
img2 = logical(img2);
figure; imshow(img2);

% call the function to calculate the similarity scores
[diceScore jaccardIndex precision recall hd] = computeScore(img1, img2);
fprintf('Similarity Score:\n Dice Score: %f\n Jaccard Index: %f\n Precision: %f\n Recall: %f\n  haur dis: %f\n boundary:%f\n', ...
    diceScore, jaccardIndex, precision, recall, hd, boundaryIoU);

