function [diceScore jaccardIndex precision recall hd] = computeScore(img1, img2)

    % Compute the intersection and union
    intersection = sum(img1(:) & img2(:));
    union1 = sum(img1(:)) + sum(img2(:));
    union2 = sum(img1(:) | img2(:)); 

    % Compute Dice score
    diceScore = 2 * intersection / union1;
    
    % Compute Jaccard Index
    jaccardIndex = intersection / union2;

    % Compute Precision and Recall
    precision = intersection / sum(img2(:));
    recall = intersection / sum(img1(:));

    % Compute the percentage of the overlap over the whole tissue area
    bw1 = bwperim(img1);
    bw2 = bwperim(img2);
    % 
    [y1, x1] = find(bw1);  % fixed image
    [y2, x2] = find(bw2);  % registered/moving image

    A = [x1, y1];
    B = [x2, y2];

    % Compute pairwise distances between points in A and B
    D = pdist2(A, B);

    % Directed distances
    dAB = max(min(D, [], 2));  % A to B
    dBA = max(min(D, [], 1));  % B to A

    % Symmetric Hausdorff distance
    hd = max(dAB, dBA);
    % hd=0;


end


