function [deepSim] = compute_easysimilarity(img1, img2)
   
    % Load pre-trained network
    net = alexnet;
    % Resize images to network input size
    img1_resized = imresize(img1, [net.Layers(1).InputSize(1), net.Layers(1).InputSize(2)]);
    img2_resized = imresize(img2, [net.Layers(1).InputSize(1), net.Layers(1).InputSize(2)]);
    % Extract features
    features3 = activations(net, img1_resized, 'fc7');
    features4 = activations(net, img2_resized, 'fc7');
    % Compute cosine similarity between feature vectors
    vec1 = squeeze(features3);
    vec2 = squeeze(features4);
    deepSim = dot(vec1, vec2) / (norm(vec1) * norm(vec2));

end

