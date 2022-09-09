function outputID=FaceRecognitionNew(trainingImgSet, trainPersonID, testPath)
testImgNames = dir([testPath,'*.jpg']);
testImgLen = size(testImgNames,1); % Define the number of test images
trainImgLen= size(trainingImgSet,4); % Define the number train images
imgSize = [224,224,3]; %Define
trainArrSize = [4096,1,trainImgLen]; % Define the size of output array for train images
testArrSize = [4096,1,testImgLen]; % Define the size of output array for test images

ppTrainImg=zeros(224,224,3,trainImgLen); % Define the size of pre-processed array for train images
trainVector = zeros(trainArrSize,"double");
dltrainVector = dlarray(trainVector,"CSB");

testVector = zeros(testArrSize, "double");
dltestVector = dlarray(testVector,"CSB");

faceDetector = vision.CascadeObjectDetector;

for img = 1:trainImgLen
   ppTrainImg(:,:,:,img) = Preprocess(trainingImgSet(:,:,:,img),faceDetector); 
end

disp("Creating Nueral Network")
[net, fcParams] = SiameseNetwork(ppTrainImg, trainPersonID,imgSize); % Creating Neural network
outputID = [];

disp("Get training vector")
for i = 1:trainImgLen 
    currImg = dlarray(double(ppTrainImg(:,:,:,i)),"SSCB"); 
    dltrainVector(:,:,i) = getFeature(currImg,net); % Pass into neural network to obtain encoded feature from training images
end
disp("Get test vector")
for i = 1:testImgLen    
    TestImg = imread([testPath,testImgNames(i).name]);
    currTestImg = Preprocess(TestImg,faceDetector);
    currImg = dlarray(double(currTestImg),"SSC");
    dltestVector(:,:,i) = getFeature(currImg,net); % Pass into neural network to obtain encoded feature from test images 
end

disp("Start Predicting");
for i = 1:testImgLen % Loop through 1344 of test images
    disp("currImg");
    disp(i)

    scoreArr = zeros(1,trainImgLen);
    for j = 1:trainImgLen % Loop through 100 of train images
        Y = getDistance(dltestVector(:,:,i), dltrainVector(:,:,j)); % Calculate the similarities/differences between 2 encoded features
        scoreArr(j) = Y; % Record the score in an array
    end
    [~,predLabel] = max(scoreArr); % Obtain the pair of images with highest similarity score through argmax
    predID = trainPersonID(predLabel,:); 
    disp(predID);
    outputID=[outputID; predID];
end
end

function outputImg = Preprocess(img, faceDetector) 
% Image Pre-Processing function
img = rgb2gray(img); 
img = cat(3,img,img,img); % increase dimensions of gray-scale image to 3-dimensions
faceDetector.MergeThreshold = 6; 
bbox = faceDetector(img); % Detect faces in image with Viola-Jones, and output the coordinates
if ~isempty(bbox)
    img = imcrop(img, bbox(1,:)); % Crop the image according to the coordinates
end

outputImg = imresize(img,[224,224]);
end

function [net, fcParams]=SiameseNetwork(trainImgSet, trainPersonID, imgSize) 
% Create neural network
lgraph = importKerasLayers("resnet50_04.h5", 'ImportWeights', true); % Import pre-trained network from Keras
net = dlnetwork(lgraph);
fcWeights = dlarray(0.01*randn(1,4096));
fcBias= dlarray(0.01*randn(1,1));
fcParams = struct("FcWeights", fcWeights, "FcBias", fcBias);

end

function Y = getFeature(X,net)
% Get encoded features
Y = predict(net,X);
end

function score = getDistance(X1,X2)
% Similarity with Cosine Similarity
score= sum(X1.*X2)./sqrt(sum(X1.^2).*sum(X2.^2));
end




