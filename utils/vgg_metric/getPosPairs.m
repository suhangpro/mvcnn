function pos_pairs = getPosPairs(personID, personList, numPairs)
%GETPOSPAIRS Returns randomly sampled positive face pair indices
% personID(i) has the class label (person) for the i-th sample in the dataset
%   1xN where N=total number of samples/images in dataset
% personList is the list of all unique persons present in the dataset
%   1xP where P=number of persons in dataset
% numPairs is the number of positive pairs to be returned

    rng(65535);
    pos_pairs = zeros(2, numPairs);
    numSamples = length(personList);
    
    % number of data-samples per person
    personNumImg = hist(personID, [1:numSamples]); %#ok<NBRAK>
    
    % IDs of persons that have >= 2 data-samples
    personSet = personList(personNumImg >= 2); 
    
    
    for i = 1:numPairs
        
        % select person
        pid = personSet(randi(length(personSet)));
        
        % All image indices of that person
        idxData = find(personID == pid); %linear indexing -> position
        
        % generate random pair of image-indices of that person
        t = randperm(length(idxData), 2);
        pairIdx1 = t(1);
        pairIdx2 = t(2);
        pos_pairs(1, i) = idxData(pairIdx1);
        pos_pairs(2, i) = idxData(pairIdx2);
         
    end
end