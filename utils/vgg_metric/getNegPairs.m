function neg_pairs = getNegPairs(personID, personList, numPairs)
%GETNEGPAIRS Returns randomly sampled negative face pair indices
% personID(i) has the class label (person) for the i-th sample in the dataset
%   1xN where N=total number of samples/images in dataset
% personList is the list of all unique persons present in the dataset
%   1xP where P=number of persons in dataset
% numPairs is the number of negative pairs to be returned

    rng(65535);  
    neg_pairs = zeros(2, numPairs);
   
    for i = 1:numPairs
        
        % select two different persons
        t = randperm(length(personList), 2);
        personListIdx1 = personList(t(1));
        personListIdx2 = personList(t(2));      
        
        % All data indices of that person
        idxData1 = find(personID == personListIdx1);
        idxData2 = find(personID == personListIdx2); 
        
        % generate random pair of image-indices of that person
        neg_pairs(1, i) = idxData1(randi(length(idxData1)));
        neg_pairs(2, i) = idxData2(randi(length(idxData2)));     
    end
end
