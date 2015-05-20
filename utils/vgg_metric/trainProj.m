function [ model ] = trainProj( faceFeat, personID, personList, params )
%TRAINPROJ Discriminative dimensionality reduction of Fisher Vectors
%   params.targetDim Reduced dimension
%   params.numPairs  total number of +ve and -ve pairs
%   params.numIter   total number of iterations
%   params.W         [] or pre-computed
%   params.b         [] or pre-computed
%   params.gamma     [] or pre-computed
%   params.gammaBias [] or pre-computed
%   params.lambda    [] or pre-computed
%
%   personID(i) has the class label (person) for the i-th sample in the dataset
%     1xN where N=total number of samples/images in dataset
%   personList is the list of all unique persons present in the dataset
%     1xP where P=number of persons in dataset

% TODO - give Validation pos and neg pairs for gamma & gammaBias grid-search


% =========================================================================
%                          INITIALIZATION
% =========================================================================

    % Intializations
    numPairs = params.numPairs;
    t0 = 1;
    idxPos = 0;
    idxNeg = 0;
    numTrainBias = 1e3; % edit to 1e3

    if ~isfield(params,'numIter') || isempty(params.numIter), 
        numIter = 1e6;
    else
        numIter = params.numIter;
    end

    if ~isfield(params,'gamma') || isempty(params.gamma), 
        gamma = 0.0001;
    else
        gamma = params.gamma;
    end

    if ~isfield(params,'gammaBias') || isempty(params.gammaBias), 
        gammaBias = 0.0001;
    else
        gammaBias = params.gammaBias;
    end
    
    if ~isfield(params,'lambda') || isempty(params.lambda), 
        lambda = 0.01;
    else
        lambda = params.lambda;
    end
    
    % Form positive and negative pairs
    disp('Forming positive and negative pairs.');
    posPair = getPosPairs(personID, personList, numPairs/2); %2xnumPairs    
    negPair = getNegPairs(personID, personList, numPairs/2); %2xnumPairs 

    
    nPos = size(posPair, 2);
    nNeg = size(negPair, 2);
    
      
    % Initialize W to PCA-whitening matrix
    disp('PCA-whitening.');
    if isempty(params.W)     
        W = getPCAWhite( faceFeat, params.targetDim );
    else
        W = params.W;
    end
    model.W = W;
    
    
    % Dataset for initializing bias - subset from pos and neg pairs
    idxBiasTrainPos = posPair(:, randi(nPos, numTrainBias, 1));
    idxBiasTrainNeg = negPair(:, randi(nNeg, numTrainBias, 1));
        
        
    % Initialize bias b on validation-set FV differences
    % TODO - testing and debugging initValidBias() on real inputs
    disp('Initializing bias b.');
    if isempty(params.b)
        b = initValidBias(faceFeat, W, idxBiasTrainPos, idxBiasTrainNeg);       
    else
        b = params.b;
    end
    model.b = b;
       


% =========================================================================
%                STOCHASTIC GRADIENT DESCENT ON W & BIAS b - VGG code
% =========================================================================

    % SGD iterations
    disp('Beginning stochastic gradient descent.');
    for t = t0:numIter

        if mod(t, 2) == 1            
           
            % positive sample
            idxPos = idxPos + 1;
            
            if idxPos > nPos
                idxPos = 1;
            end
            
            % feature vector
            featDiff = faceFeat(:, posPair(1, idxPos)) - faceFeat(:, posPair(2, idxPos));
            featDiffProj = W * featDiff;
                        
            % update
            if norm(featDiffProj) ^ 2 > b - 1
                W = W - (gamma * featDiffProj) * featDiff';
                b = b + gammaBias;
            end
            
        else
            
            % negative sample
            idxNeg = idxNeg + 1;
            
            if idxNeg > nNeg
                idxNeg = 1;
            end
            
            % feature vector
            featDiff = faceFeat(:, negPair(1, idxNeg)) - faceFeat(:, negPair(2, idxNeg));
            featDiffProj = W * featDiff;
            
            % update W
            if norm(featDiffProj) ^ 2 < b + 1
                W = W + (gamma * featDiffProj) * featDiff';
                b = b - gammaBias;
            end
            
        end  

        % regularization
        W = (1-gamma*lambda) * W;

        if mod(t,1e4)==0, fprintf('.'); end;
        if mod(t,1e5)==0, fprintf(' [%d/%d]\n', t,numIter); end;
            
    end
    disp('Completed.');
    
% =========================================================================
%                SETTING LEARNED MODEL VALUES
% =========================================================================
       
    model.W = W;
    model.b = b;
    model.discrDimRed = true;
        
end



% Currently un-used!
function eer = get_val_eer(W)
% validation EER

    valFeatsProj = W * valData.pairFeat;    
    valDist = sum(valFeatsProj .^ 2, 1);

    [~,~,info] = vl_roc(valData.anno, -valDist);
    eer = 1 - info.eer;

end

