function b = initValidBias( faceFeat, W, idxBiasTrainPos, idxBiasTrainNeg )
%INITVALIDBIAS Initialize bias b on the validation set 
%   
    numTrainBias = size(idxBiasTrainPos,2) + size(idxBiasTrainNeg, 2);
    featDiff = W * (faceFeat(:, idxBiasTrainPos(1, :)) - faceFeat(:, idxBiasTrainPos(2, :)));
    biasTrainDistPos = sum(featDiff .^ 2, 1);

    featDiff = W * (faceFeat(:, idxBiasTrainNeg(1, :)) - faceFeat(:, idxBiasTrainNeg(2, :)));
    biasTrainDistNeg = sum(featDiff .^ 2, 1);

    biasTrainAnno = [ones(1, numTrainBias/2, 'single'), -ones(1, numTrainBias/2, 'single')];

    [~, extra] = evalBestThresh(-[biasTrainDistPos, biasTrainDistNeg], biasTrainAnno);
    b = -extra.bestThresh;
end

