function W = getPCAWhite( fv_data, targetDim )
%GETPCAWHITE Compute PCA-Whitening matrix 
%   Dependency on pca() function of MATLAB Statistics Toolbox
    
    % pca() expects data to be row-wise, hence transposed fv_data
    [pca_proj, ~, pca_eigvals] = ...
                   pca(fv_data', 'NumComponents', targetDim, ...
                    'Centered',false);
    
    whiteMat =  diag(1./sqrt(pca_eigvals(1:targetDim) + 1e-5));           
    W = single(whiteMat * pca_proj');
    
end

