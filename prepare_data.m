function prepare_data(datasetname, y, A, splits, mode, mat_type, distribution, onelarge_factor)
%SPITATOMMATRIX Export data from matrix format for dFW C++/MPI impl.
%   AUTHOR: Alireza Bagheri Garakani (me@alirezabagheri.com), 11/4/2014
%
%   INPUT: 
%       datasetname (string)
%       y (vector) - labels (for size, see 'distribution' option)
%       A (matrix) - data matrix (dimn x atoms)
%       splits (int) - number of splits over atoms (i.e. MPI hosts)
%       mode (string) - either 'distribfeatures' (as in Lasso Regression)
%               or 'distribexamples' (as in SVM)
%       mat_type (string) - specifies how data should be written out (use 
%               either 'sparse' or 'dense')
%       distribution (string) - determines how to split data; either
%               'uniform' (evenly), 'weighted' (uneven), or 'one-large'.
%       onelarge_factor (double) - specifies extent at which the one-large
%               split is larger (ignored if distribution ~= one-large).
%               (OPTIONAL; default = 2)
%
%   OUTPUT: data written in text format to disk (count = splits).


[dimn, atoms] = size(A);

if (~exist('onelarge_factor'))
    onelarge_factor = 2;
end

switch(distribution)
    case 'uniform'
        location = randsample(splits,atoms,true);
    case 'weighted'
        node_weights = abs(normrnd(0,1,splits,1));
        location = randsample(splits,atoms,true,node_weights);
    case 'one-large'
        node_weights = ones(splits,1); node_weights(end) = onelarge_factor;
        location = randsample(splits,atoms,true,node_weights);
    otherwise
        error('invalid distribution');
end

for idx = 1:splits  
    filename = [datasetname '.' mat_type '.' mode '.' distribution ...
        '_' num2str(onelarge_factor) '.of' num2str(splits) '.' num2str(idx)]
    
    switch(mode)
        case 'distribfeatures'
            PART = [y(:) A(:,location==idx)]; % label is first col
        case 'distribexamples'
            PART = [y(location==idx); A(:,location==idx)]; % label is first row
        otherwise
            error('invalid opt');
    end
    
    switch(mat_type)
        case 'sparse'
            [i,j,val] = find(PART);
            data_dump = [i,j,val];
            meta = [max(i),max(j),0];
        case 'dense'
            data_dump = PART;
            meta = [size(PART,1) size(PART,2)];
        otherwise
            error('invalid mat_type');
    end
    
    dlmwrite(filename, full(meta), 'delimiter', ',', 'precision', '%i');
    dlmwrite(filename, full(data_dump), '-append', 'delimiter', ',', 'precision', 6);
end

