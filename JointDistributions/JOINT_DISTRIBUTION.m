classdef JOINT_DISTRIBUTION < handle
    % create a joint distribution from a list of marginal distributions
    % An important component is that the joint distribution can be
    % adaptively updated over time.
    % So, an update may depend upon the joint distribution at the last time
    % step at which it was calculated
    properties
        MDarray % 1XM cell array. Each element is a MARGINAL_DISTRIBUTION 
                % instance
    end
    methods
        function obj = JOINT_DISTRIBUTION()
        end
        function input_MDarray(obj,MDarray)
            % input the set of marginal distributions
            % inputs:
            %  - MDarray: 1XM cell array
            obj.MDarray = MDarray;
        end
        function input_data(obj,margcellXy,update)
            % input data into each of the MD instances
            % inputs:
            %  - margcellXy: 1XM cell array. Each element ii is NX(M_ii)
            %                matrix. The first (M_ii)-1 columns are 
            %                predictor variables, the last column is the 
            %                response variable
            %  - update  : 1XM cell array.  See MARGINAL_DISTRIBUTIONS
            assert(size(margcellXy,1)==1,'margcellXy should be 1XM')
            for ii=1:length(margcellXy)
                d = margcellXy{ii};
                MD = obj.MDarray{ii};
                if isempty(d)
                    MD.input_data(d,d,update{ii});
                else
                    MD.input_data(d(:,1:end-1),d(:,end),update{ii});
                end
            end
        end
        function input_zero_one_percentiles(obj,zopcell)
            % this is used to input the 0 and 100 percentile function
            % handles into all the marginal distributions
            % these handles are created from a GET_MARGINAL_DATA instance
            for ii=1:size(obj.MDarray,2)
                MD = obj.MDarray{ii};
                MD.input_zero_one_percentiles(zopcell{1,ii},zopcell{2,ii});
            end
        end
        function p = cdf(obj,margcellXy)
            % get the percentiles corresponding to the predictor variables
            % in margcell
            % inputs:
            %  - margcell: 1XM array. Each cell is NX(M_i) array. 
            %              Columns 1:(end-1) are the predictor variables
            %              Column end is the response varaible
            % returns:
            %  - p: NXM array. Element (i,j) is the cdf of predictor
            %       variable i in marginal distribution j
            MD1 = margcellXy{1};
            p1  = MD1.cdf(MD1(:,1:end-1),MD1(:,end));
            p   = zeros(size(p1,1),length(obj.MDarray));
            p(:,1) = p1;
            for ii=2:length(obj.MDarray)
                MD = margcellXy{ii};
                p(:,ii) = MD.cdf(MD(:,1:end-1),MD(:,end));
            end
        end
        function y = inverse_cdf(obj,margcellX,p)
            % get the response variables corresponding to predictor
            % variables in margcell and the percentiles in p
            % inputs:
            %  - margcellX: 1XM array. Each cell is NX(M_i-1) array. 
            %               Columns 1:end are the predictor variables
            %  - p        : NXM array of percentiles
            % returns:
            %  - y: NXM array. Element (i,j) is the inverse cdf of 
            %       predictor variable i in marginal distribution j
            %       corresponding to percentile p(i,j)
            MD1 = obj.MDarray{1};
            y1  = MD1.inverse_cdf(margcellX{1},p(:,1));
            y   = zeros(size(y1,1),length(margcellX));
            y(:,1) = y1;
            for ii=2:length(margcellX)%length(obj.MDarray)
                MD = obj.MDarray{ii};
                y(:,ii) = MD.inverse_cdf(margcellX{ii},p(:,ii));
            end
        end
        function p = get_percentiles_all(obj)
            % get the percentiles of all data that has been input, whether
            % or not the marginal distribution was trained on all the data
            % returns:
            %  - p: NXlength(MDarray)
            MD1 = obj.MDarray{1};
            p1  = MD1.get_percentiles_all();
            p   = zeros(size(p1,1),length(obj.MDarray));
            p(:,1) = p1;
            for ii=2:length(obj.MDarray)
                MD = obj.MDarray{ii};
                p(:,ii) = MD.get_percentiles_all();
            end
        end
        function p = get_percentiles_trained(obj)
            % get the percentiles of all the data the marginal
            % distributions were trained on
            % returns:
            %  - p: NXlength(MDarray)
            MD1 = obj.MDarray{1};
            p1  = MD1.get_percentiles_trained();
            p   = zeros(size(p1,1),length(obj.MDarray));
            p(:,1) = p1;
            for ii=2:length(obj.MDarray)
                MD = obj.MDarray{ii};
                p(:,ii) = MD.get_percentiles_trained();
            end
        end
        function p = get_percentiles_trained_last(obj)
            % get the percentiles of all the data the marginal
            % distributions were trained on
            % returns:
            %  - p: NXlength(MDarray)
            MD1 = obj.MDarray{1};
            p1  = MD1.get_percentiles_trained_last();
            p   = zeros(size(p1,1),length(obj.MDarray));
            p(:,1) = p1;
            for ii=2:length(obj.MDarray)
                MD = obj.MDarray{ii};
                p(:,ii) = MD.get_percentiles_trained_last();
            end
        end
        
    end
    methods (Sealed)
        function obj = train_MD(obj,init_MD_training, MDtypelist,...
                            prev_MD_training_cell, numdist)
            % inputs:
            %
            % OPTION 1 ----------------------------------------------------
            %  - init_MD_training: an initial MD_training 
            %  - MDtypelist: list of integers (aka types)
            %            length(MDlist)=length(obj.MDarray)
            %            obj.MDarray(MDlist==integer) all have the same
            %            type of predictor variables (i.e. MPRD is the
            %            same)
            %  - numdist: empty
            %
            % OPTION 2 ----------------------------------------------------
            %  - init_MD_training: empty 
            %  - MDtypelist: empty
            %  - prev_MD_training_cell: cell array 
            %            length(prev_MD_training_cell)=length(obj.MDarray)
            %  - numdist: positive integer, number of marginal
            %       distributions to train
            %
            if nargin<4
                prev_MD_training_cell = [];
            end
            if ~isempty(init_MD_training)
                assert(~isempty(MDtypelist))
%                 MDtypelist
                assert(length(obj.MDarray)==length(MDtypelist))
                assert(isempty(prev_MD_training_cell))
                
                typelist = unique(MDtypelist);
                Ntypes   = length(typelist);
                prevtype = NaN(1,Ntypes); 
                for ii=1:length(obj.MDarray)
                    curtype = MDtypelist(ii);
                    index   = find(typelist==curtype); % index in typelist
                    prev    = prevtype(index);
                    if isnan(prev)
                        prev_MD_training = init_MD_training;
                    else
                        MD = obj.MDarray{prev};
                        prev_MD_training = MD.get_training();
                        % update the number of time steps it's been since
                        % this type has been trained
                        prev_MD_training.nts_since_last_trained = ...
                            prev_MD_training.nts_since_last_trained ...
                            +ii-prev-1; 
                    end
%                     [ii prev_MD_training.nts_since_last_trained]
                    MD = obj.MDarray{ii};
                    MD.train(prev_MD_training);
                    prevtype(index) = ii;
                end
            else
                assert(isempty(MDtypelist))
                assert(length(prev_MD_training_cell)==length(obj.MDarray))
                
                for ii=1:numdist%length(obj.MDarray)
                    MD = obj.MDarray{ii};
                    MD.train(prev_MD_training_cell{ii});
                end
            end
            
        end
        function obj = train_MD_orig(obj, prev_MD_training, MD_training_multiple)
            % train each of the marginal distributions
            if nargin<3
                MD_training_multiple = false;
            end
            if ~MD_training_multiple
                prev_MD_training_ii = prev_MD_training;
            end
            for ii=1:length(obj.MDarray)
                if MD_training_multiple
                    prev_MD_training_ii = prev_MD_training{ii};
                end
                MD = obj.MDarray{ii};
                MD.train(prev_MD_training_ii);
                if ~MD_training_multiple
                    prev_MD_training_ii = MD.get_training();
                end
            end
        end
    end
    methods (Abstract)
        obj = train(obj,prev_MD_training, MDtypelist, ...
                        prev_MD_training_cell,numdist)
            % train the joint distribution - if applicable, the joint
            % distribution should be updated instead of trained from
            % scratch
            % the first line of this function should call 
            %   train_MD(prev_MD_training)
            % inputs:
            %  - MD_training: some type of object that may contain
            %     information from the MD that was trained at the previous
            %     time step
            %     This allows the MDs to have dependence going forward in
            %     time
            %     e.g. do a time adaptive approach
        obj = train_clear(obj)
            % remove the trained joint distribution
            % this makes it so that when train() is called the joint
            % distribution is learned from scratch
        s = generate(obj,margcellX,numdist)
            % generate scenarios
            % inputs:
            %  - margcellX: almost same as in input_data
            %               1XM cell array. Each element ii is NX(M_ii)
            %               matrix. All columns are the predictor
            %               variables. (There's no respones variable like
            %               in input_data.)
            %  - numdist: number of marginal distributions used to generate
            % returns:
            %  - s: NXM array of scenarios.  Each row ii=1:N corresponds to
            %       a scenario generated with predictor variables given by
            %       row ii in margcellX
        s = generate_conditional(obj,margcellX,respscen)
            % generate scenarios conditioned on response scenarios
            % inputs:
            %  - margcellX: almost same as in input_data
            %               1XM cell array. Each element ii is NX(M_ii)
            %               matrix. All columns are the predictor
            %               variables. (There's no respones variable like
            %               in input_data.)
            %  - respscen : NXR array, 1<=R<M are response variables
            %               respscen(ii,jj) is the response variable of
            %               scenario ii for marginal distribution jj
            % returns:
            %  - s: NX(M-R) array of scenarios.  s(ii,jj) is the response
            %       variable of scenario ii in marginal distribution jj+R
            %       (where the generated response variables s(:,R+1:end)
            %       are conditioned on s(:,1:R))
    end
    methods (Static)
        function [margcell,zerop,onep,sigma] =...
                                test_generate_data_1(M,N,d,scale)
            % M is number of marginal distributions
            % N is number of data points
            % d is number of predictor variables
            % the response variable has a standard normal distribution with
            % a mean equal to the sum of the predictor variables
            margcell = cell(1,M);
            % create predictor variables 
            for ii=1:M
                margcell{ii} = rand(N,d)*scale;
            end
            % generate random positive definite covariance matrix that has
            % ones along the diagonal
            sigma = rand(M);
            di = diag(sigma*sigma').^0.5;
            sigma = (sigma*sigma')./(di*di');
            sigma = exp(10*(sigma-1)); % make the covariances smaller
            % generate N d-dimensional multivariate normal random variables
            % with mean 0 and covariance sigma
            y = mvnrnd(zeros(M,1),sigma,N);
            maxabsy = 4;
            y = min(max(y,-maxabsy),maxabsy);
            % now add the mean to the response variables
            for ii=1:M
                x = margcell{ii};
                margcell{ii} = [ x , sum(x,2)+(1+sum(x,2)*0.2).*y(:,ii) ];
            end
            zerop = @(x) sum(x,2)-maxabsy;
            onep  = @(x) sum(x,2)+maxabsy;
            
        end
        
    end
    
    
end