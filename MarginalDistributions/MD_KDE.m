classdef MD_KDE < MARGINAL_DISTRIBUTIONS
    % multivariate product kernel density estimation
    % see: "Time Adaptive Conditional Kernel Density Estimation for Wind Power
    % Forecasting", 2012
    %
    % the product kernel works as follows:
    % let X be nX1 array, y be a scalar
    % let K_i be a kernel function for i=1,...,n+1, i.e.
    %  p = K_i(r,bw), with
    %    r a real scalar, bw (bandwidth) a non-negative scalar, p a
    %    non-negative scalar
    %    K_i must satisfy for any bw: 
    %      \integral_{r=-inf}^{r=inf} K_i(r,bw) dr = 1
    % then the joint probability density function, pdf, is 
    %    p = K_{n+1}(y,bw_{n+1}) \times \prod_{i=1}^{n} K_i(X(i),bw_i)
    % notice that there are n+1 bandwidth parameters, bw_i for i=1,...,n+1
    % also notice that this is a pdf because p>=0 and the integral over all
    %   y and all X(i), i=1,...,n is equal to 1
    % There are three main functions:
    %  let X be nXd, y be nX1, and p be nX1
    %  1. p = pdf(X,y); % joint proability density function of [X,y]
    %  2. p = cdf(X,y); % cumulative density function of y conditioned on X
    %  3. y = inverse_cdf(X,p); % y = inverse_cdf(X,cdf(X,y))
    % the bandwidths are determined using cross validation to maximize some
    % objective function, e.g. negative log likelihood
    %
    % There are a couple of computational optimizations:
    %
    % 1. The product kernels K_i need to be evaluated very often, so the
    % kernel can be precomputed using the SavedKernel
    % SavedKernel is a lookup table and uses linear interpolation to
    % evaluate the kernel between nodes
    %
    % 2, 3. The bandwidth parameters can be initialized with a best-guess
    % optimized valueusing the previous MD training results:
    %  prev_MD_training = struct('bw',[],...
    %                            'nts_since_last_trained',Inf);
    % This is used to incoporate the fact that marginal distributions can
    % be sequentially trained, with each marginal distribution
    % corresponding to a different time step; the data (X and y) used at 
    % time step t+1 is very similar to the data at time step t, so it is
    % expected that the optimized bandwidths won't change very much
    %
    % 2. If{ prev_MD_training.bw is not empty and
    % prev_MD_training.nts_since_last_trained<=obj.train_MD_every_nts} then
    % the optimized bandwidths are set equal to prev_MD_training.bw
    %
    % 3. If{ prev_MD_training.bw is not empty and
    % prev_MD_training.nts_since_last_trained>obj.train_MD_every_nts} then
    % the bandwidths are optimized in the neighborhood of
    % prev_MD_training.bw

    properties
        pred_kernels 
        pred_bds
        resp_kernel
        resp_bds
        pred_bandwidths
        resp_bandwidth
        pred_bw_dis % 1XN array of allowable values of pred_bandwidths
        resp_bw_dis % 
        istrained % boolean
        train_method
        do_bw_nn % boolean; if true than vary the bandwidth based upon the 
                 % number of nearest neighbors
    end
    methods
        function obj = MD_KDE()
            obj.istrained = false;
            obj.train_method = 'lik'; % training method
            obj.do_bw_nn = false; % variable bandwidth
            obj.train_MD_every_nts = 1;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % descriptions of this MD child
        
        function d = get_description(obj)
            d = ['MD_KDE ',...
                    ' train_method:',obj.train_method,...
                    ' weight_exp_lambda:',num2str(obj.weight_exp_lambda),...
                    ' train_MD_every_nts:',num2str(obj.train_MD_every_nts)];
        end
        function name = get_name(obj)
            name = 'KDE';
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % input the kernels
        
        function obj = input_predictor_kernels(obj,kernels_cell)
            assert(size(kernels_cell,1)==1,...
                'kernels_cell must be an array')
            obj.pred_kernels = kernels_cell;
            % make the upper and lower bounds
            obj.pred_bds = zeros(2,length(kernels_cell));
            for ii=1:length(kernels_cell)
                k = kernels_cell{ii};
                obj.pred_bds(1,ii) = k.get_lower_bound();
                obj.pred_bds(2,ii) = k.get_upper_bound();
            end
        end
        function obj = input_response_kernel(obj,kernel)
            obj.resp_kernel = kernel;
            % need to check that the kernel is bounded above and below and 
            % is not circular
            assert(kernel.get_lower_bound() > -Inf,...
                'respones kernel must be bounded below')
            assert(kernel.get_upper_bound() <  Inf,...
                'respones kernel must be bounded above')
            assert(~kernel.get_iscircular(),...
                'response kernel cannot be circular')
            % make the upper and lower bounds
            obj.resp_bds = zeros(2,1);
            obj.resp_bds(1,1) = kernel.get_lower_bound();
            obj.resp_bds(2,1) = kernel.get_upper_bound();
        end
        function obj = input_SavedKernel(obj,SK)
            obj.resp_kernel = obj.resp_kernel.input_SavedKernel(SK);
            for ii=1:length(obj.pred_kernels)
                k = obj.pred_kernels{ii};
                obj.pred_kernels{ii} = k.input_SavedKernel(SK);
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % training
        
        function obj = input_training_method(obj,train_method)
            % train_method can either be 'lik' (recommended) or 'cdf'
            obj.train_method = train_method;
        end
        function training = get_training(obj)
            if obj.istrained
                bw = [obj.pred_bandwidths obj.resp_bandwidth];
                training = struct('bw',bw,...
                    'nts_since_last_trained',obj.nts_since_last_trained);
            else
                training = struct('bw',[],...
                    'nts_since_last_trained',Inf);
            end
        end
        function obj = clear_train(obj)
            % clear bandwidth parameters
            obj.pred_bandwidths = [];
            obj.resp_bandwidth = [];
        end
        function obj = train(obj,prev_MD_training)
            % need to estimate the bandwidth parameters
            % first check that all samples respect the bounds
            if nargin<2
                prev_MD_training = struct('bw',[],...
                                          'nts_since_last_trained',Inf);
            end
            notnanX = ~isnan(obj.X);
            notnany = ~isnan(obj.y);
            for ii=1:size(obj.X,2)
                assert(all(obj.X(notnanX(:,ii),ii)>=obj.pred_bds(1,ii)),...
                    ['training data violates lower bound, response variable ',num2str(ii)]);
                assert(all(obj.X(notnanX(:,ii),ii)<=obj.pred_bds(2,ii)),...
                    ['training data violates upper bound, response variable ',num2str(ii)]);
            end
            assert(all(obj.y(notnany)>=obj.resp_bds(1,1)),...
                'training response variable violates lower bound')
            assert(all(obj.y(notnany)<=obj.resp_bds(2,1)),...
                'training response variable violates upper bound')
            % now do the training to minimize the negative log liklihood by
            % tuning the bandwidth parameters
            obj.train_helper();
            % prev_MD_training = { bandwidths, # of time steps it's been
            % since a marginal distribution has been trained }
            nts = prev_MD_training.nts_since_last_trained;
            if nts+1>=obj.train_MD_every_nts
                % then it's been to long since a marginal distribution has
                % been trained (or a MD has never been trained)
                % so, do the training
                % make the bandwidth domain
                domain = cell(1,length(obj.pred_kernels)+1);
                % the domain can be restricted to the neighborhood of some
                % point `center`
                center = [];
                if obj.istrained
                    center = [ obj.pred_bandwidths(ii) , obj.resp_bandwidth ];
                end
                if ~isinf(nts) && ~isempty(prev_MD_training.bw)
                    center = prev_MD_training.bw;
                end
                if isempty(center)
                    disp('doing full training')
                else
                    disp('doing neighborhood training')
                end
                % the neighborhoor is a hyperrectangle 
                neighb_n = 5;
                for ii=1:length(obj.pred_kernels)
                    k = obj.pred_kernels{ii};
                    d = k.get_bandwidth_discretization();
                    if ~isempty(center)
                        h = center(ii);
                        [~,ind] = min(abs(d-h));
                        d = d(max(1,ind-neighb_n):min(length(d),ind+neighb_n));
                    end
                    domain{ii} = d;
                end
                d = obj.resp_kernel.get_bandwidth_discretization();
                if ~isempty(center)
                    h = center(end);
                    [~,ind] = min(abs(d-h));
                    d = d(max(1,ind-neighb_n):min(length(d),ind+neighb_n));
                end
                domain{end} = d;
                % make the cross folds
                % exclude data points with NaNs
                notnan = and(notnany,all(notnanX,2));
                L = 10; % number of folds
                N = sum(notnan);
                train_inds_mat_tmp = ones(N,L);
                for ii=1:L
                    train_inds_mat_tmp(randsample(N,round(N/L)),ii) = 0;
                end
                train_inds_mat = -ones(size(obj.X,1),L);
                train_inds_mat(notnan,:) = train_inds_mat_tmp;
                if strcmp(obj.train_method,'lik')
                    % maximize negative log liklihood
                    f = @(h) obj.negloglik_train_helper(...
                        obj.X,obj.y,h,train_inds_mat);
                elseif strcmp(obj.train_method,'cdf')
                    % maximize even distribution of cdf
                    f = @(h) obj.train_eval_inverse_cdf_helper(...
                        obj.X,obj.y,h,train_inds_mat);
                else
                    obj.train_method
                    error(['unrecognized KDE training method: ',obj.train_method])
                end
                if isempty(center)
                    % then do optimization over full domain, so have more
                    % iterations
                    maxiter = 75;
                else
                    % then doing optimization over neighborhood domain, so
                    % have fewer iterations
                    maxiter = 25;
                end
                tic
                [bwstar,fstar] = MYOPT.SRBF(f,domain,maxiter);
%                 bwstar = [ 0.001 0.001*ones(1,length(obj.pred_kernels))];%[.0008,.0001];
%                 fstar  = 0;
                toc
                bw = bwstar;
                disp(['done: bwopt: ',num2str(bw),', fopt: ',num2str(fstar)])
                obj.nts_since_last_trained = 0;
            else
                % we don't need to do the training
                % just use the same bandwdiths as the marginal distribution
                % in the previous time step
                bw = prev_MD_training.bw;
                obj.nts_since_last_trained = nts+1;
            end
            % now input the trained bandwidths
            obj.input_pred_bandwidths(bw(1:end-1));
            obj.input_resp_bandwidth(bw(end));
            obj.istrained = true;
        end
        function obj = input_pred_bandwidths(obj,bw)
            assert(length(bw)==size(obj.pred_kernels,2),...
                'length of bw must be the same as the number of predictor kernels')
            obj.pred_bandwidths = bw;
        end
        function obj = input_resp_bandwidth(obj,bw)
            assert(length(bw)==1,'bw must be a double')
            obj.resp_bandwidth = bw;
        end
        
        function p = cdf(obj,X,y,Xtrain,ytrain,resp_bw,pred_bw,dataweights)
            % get the cumulative density
            if nargin<=3
                Xtrain = obj.X;
                ytrain = obj.y;
            end
            if nargin<=5
                resp_bw = obj.resp_bandwidth;
                pred_bw = obj.pred_bandwidths;
            end
            if nargin<=6
                dataweights = obj.get_data_weights(size(Xtrain,1));
            end
            % get the predictor variable weights
            % these are also scaled according to the data point weights
            notnanXtrain = sum(isnan(Xtrain),2)<0.5;
            notnanytrain = ~isnan(ytrain);
            notnantrain  = notnanXtrain & notnanytrain;
            
            pred_bw = obj.get_vector_bws(Xtrain,pred_bw); %%% variable width
            
            weights = obj.get_pred_weights_normalized(X,Xtrain,pred_bw,dataweights);
            weights = weights(:,notnanytrain(notnanXtrain));
            weights = weights./ ( sum(weights,2)*ones(1,sum(notnantrain)) );
            % response kernel cdfs
            resp_bw = obj.get_vector_bws(ytrain(notnantrain),resp_bw); %%% variable width

            p = obj.resp_kernel.cdf(y,ytrain(notnantrain),resp_bw);
            p = sum(p.*weights,2);
        end
        function p = pdf(obj,X,y,Xtrain,ytrain,resp_bw,pred_bw,dataweights)
            % get the product kernel probability density
            if nargin<=3
                Xtrain = obj.X;
                ytrain = obj.y;
            end
            if nargin<=5
                resp_bw = obj.resp_bandwidth;
                pred_bw = obj.pred_bandwidths;
            end
            if nargin<=6
                dataweights = obj.get_data_weights(size(Xtrain,1));
            end
            % get the predictor variable weights
            % these are also scaled according to the data point weights
            pred_bw = obj.get_vector_bws(Xtrain,pred_bw); %%% variable width
            weights = obj.get_pred_weights_normalized(X,Xtrain,pred_bw,dataweights);
            % the response kernel weights
            resp_bw = obj.get_vector_bws(ytrain,resp_bw); %%% variable width
            p = obj.resp_kernel.weight(y,ytrain,resp_bw,true);
            p = sum(p.*weights,2);
        end
        function y = inverse_cdf(obj,X,p)
            % inputs:
            %  X: nXd array of predictor variables
            %  p: nX1 array, conditional cumulative probability; 0<=p(ii)<=1
            % returns:
            %  y: nX1 array of response variables s.t. obj.cdf(X,y)=p
            
            % scale 
            notnanX = sum(isnan(obj.X),2)<0.5;
            notnany = ~isnan(obj.y);
            notnan  = notnanX & notnany;
            % weights is (size(X,1) X sum(notnanX))
            weights = obj.get_pred_weights_normalized(X);
            weights = weights(:,notnany(notnanX));
            % weights is now (size(X,1) X sum(notnan))
            weights = weights./ ( sum(weights,2)*ones(1,sum(notnan)) );
            ydis = zeros(sum(notnan),obj.resp_kernel.get_n_kernel_dis());
            ytrain = obj.y(notnan);
            
            resp_bw = obj.get_vector_bws(ytrain,obj.resp_bandwidth); %%% variable width
            
            for row=1:size(ydis,1)
                % ydis is (size(obj.y,1) X N), where N is the number of
                % discretizations of the kernel
                ydis(row,:) = obj.resp_kernel.get_discretized_weight(...
                    ytrain(row),resp_bw(row),true);  %%% variable width
            end
            % mypdf is (size(X,1) X N)
            % mypdf(row,:) = weights(row,:)*ydis
            % mypdf(row,:) is discretized pdf of response variable given
            %     X(row,:)
            mypdf = weights*ydis;
            scale = true; % scale to domain
            y = obj.resp_kernel.inverse_cdf_integrate(p',mypdf',scale)';
            % do a sanity check
            if ~all(abs(obj.cdf(X,y)-p)<1e-8)
                figure
                hold on
                plot(abs(obj.cdf(X,y)-p),'r')
                plot(abs(obj.cdf(X,y)-p)<1e-8,'b')
                figure
                hold on
                plot(obj.cdf(X,y),'r')
                plot(p,'b')
                plot(abs(obj.cdf(X,y)-p)<1e-8)
                max(abs(obj.cdf(X,y)-p))
                size(p)
                size(obj.cdf(X,y))
                [sum(sum(isnan(X))),sum(sum(isnan(y))),...
                    sum(sum(isnan(obj.cdf(X,y)))),sum(sum(isnan(p)))]
                assert(all(abs(obj.cdf(X,y)-p)<1e-8))
            end
            
        end
        function plot_pcdf_2d(obj,Xtrain,ytrain,Xtest,ytest,is_cdf)
            % make a plot of the log(pdf+1) or cdf when there is ONE 
            % predictor variable the support must be bounded for both
            assert(length(obj.pred_kernels)==1)
            if nargin<=1
                Xtrain = obj.X;
                ytrain = obj.y;
            end
            if nargin<=3
                Xtest = obj.X;
                ytest = obj.y;
            end
            if nargin<=5
                is_cdf = true;
            end
            ngrid = 20;
            k = obj.pred_kernels{1};
            minp = k.get_lower_bound();
            maxp = k.get_upper_bound();
            k = obj.resp_kernel;
            minr = k.get_lower_bound();
            maxr = k.get_upper_bound();
            assert( -Inf<minp & -Inf<minr & maxp<Inf & maxr<Inf)
            xax = linspace(minp,maxp,ngrid);
            yax = linspace(minr,maxr,ngrid);
            [Xgrid, Ygrid] = meshgrid(xax,yax);
            Xline = reshape(Xgrid,ngrid^2,1);
            Yline = reshape(Ygrid,ngrid^2,1);
            if is_cdf
                pline = obj.cdf(Xline,Yline);
                pgrid = reshape(pline,ngrid,ngrid);
            else
                pline = obj.pdf(Xline,Yline);
                pline = log(log(pline+1)+1); % NOTE!!!! - plot log(p+1)
                pgrid = reshape(pline,ngrid,ngrid);
            end
            figure()
            hold on
            contourf(Xgrid,Ygrid,pgrid)
            xlabel('predictor variable')
            ylabel('response variable')
            contourcmap('jet','colorbar','on')
            if nargin>=3
                assert(size(Xtrain,2)==1,'Xtrain must be one dimensional')
                plot(Xtrain,ytrain,'k.')
            end
            if nargin>=5
                assert(size(Xtest,2)==1,'Xtest must be one dimensional')
                plot(Xtest,ytest,'w.')
            end
        end
    end
    methods (Access = private)
        function weights = get_pred_weights_normalized(obj,X,Xtrain,pred_bw,dataweights)
            % get the product of the predictor variable kernels
            % weights is nXm array, n=size(X,1), m=size(Xtrain,1)
            % weights are normalized s.t. sum(weights,2)=ones(n,1)
            if nargin<=2
                Xtrain = obj.X;
            end
            if nargin<=3
                pred_bw = obj.pred_bandwidths;
            end
            
            
            if size(Xtrain,1)~=size(pred_bw,1)
                pred_bw = obj.get_vector_bws(Xtrain,pred_bw);
            end
            
            
            if nargin<=4
                dataweights = obj.get_data_weights(size(Xtrain,1)); % (size(Xtrain,1)X1)
            end
            weights = 1;
            assert(~isempty(Xtrain),'there is no data: obj.X (or Xtrain) is empty')
            % don't use data points with NaNs
            notnantrain = sum(isnan(Xtrain),2)<0.5;
            dataweights = dataweights(notnantrain);
            for ii=1:length(obj.pred_kernels)
                k = obj.pred_kernels{ii};
                weights = weights.*(k.weight(X(:,ii),Xtrain(notnantrain,ii),pred_bw(notnantrain),false));
            end
            assert(~any(any(isnan(weights))))
            % apply data point weighting
            weights = weights.*(ones(size(weights,1),1)*dataweights');
            assert(~any(isnan(dataweights)))
            assert(~any(any(isnan(weights))))
            weights = weights./(sum(weights,2)*ones(1,size(weights,2)));
            assert(~any(any(isnan(weights))))
        end
        function dataweights = get_data_weights(obj,N)
            % get the data point weights
            % there are N data points
            % returns an NX1 vector
            if ~isempty(obj.weight_exp_lambda)
                dist = abs(N-obj.weight_exp_curtime-(1:N));
                dataweights = obj.weight_exp_lambda.^(dist);
                dataweights = dataweights';
            else
                dataweights = ones(N,1);
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % cross validation training objective functions
        
        % negative log likelihood 
        function val = negloglik_train_helper(obj,Xall,yall,h,...
                                              train_inds_mat)
            % this calls negloglik with different arguments
            % this function is called when the bandwidth parameters are
            % being trained
            % inputs:
            %  - Xall : NXM matrix of predictor variables
            %  - yall : NX1 array of response variables
            %  - h    : 1XM array of bandwidths. h(1:end-1) are response
            %           bandwidths, h(end) is reponse bandwidth
            %  - train_inds_mat: NXL array
            %           train_inds_mat(:,ii)==1 are the training indexes for 
            %           fold ii = 1:L
            %           train_inds_mat(:,ii)==0 are the testing indexes for
            %           fold ii = 1:L
            %           train_inds_mat(:,ii)==-1 are not used (e.g. may be
            %           nans in data)
            % returns:
            %  - val: double, negative log liklihood
            valtot = 0;
            for ii = 1:size(train_inds_mat,2)
                train_inds = abs(train_inds_mat(:,ii)-1)<1e-10;
                test_inds  = abs(train_inds_mat(:,ii)-0)<1e-10;
                traindataweights = obj.get_data_weights(size(Xall,1));
                traindataweights = traindataweights(train_inds,1);
                val = obj.negloglik(Xall(train_inds,:),yall(train_inds,1),...
                                    Xall(test_inds,:), yall(test_inds,1),...
                                    h(1:end-1),h(end),traindataweights);
                valtot = valtot + val;
            end
            val = valtot;
        end
        function p = negloglik(obj,Xtrain,ytrain,Xtest,ytest,...
                pred_b,resp_b,traindataweights)
            % get the log liklihood of data points [Xtest,ytest] given
            % training data [Xtrain,ytrain] and bandwidths pred_b, resp_b
            % get the pdf of the testing points and calc the log lik
            p = obj.pdf(Xtest,ytest,Xtrain,ytrain,resp_b,pred_b,traindataweights);
            p = -sum(log(p));
        end
        
        % the inverse cdfs should be uniformly distributed
        function val = train_eval_inverse_cdf_helper(obj,...
                Xall,yall,h,train_inds_mat)
            % this calls train_eval_inverse_cdf using train_inds_mat to do
            % the cross fold validation
            % see negloglik_train_helper for inputs and returns
            valtot = 0;
            for ii = 1:size(train_inds_mat,2)
                train_inds = abs(train_inds_mat(:,ii)-1)<1e-10;
                test_inds  = abs(train_inds_mat(:,ii)-0)<1e-10;
                traindataweights = obj.get_data_weights(size(Xall,1));
                traindataweights = traindataweights(train_inds,1);
                val = obj.train_eval_inverse_cdf(...
                                    Xall(train_inds,:),yall(train_inds,1),...
                                    Xall(test_inds,:), yall(test_inds,1),...
                                    h(1:end-1),h(end),traindataweights);
                valtot = valtot + val;
            end
            val = valtot;
        end
        function val = train_eval_inverse_cdf(obj,Xtrain,ytrain,Xtest,ytest,...
                pred_b,resp_b,traindataweights)
            % this is an objective function used to perform training
            % inputs:
            %  - Xall : NXM matrix of predictor variables
            %  - yall : NX1 array of response variables
            %  - h    : 1XM array of bandwidths. h(1:end-1) are response
            %           bandwidths, h(end) is reponse bandwidth
            % returns:
            %  - val: double
            N = size(Xtest,1);
            p = obj.cdf(Xtest,ytest,Xtrain,ytrain,resp_b,pred_b,traindataweights);
            % p is the cdfs of all points yall
            % p should be evenly distributed along [0,1]
            % i.e. want sort(p) = linspace(1/2/N,1-1/2/N,N)
            p = sort(p)';
            ideal = linspace(1/2/N,1-1/2/N,N);
            val = sum( (p-ideal).^2 );
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    end
    methods (Access=private)
        function bandwidths = get_vector_bws(obj,xcenter,h)
            % xcenter is MX1 array
            % h is double
            % get the vector of bandwidths which depends on the nearest
            % neighbor distance
            if ~obj.do_bw_nn
                bandwidths = repmat(h,size(xcenter,1),1);
                return
            end
            xcenter = xcenter';
            ndims   = size(xcenter,1);
            npoints = size(xcenter,2);
            assert(size(h,1)==1)
            assert(size(h,2)==ndims)
            if npoints<20
                bandwidths=h*ones(size(xcenter'));
                disp('nope')
                a=asdf;
                return
            end
            knn = min(36,floor(npoints/2)); % half the number of nearest neighbors
            if mod(knn,2)
                knn = knn+1; % must be even
            end
            bandwidths = zeros(size(xcenter));
            for row=1:ndims
                [s,order] = sort(xcenter(row,:));
                m = conv(s,ones(1,2*knn+1)); % sum
                m = m(knn+1:end-knn); % sum
                m(1:knn) = m(1:knn)./((knn+1):(2*knn));
                m(knn+1:end-knn) = m(knn+1:end-knn)/(2*knn+1);
                m(end-knn+1:end) = m(end-knn+1:end)./((2*knn):-1:(knn+1));
                bandwidths(row,:) = h(row)*m(order);
            end
            assert(size(xcenter,1)==size(bandwidths,1))
            assert(size(xcenter,2)==size(bandwidths,2))
            bandwidths = max(bandwidths,1e-5);
            bandwidths = bandwidths';
        end
    end
    methods (Access=private, Static)
        function MDK = deep_copy(obj)
            % make a deep copy
            MDK = MD_KDE();
            % predictor kernels
            pred_kernel_cell = cell(1,length(obj.pred_kernels));
            for ii=1:length(obj.pred_kernels)
                k = obj.pred_kernels{ii};
                pred_kernel_cell{ii} = k.deep_copy();
            end
            MDK.input_predictor_kernels(pred_kernel_cell);
            % response kernel
            MDK.input_response_kernel(obj.resp_kernel.deep_copy());
            % bandwidths
            MDK.pred_bandwidths = obj.pred_bandwidths;
            MDK.resp_bandwidth  = obj.resp_bandwidth;
            % discretizations - cells are deep copied
            MDK.pred_bw_dis = obj.pred_bw_dis;
            MDK.resp_bw_dis = obj.resp_bw_dis;
            % training
            MDK.istrained = obj.istrained;
            % SavedKernel
            SK = obj.resp_kernel.SavedKernel;
            MDK.input_SavedKernel(SK.deep_copy());
            
        end
    end
    methods (Static)
        function test_train()
            % try training the KDE on some artificial data
            LB = [-3,1]; % dimension 1 = predictor
            UB = [-1,5]; % dimension 2 = response
            Nsamp = 24*14;
%             [Xall,yall,MD] = MD_KDE.test_make_data_and_instance(Nsamp,LB,UB);
            [Xall,yall,MD] = MD_KDE.test_make_data_and_instance_wind(Nsamp);
            MD.input_data(Xall,yall,false);
            
            
            SK = KDE_SAVEDKERNEL();
            SK = SK.load_beta();
            MD.input_SavedKernel(SK);
            
            MD.train();
            
            MD.plot_pcdf_2d(Xall,yall,Xall,yall,false); % pdf
            title(['pdf. pred h=',num2str(MD.pred_bandwidths),...
                ', resp h=',num2str(MD.resp_bandwidth)])
            MD.plot_pcdf_2d(Xall,yall,Xall,yall,true); % cdf
            title(['cdf. pred h=',num2str(MD.pred_bandwidths),...
                ', resp h=',num2str(MD.resp_bandwidth)])
            
            
        end
        function test_rank_histogram()
            % see what the rank histogram looks like 
            N = 24*130;
            [Xall,yall,MD,zp,op] = MD_KDE.test_make_data_and_instance_wind(N);
            
            good = Xall>=-0.2;
            N    = sum(good)
            Xall = Xall(good);
            yall = yall(good);
            
            SK = KDE_SAVEDKERNEL();
            SK = SK.load_beta();
            MD.input_SavedKernel(SK);
            MD.input_zero_one_percentiles(zp,op);
            
            % train on some data and get cdfs
            train_size = 24*60;
            test_size  = 24*7; % <= train_size
            test_offset = 24*30; % <=train_size-test_size
            niter = floor((N-train_size)/test_size)+1;
            c = zeros(test_size*niter,1);
            ct = 1;
            for ii=1:niter
%                 disp(['ii=',num2str(ii)])
                N1 = test_size*(ii-1)+1;
                N2 = N1+train_size-1;
%                 [N1,N2]
                Xtrain = Xall(N1:N2,:);
                ytrain = yall(N1:N2,1);
                % train
                MD.input_data(Xtrain,ytrain,false);
%                 disp('training')
                MD.train();
%                 disp('done training')
                % get cdfs
                N1 = N1+test_offset;
                N2 = N1+test_size-1;
                Xtest = Xall(N1:N2,:);
                ytest = yall(N1:N2,1);
                ctmp = MD.cdf(Xtest,ytest);
                c(ct:ct+length(ctmp)-1,1) = ctmp;
                ct = ct+length(ctmp);
            end
            
%             MD.input_data(Xall,yall,false);
%             disp('training')
%             MD.train();
%             disp('done training')
% %             pred_bw = 1e-3;
% %             resp_bw = 1e-3;
% %             MD.input_pred_bandwidths(pred_bw);
% %             MD.input_resp_bandwidth(resp_bw);
%             
% %             % make plot of correct cdf
% %             MD.plot_pcdf_2d(Xall,yall,Xall,yall,true); % cdf
% %             title('true cdf')

%             
%             N2 = 1;%round(N/4);
%             N3 = N;%round(N/2);%round(3*N/4);
%             disp('--------------')
%             c = MD.cdf(Xall(N2:N3,:),yall(N2:N3,1));
            
            % make a plot of the distribution of cdfs
            c = sort(c);
            figure
            hold on
            plot(c,'r-')
            q = 1:length(c);
            plot(q,q/length(q))
            title('cdfs. should be a line')
            xlabel('index of sorted cdfs of samples')
            ylabel('cdf of samples')
            
            % make a plot of the expected rank histogram
            Nrank = 100;
%             M = length(c);
%             c = linspace(1/2/M,1-1/2/M,length(c));
            ranks = zeros(1,Nrank+1);
            for ii=1:length(c)
                ranks = ranks + binopdf(0:Nrank,Nrank,c(ii));
            end
            figure
            hold on
            plot(ranks*(Nrank+1)/length(c),'*-')
            plot([1,Nrank+1],[1,1],'k')
            ci95 = icdf('Binomial',[0.05,0.95],N,1/(Nrank+1));
            ci95 = ci95/(N/(Nrank+1));
            plot([1,Nrank+1],ci95(1)*[1,1],'r')
            plot([1,Nrank+1],ci95(2)*[1,1],'r')
            title('normalized expected rank histogram given cdfs. should =1')
            xlabel('rank')
            
            
        end
        function test_inverse_cdf()
            % test the cdf and inverse_cdf function
            
            N = 24*130;
            [Xall,yall,MD,zp,op] = MD_KDE.test_make_data_and_instance_wind(N);
            SK = KDE_SAVEDKERNEL();
            SK = SK.load_beta();
            MD.input_SavedKernel(SK);
            MD.input_zero_one_percentiles(zp,op);
            MD.input_data(Xall,yall,false);
            
            lambda = (0.01)^(1/N);
            cur_time = N;
            MD.input_data_weights_exp(lambda,cur_time);
            
            MD.train();
            
            % now draw samples from response variable
            % first do all data points
            figure
            hold on
            pred = 0.5;
            M = 100;
            respax = linspace(0,1,M);
            % get the analytical cdf
            p = MD.cdf(pred*ones(M,1),respax');
            plot(respax,p,'r')
            % now calculate inverse cdf
            y = MD.inverse_cdf(pred*ones(M,1),respax');
            % plot the cdf using inverse_cdf
            plot(y,respax,'k.--')
            % plot cdf of all training samples
            ys = sort(yall);
            plot(ys,(1:size(ys,1))/size(ys,1),'b')
            plot(ys,-0.05,'b.')
            xlabel('response variable')
            ylabel('cdf')
            legend('cdf of x-axis|pred=0.5',...
                'inverse cdf of y-axis|pred=0.5',...
                'cdf of alls data points','data points')
            title({'Checking cdf and inverse cdf',...
                'red line (cdf) and black dots (inverse cdf) should be the same',...
                'cdf and inverse cdf are conditioned on predictor variable=0.5'})
            
            
            figure
            hold on
            nintervals = 9;
            predlist = linspace(1/2/nintervals,1-1/2/nintervals,nintervals);
            delta = predlist(2)-predlist(1);
            M = 100;
            respax = linspace(0,1,M);
            M2 = 100;
            rax2 = linspace(0,1,M2);
            delta2 = rax2(2)-rax2(1);
            for ii=1:9%nintervals
                subplot(3,3,ii)
                hold on
                pred = predlist(ii);
%                 pred = max(pred+delta/2,0); % SHIFT #1
                % get the analytical cdf
                p = MD.cdf(pred*ones(M,1),respax');
                plot(respax,p,'r')
                % now draw lots of samples
                y = MD.inverse_cdf(pred*ones(M,1),respax');
                % plot the cdf using inverse_cdf
                plot(y,respax,'k.--')
                title(['pred=',num2str(pred)])
                xlabel('response variable')
                ylabel('cdf')
                % plot the empirical cdf
%                 pred = pred-delta/2; % SHIFT #2
                inbin = abs(Xall-pred)<delta/2;
                ninbin = sum(inbin);
                if ninbin>0
                    yinbin = yall(inbin);
                    yinbin = sort(yinbin)';
                    indlist = yinbin/delta2;
                    indlist = max(min(round(indlist),M2),1);
%                     rax2
                    p2 = zeros(1,M2);
                    for jj=1:ninbin
                        p2(indlist(jj):end) = p2(indlist(jj):end)+1;
                    end
                    p2 = p2/ninbin;
                    plot(rax2,p2,'b')
                    plot(yinbin,-0.05,'b.')
                end
            end
            
        end
        function test_log_likelihood()
            % plot the log liklihood on some test data to see what it looks
            % like
            Ntrain = 24*21; % number of samples for training
            Ntest  = 24*11; % number of samples for testing (computing log liklihood)
            N = Ntrain+Ntest;
            [Xall,yall,MD] = MD_KDE.test_make_data_and_instance_wind(N);
            
            % manually create the weights based on input lambda
            % normally  would just use initialization
            %   MD.input_data_weights_exp(lambda,curtime)
            lambda = 0.01^(1/N);
            curtime = N;
            dataweights = lambda.^( abs( N-curtime-(1:N)' ) );
            
            train_inds = zeros(N,1);
            train_inds(randsample(N,Ntrain),1) = 1;
            train_inds = train_inds>0;
            test_inds  = ~train_inds;
            Xtrain = Xall(train_inds,:);
            Xtest  = Xall(test_inds,:);
            ytrain = yall(train_inds,1);
            ytest  = yall(test_inds,1);
            dataweightstrain = dataweights(train_inds,1);
            
            SK = KDE_SAVEDKERNEL();
            SK = SK.load_beta();
            MD.input_SavedKernel(SK);
            
            minbw = 1e-4; % minimum bandwidth parameter
            maxbw = 1;   % maximum
            ngrid = 10; % number of samples in grid
            % do a grid plot of the log liklihood
            bwgrid = exp(linspace(log(minbw),log(maxbw),ngrid));
            [Xgrid,Ygrid] = meshgrid(bwgrid,bwgrid);
            Zgrid = zeros(size(Xgrid)); % log liklihood
            
            figure
            hold all
            leg = cell(1,ngrid);
            size(Xall)
            max(Xall)
            for row=1:ngrid
                for col=1:ngrid
                    
                    bwpred = Xgrid(row,col); % predictor bandwidth
                    bwresp = Ygrid(row,col); % response bandwidth
%                     Zgrid(row,col) = ...
%                         MD.negloglik(Xtrain,ytrain,Xtest,ytest,...
%                                          bwpred,bwresp);
%                     Zgrid(row,col) = MD.train_eval_inverse_cdf(...
%                         Xall,yall,[bwpred,bwresp]);
                    if row==col
                        p = MD.cdf(Xall,yall,Xall,yall,bwresp,bwpred,dataweights);
                        p = sort(p);
                        plot(p)
                        leg{row} = num2str(bwresp);
                    end
%                     Zgrid(row,col) = MD.train_eval_uniform_dist(...
%                         Xall,yall,[bwpred,bwresp]);
                    Zgrid(row,col) = MD.negloglik(Xtrain,ytrain,Xtest,ytest,...
                                    bwpred,bwresp,dataweightstrain);
                    
                end
            end
            legend(leg)
            Zgrid
            figure()
            surf(Xgrid,Ygrid,Zgrid)
            xlabel('predictor bandwidth')
            ylabel('response bandwidth')
            zlabel('negative log likelihood')
            % now plot the pdf and cdf with three sets of bandwidths
            MD.input_data(Xtrain,ytrain,false);
            % minimum bw
            MD.input_pred_bandwidths(minbw);
            MD.input_resp_bandwidth(minbw);
            MD.plot_pcdf_2d(Xtrain,ytrain,Xtest,ytest,false); % pdf
            title(['Min. pdf with pred, resp bandwidth = ',num2str(minbw)])
            MD.plot_pcdf_2d(Xtrain,ytrain,Xtest,ytest,true); % cdf
            title(['Min. cdf with pred, resp bandwidth = ',num2str(minbw)])
            % maximum bw
            MD.input_pred_bandwidths(maxbw);
            MD.input_resp_bandwidth(maxbw);
            MD.plot_pcdf_2d(Xtrain,ytrain,Xtest,ytest,false); % pdf
            title(['Max. pdf with pred, resp bandwidth = ',num2str(maxbw)])
            MD.plot_pcdf_2d(Xtrain,ytrain,Xtest,ytest,true); % cdf
            title(['Max. cdf with pred, resp bandwidth = ',num2str(maxbw)])
            % best bw
            [~,ind] = min(reshape(Zgrid,1,ngrid^2));
            pb = Xgrid(ind);
            rb = Ygrid(ind);
            MD.input_pred_bandwidths(pb);
            MD.input_resp_bandwidth(rb);
            MD.plot_pcdf_2d(Xtrain,ytrain,Xtest,ytest,false); % pdf
            title(['Best. pdf with pred bw=',num2str(pb),' resp bw=',num2str(rb)])
            MD.plot_pcdf_2d(Xtrain,ytrain,Xtest,ytest,true); % cdf
            title(['Best.cdf with pred bw=',num2str(pb),' resp bw=',num2str(rb)])
            % plot the theoretical rank histogram
            cdflist = MD.cdf(Xall,yall);
            nranks = 20;
            MD.plot_rank_hist(cdflist,nranks)
            title(['rank histogram with pred bw=',num2str(pb),' resp bw=',num2str(rb)])
            
        end
        function test()
            % two dimensional test
            LB = [0,0]; % dimension 1 = predictor
            UB = [5,8]; % dimension 2 = response
            % gnerate data
            N = 100;
            [X,y,MD] = MD_KDE.test_make_data_and_instance(N,LB,UB);
            MD.input_data(X,y,false);
            MD.train();
        end
    end
    methods (Static)
        function [X,y, MD] = test_make_data_and_instance(N,LB,UB)
            % N = number of samples
            % LB, UB = 1X2 lower bound and upper bound. LB(1), UB(1) is
            % predictor variable, LB(2), UB(2) is the response variable
            d1 = UB(1)-LB(1);
            X = rand(N,1)*d1+LB(1);
            d2 = UB(2)-LB(2);
            y = 0.8*d2/d1*(X-LB(1))+LB(2)+d2*0.1+(rand(N,1)*0.5-0.25)*d2;
            y = min(y,UB(2));
            y = max(y,LB(2));
            
            MD = MD_KDE();
            k1 = KDEK_BETA(LB(1),UB(1));
            k2 = KDEK_BETA(LB(2),UB(2));
            MD.input_predictor_kernels({k1});
            MD.input_response_kernel(k2);
        end
        function [X,y,MD,zp,op] = test_make_data_and_instance_wind(N)
            % N = number of samples
            % LB, UB = 1X2 lower bound and upper bound. LB(1), UB(1) is
            % predictor variable, LB(2), UB(2) is the response variable
            start_date = datenum(2013,7,1,0,0,0);
            end_date   = datenum(2013,7,1,round(1.2*N),0,0);
            [r,f] = DATA_HANDLER.get_seq_data_inclusive(start_date,end_date);
            y = r(2:end);
            X = f(1:end-1,1); 
            nanrows = isnan(y) | any(isnan(X),2);
            y = y(~nanrows);
            X = X(~nanrows,:);
            y = y(1:N);
            X = X(1:N,:);
            
            MD = MD_KDE();
            k1 = KDEK_BETA(0,1);
            k2 = KDEK_BETA(0,1);
            MD.input_predictor_kernels({k1});
            MD.input_response_kernel(k2);
            
            zp = @(x) zeros(size(x,1),1);
            op = @(x) ones(size(x,1),1);
        end
    end
end