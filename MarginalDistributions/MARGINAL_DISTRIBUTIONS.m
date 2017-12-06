classdef MARGINAL_DISTRIBUTIONS < handle
    % This class defines the basic interface methods required for
    % estimating marginal distributions.
    % The basic setup is:
    % Input:
    %  - X: nXd array of predictor variables
    %  - y: nX1 array of response variables
    %  - estimate pdf of y conditioned on X
    %
    % Child classes implement various techniques of estimating the pdf
    % (e.g. quantile regression, kernel density estimation, ...)
    % 
    % An example of a general use of a child class would be:
    %
    % >> MD = Child_Class();
    % >> % any other MD initializations
    % >> 
    % >> % make up some data
    % >> n=100; d=3; % n=number of data points, d=dimension of X
    % >> X=rand(n,d); y=sum(X,2)+0.1*randn(n,1);
    % >> miny=-1; maxy=d+1;
    % >> zerop = @(x) size(x,1)*miny; % zeroth percentile
    % >> onep  = @(x) size(x,1)*maxy; % 100th percentile
    % >> appendData = false; 
    % >>
    % >> % input data and train:
    % >> MD.input_zero_one_percentile(zerop,onep);
    % >> MD.input_data(X,y,appendData);
    % >> MD.train();
    % >> p = MD.cdf(X,y); % p is nX1 array, p(ii) is percentile of y(ii)
    %                       conditioned on X(ii,:)
    % >> y = MD.inverse_cdf(X,p); % inverse cdf
    properties
        X % predictor variables, nXd
        y % response variables, nX1
        zerop % function handle; see input_zero_one_percentiles
        onep  % function handle; see input_zero_one_percentiles
        ntrained     % non-negative integer, number of data points the m.d.  
                     % was most recently trained with
        ntrainedprev % non-negative integer, number of data points the m.d.
                     % was second-most recently trained with
        weight_exp_lambda
        weight_exp_curtime
        train_MD_every_nts
        nts_since_last_trained 
    end
    methods
        function obj = MARGINAL_DISTRIBUTIONS()
            obj.ntrained     = 0;
            obj.ntrainedprev = 0;
            obj.clear_data();
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Data input and manipulation
        
        function obj = input_data(obj,X,y,append)
            % input initial data
            assert(size(X,1)==size(y,1),'number of rows in X, y must be the same');
            % update is a boolean
            if append
                % append
                obj = obj.append_data(X,y);
            else
                % replace
                obj = obj.replace_data(X,y);
            end
        end
        function obj = replace_data(obj,X,y)
            assert(~isempty(X),'do not call replace_date(X,y) with empty X');
            % reset data
            obj.X = X;
            obj.y = y;
            % update trained state
            obj.ntrained     = 0;
            obj.ntrainedprev = 0;
        end
        function obj = append_data(obj,Xappend,yappend)
            assert(~isempty(obj.X),...
              'do not call append_data when obj.X is empty; try update=false');
            % append new data
            if size(Xappend,1)>0
                obj.X = [obj.X;Xappend];
                obj.y = [obj.y;yappend];
            end
        end
        function obj = clear_data(obj)
            % clear all data
            obj.X = int16.empty(0);
            obj.y = int16.empty(0);
            % update trained state
            obj.ntrained     = 0;
            obj.ntrainedprev = 0;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Get percentiles of the input data, input 0 and 100 percentiles
        
        function obj = input_zero_one_percentiles(obj,zerop,onep)
            % input function handles for 0 and 100 percentiles
            % inputs:
            %  - zerop: function handle, y = zerop(x)
            %           x: mXd predictor
            %           y: mX1 response s.t. cdf(y(ii);x(ii)) = 1
            %  - onep : function handle, y = onep(x)
            %           x: mXd predictor
            %           y: mX1 response s.t. cdf(y(ii);x(ii)) = 0
            obj.zerop = zerop;
            obj.onep  = onep;
        end
        
        function p = get_percentiles_all(obj)
            % get the percentiles of all data that has been input, whether
            % or not the marginal distribution was trained on all the data
            p = obj.cdf(obj.X,obj.y);
        end
        function p = get_percentiles_trained(obj)
            % get the percentiles of all the data that the marginal
            % distribution was trained on
            p = obj.cdf(obj.X(1:obj.ntrained,:),obj.y(1:obj.ntrained));
        end
        function p = get_percentiles_trained_last(obj)
            % get the percentiles of the data that that has only been
            % trained on once
            nt  = obj.ntrained;
            ntp = obj.ntrainedprev;
            p = obj.cdf(obj.X(ntp+1:nt,:),obj.y(ntp+1:nt,1));
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % input some options about how to handle data points and training
        
        function obj = input_data_weights_exp(obj,lambda,cur_time)
            % weight each of the data points 
            % inputs:
            %  - lambda: >0, weight factor
            %  - cur_time: positive integer
            % The data points should be weighted as follows 
            % (though note that it is up to the marginal distribution
            % child classes to implement this)
            % the current stored data points are obj.X and obj.y, which are
            % NXM and NX1
            % the weight of data point (obj.X(ii,:),obj.y(ii,:)) should be
            %   lambda^( | (N-cur_time) - ii | )
            % this is an exponential scaling
            obj.weight_exp_lambda = lambda;
            obj.weight_exp_curtime = cur_time;
        end
        function obj = input_train_every(obj,nts_train_every)
            % if the marginal distributions are trained only every few time
            % steps
            % Note: it is up to the MARGINAL_DISTRIBUTION children to
            % implement this
            obj.train_MD_every_nts = nts_train_every;
        end
        
    end
    methods (Sealed)
        function obj = train_helper(obj)
            % update ntrained and ntrainedprev
            obj.ntrainedprev = obj.ntrained;
            obj.ntrained     = size(obj.X,1);
            % also check that the 0 and 100 percentile functions aren't
            % violated
            myy = obj.y;
            upper = obj.onep(obj.X);
            % handle NaN problem
            upper(isnan(upper)) = Inf;
            myy(isnan(obj.y))= - Inf;
            assert(all(upper>=myy),...
                    'data violdated the one hundred percentile')
            lower = obj.zerop(obj.X);
            % handle NaN problem
            lower(isnan(lower)) = -Inf;
            myy(isnan(obj.y))= Inf;
            assert(all(lower<=myy),...
                    'data violdated the zero percentile');
        end
        function obj = cdf_helper(obj,X,y)
            % call this function at the beginning of cdf(X,y) and at the 
            % end of inverse_cdf(X,p)
            myy = y;
            upper = obj.onep(X);
            upper(isnan(upper)) = Inf;
            myy(isnan(y)) = -Inf;
            assert(all(upper>=myy),...
                    'data violdated the one hundred percentile')
            lower = obj.zerop(X);
            lower(isnan(lower)) = -Inf;
            myy(isnan(y)) = Inf;
            assert(all(lower<=myy),...
                    'data violdated the zero percentile');
        end
        function obj = inverse_cdf_helper(obj,X,y)
            % call this function at the beginning of cdf(X,y) and at the 
            % end of inverse_cdf(X,p)
            upper = obj.onep(X);
            e = upper-y;
            assert(all(upper+1e-6>=y),...
               ['data violdated the one hundred percentile, error=',num2str(max(e))])
            lower = obj.zerop(X);
            e = lower-y;
            assert(all(lower-1e-6<=y),...
                ['data violdated the zero percentile, error=',num2str(max(e))]);
        end
        
    end
    methods (Abstract)
        % The following are the methods that child classes must define
        name = get_name(obj)
            % return a string that describes this class
        obj = train(obj)
            % train some model in order to perform later functions
            % this needs to call obj.train_helper();
        obj = clear_train(obj)
            % clear (reset) the trained model
        p = cdf(obj,X,y);
            % calculate the cdf 
            % inputs:
            %  - X: mXd array of predictor variables
            %  - y: mX1 array of response variables
            % returns:
            %  - p: mX1 array of cdf; p(ii) = cdf(y(ii);x(ii))
        y = inverse_cdf(obj,X,p)
            % calculate the inverse cdf
            % inputs:
            %  - X: mXd array of predictor variables
            %  - p: mX1 array of percentiles, 0<=p(ii)<=1
            % returns:
            %  - y: mX1 array of response variables
    end
    methods
        function plot_quantiles_1d(obj)
            assert(size(obj.X,2)==1,'only call this function when the predictor variables are 1-dimensional')
            xmin = min(obj.X);
            xmax = max(obj.X);
            edge = 0.*(xmax-xmin); % percent
            npts = 20;
            xquant = linspace(xmin-edge,xmax+edge,npts)';
            xquant = [0; .005; 0.01; 0.015; xquant(2:end)];
            npts = numel(xquant);
%             qlist = [0,0.001,0.01,0.05, 0:0.1:0.9, 0.95,0.99,0.995,0.999,1];
            qlist = [0 0.25 0.45 0.55 0.75 1];
            yquant = zeros(length(xquant),length(qlist));
            for jj=1:length(qlist)
                yquant(:,jj) = obj.inverse_cdf(xquant,qlist(jj)*ones(npts,1));
            end
            obj.plot_quantiles_1d_general(obj.X,obj.y,qlist,xquant,yquant);
            xlabel('predictor')
            ylabel('response')
        end
        function plot_quantiles_1d_slice(obj,dim,centers)
            % plot a 1D slice of dimension dim holding other dimension
            % values constant
            % centers is 1Xlength(obj.X) and is the constant value
            assert(size(centers,1)==1)
            assert(size(centers,2)==size(obj.X,2))
            xmin = min(obj.X(:,dim));
            xmax = max(obj.X(:,dim));
            edge = 0.1*(xmax-xmin); % percent
            npts = 20;
            xquant = linspace(xmin-edge,xmax+edge,npts)';
            qlist = [0,0.001,0.01,0.05, 0:0.1:0.9, 0.95,0.99,0.995,0.999,1];
            yquant = zeros(length(xquant),length(qlist));
            Xval = ones(length(xquant),1)*centers;
            Xval(:,dim) = xquant;
            for jj=1:length(qlist)
                yquant(:,jj) = obj.inverse_cdf(Xval,qlist(jj));
            end
            obj.plot_quantiles_1d_general([],[],qlist,xquant,yquant);
        end
    end
    methods (Static)
        function plot_rank_hist(cdflist,nranks)
            % plot the theoretical rank histogram
            % inputs:
            %  - cdflist: NX1 array of cdfs, 0<=cdflist(ii)<=1
            %  - nranks : integer
            figure()
            hold all
            myranks = zeros(1,nranks+1);
            for ii=1:length(cdflist)
                myranks = myranks + binopdf(0:nranks,nranks,cdflist(ii));
            end
            myranks = myranks/length(cdflist);
            plot(myranks)
            plot([1,nranks+1],[1,1]/(nranks+1))
            legend('theoretical','ideal')
            xlabel('rank')
            ylabel('theoretical proportion')
        end
        function plot_quantiles_1d_general(x,y,qlist,xquant,yquant)
            % plot quantiles 
            % inputs:
            %  - x     : nX1 array of predictors
            %  - y     : nX1 array of response
            %  - qlist : 1XQ array of quantiles that are plotted
            %  - xquant: mX1 array, x-axis of plotted quantiles 
            %  - yquant: mXQ array, yquant(:,ii) is qlist(ii) quantile
            figure()
            hold on
            xx = [xquant',fliplr(xquant')];
            for ii=1:length(qlist)-1
                plot(xquant,yquant(:,ii))
                yy = [yquant(:,ii)',fliplr(yquant(:,ii+1)')];
                ftmp = fill(xx,yy,[0,abs(ii-size(yquant,2)/2)/size(yquant,2)*2,1]);
                set(ftmp,'EdgeColor','none')
            end
            plot(x,y,'r.')
        end
        function [x,y,zerop,onep] = test_generate_samples_1d(xmax,N)
            % a test function to draw some random samples from a
            % distribution with 1 predictor variable. 
            % inputs:
            %  - xmax: positive scalar, predictor variables are in [0,xmax]
            %  - N   : positive scalr, number of samples to draw
            % returns:
            %  - x    : NX1 predictor variables
            %  - y    : NX1 response variables
            %  - zerop: function handle, returns 0 percentile
            %  - onep : function handle, returns 100 percentile
            %
            % the pdf f is
            %    f(y;x) = { y/(a^2)            if 0<=y<=a
            %             { 1/(a) - y/(a^2)    if a<=y<=2*a
            %    where a = (1+x^2)
            % f(y;x,t) is therefore a triangle with width 2*a and height 
            % 1/a, where the maximum is achieved at a
            x = rand(N,1)*xmax;
            a = (1+x.^2);
            u = rand(N,1); % uniform random 
            % now calculate y such that \int_0^y f(y;a) dy = u
            y = zeros(N,1);
            y(u<=0.5) = (2*u(u<=0.5).*a(u<=0.5).^2).^0.5;
            y(u>0.5)  = a(u>0.5)*2 - (2*(1-u(u>0.5)).*a(u>0.5).^2).^0.5;
            zerop = @(x) zeros(size(x,1),1);
            onep  = @(x) 2.2*(1+xmax^2)*ones(size(x,1),1);
        end
        
    end
end