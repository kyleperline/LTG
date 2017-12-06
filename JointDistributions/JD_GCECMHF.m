classdef JD_GCECMHF < JD_GC_GENERATE
    % Gaussian copula exponential covariance match historical fluctuations
    % covariance between marginal distributions ii and jj is
    %   exp(-|ii-jj|*myeps)
    % myeps is the single hyperparameter that needs to be selected
    % myeps is optimized to match the observation fluctuations:
    %  let obs be NX1 sequence of N historical values
    %  create the historical pdf of fluctuations obs(1,2:N)-obs(1,1:N-1)
    %  this historical pdf is the 'true' value
    % myeps is optimized by finding the value such that when M scenarios
    %  are generated their scenario fluction pdf best matches the
    %  historical pdf
    properties
        myeps % trained value
        hist_fluc % 1XN historical fluctuations
        do_repeated_train % boolean, whether or not to train for myeps at 
                          % each iteration
        mX % margcellX used to do training
        coveps
    end
    methods
        function obj = JD_GCECMHF()
            obj.do_repeated_train = true;
        end
        function d = get_description(obj)
            d = ['JD_GCECMHF ',...
                    'covexp:',num2str(obj.coveps)];
        end
        function obs = input_historical_obs(obj,obs)
            assert(size(obs,2)==1,'obs should be NX1')
            obj.hist_fluc = sort(obs(2:end,1)-obs(1:end-1,1))';
        end
        function obj = input_cov_eps(obj,coveps)
            obj.coveps = coveps;
        end
        function obj = train(obj, prev_MD_training, MDtypelist, ...
                prev_MD_training_cell,numdist)
            if nargin<5
                numdist = [];
            end
            obj.train_MD(prev_MD_training, MDtypelist, prev_MD_training_cell, numdist);
            obj.set_eps_sigma(obj.coveps);
        end
        function obj = train_clear(obj)
        end
    end
    methods (Access = private)
        function [v,bincenters,bincounts] = eps_objfun(obj,myeps,nbins)
            % the objective function used to select myeps
            if nargin<3
                get_hist = false;
                bincenters = [];
                bincounts = [];
            else
                get_hist = true;
            end
            obj.set_eps_sigma(myeps);
            size(obj.mX{1})
            Nsamp = min(50,size(obj.mX{1},1));
            Nrows = size(obj.mX{1},1);
            myrows = randsample(Nrows,Nsamp);
            mXtmp = cell(1,length(obj.mX));
            for ii=1:length(obj.mX)
                q = obj.mX{ii};
                mXtmp{ii} = q(myrows,:);
            end
            s = obj.generate(mXtmp);
            f = s(:,2:end)-s(:,1:end-1); % fluctuations
            f = sort(f,2);
            v = 0;
            p = 2; % 2 norm
            size(f)
            for row=1:size(f,1)
                v = v+obj.fast_ecdf_integrate(obj.hist_fluc,f(row,:),p);
            end
            if get_hist
                f = reshape(f,1,size(f,1)*size(f,2));
                figure
                [bincounts,bincenters] = hist(f,nbins);
                close
            end
        end
        function set_eps_sigma(obj,myeps)
            % set myeps 
            % make and set sigma
            % exponential distance
            obj.myeps = myeps;
            M = length(obj.MDarray);
            d = zeros(M,M);
            for ii=1:M
                d(ii,:) = abs((1:M)-ii);
            end
            obj.sigma = exp(-d/myeps);
        end
        function plot_eps_objfun(obj,epslist)
            % plot the objective function
            if nargin==1
                N = 5;
                mineps = 1;
                maxeps = 20;
                epslist = exp(linspace(log(mineps),log(maxeps),N));
            end
            N = length(epslist);
            v = zeros(1,N);
            nbins = 50;
            figure
            [hist_bincounts,hist_bincenters] = hist(obj.hist_fluc,nbins);
            close
            eps_bincenters  = zeros(N,nbins);
            eps_bincounts   = zeros(N,nbins);
            for ii=1:N
                [v(ii),eps_bincenters(ii,:),eps_bincounts(ii,:)] = ...
                    obj.eps_objfun(epslist(ii),nbins);
            end
            figure
            hold all
            normtmp = sum(hist_bincounts)*(hist_bincenters(2)-hist_bincenters(1));
            plot(hist_bincenters,hist_bincounts/normtmp)
            for ii=1:N
                normtmp = sum(eps_bincounts(ii,:))*(eps_bincenters(ii,2)-eps_bincenters(ii,1));
                plot(eps_bincenters(ii,:),eps_bincounts(ii,:)/normtmp )
            end
            legend(cat(2,'hist',cellstr(num2str(epslist'))'))
            title('fluctuation distributions')
            figure
            plot(epslist,v,'r*-')
            title('epsilon objective function')
            xlabel('epsilon')
            ylabel('objective function')
        end
    end
    methods (Static)
        function test_fast_ecdf_integrate()
            s1 = [0 1 1 5];
            s2 = [0 2 6];
            p = 2;
            v = JD_GCECMHF.fast_ecdf_integrate(s1,s2,p);
            vtrue = (1-0)*(1/4-1/3)^p + (2-1)*(3/4-1/3)^p + ...
                    (5-2)*(3/4-2/3)^p + (6-5)*(4/4-2/3)^p;
            assert(v==vtrue)
            disp('fast_ecdf_integrate probably works')
        end
        function v = fast_ecdf_integrate(s1,s2,p)
            % integrate two empirical cdfs
            % inputs:
            %  s1, s2: 1XN1 and 1XN2 SORTED arrays of real numbers
            %  p: >0
            % output:
            %  v = integral_{x=-inf}^{inf} (ecdf(s1)-ecdf(s2)^p dx
            N1 = length(s1);
            N2 = length(s2);
            dy1 = 1/N1;
            dy2 = 1/N2;
            s3  = [ [ dy1*ones(1,N1);s1 ] , [ -dy2*ones(1,N2);s2 ] ];
            s3 = sortrows(s3',2);
            y = cumsum(s3(:,1));
            
            if ~(abs(y(end))<1e-12)
                y(end)
                ~(abs(y(end))<1e-12)
                [N1,dy1]
                [N2,dy2]
                figure
                plot(s3(:,2),s3(:,1))
                assert(abs(y(end))<1e-12)
                a=asdf;
            end
            y = y.^p;
            v = (s3(2:end,2)-s3(1:end-1,2)).*y(1:end-1,1);
            v = sum(v);
        end
        
    end
    
    
end