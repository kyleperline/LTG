classdef KDE_KERNEL %< handle
    % a one dimensional kernel
    properties
        % bounds on support of distribution, doubles
        lower_bound % -Inf<=lower_bound<upper_bound
        upper_bound % upper_bound<=Inf
        % boolean on whether this distribution takes circular inputs
        is_circular
        SavedKernel % class instance of pre-comuted kernel
        saved_kernel % boolean of whether to use precomputed kernel or not
    end
    methods
        function obj = KDE_KERNEL()
            obj.saved_kernel = false;
        end
        function bd = get_lower_bound(obj)
            bd = obj.lower_bound;
        end
        function bd = get_upper_bound(obj)
            bd = obj.upper_bound;
        end
        function isc = get_iscircular(obj)
            isc = obj.is_circular;
        end
        function obj = input_SavedKernel(obj,SavedKernel)
            obj.SavedKernel = SavedKernel;
            obj.saved_kernel = true;
        end
        function test_integral_is_1(obj)
            % use numerical integration to ensure the kernel is a pdf 
            % (i.e. integral is 1)
            % only works if -Inf<lower_bound<=upper_bound<Inf
            assert(-Inf<obj.lower_bound)
            assert(obj.upper_bound<Inf)
            LB = obj.lower_bound;
            UB = obj.upper_bound;
            N = 1;
            xtest = rand(N,1)*(UB-LB)+LB;
            bwlb = 1e-8; % bandwidth lower bound
            bwub = 1;    % bandwidth upper bound
            bwtest = rand(N,1)*(bwub-bwlb)+bwlb;
            dMC = 5e4; 
            tol = 1e-2;
            for ii=1:N
                % do monte carlo integration
                e = 2*tol;
                sumx = 0;
                sumxx = 0;
                ntot = 0;
                while e>tol
                    xtmp = rand(dMC,1)*(UB-LB)+LB;
                    pdfs = obj.pdf(xtmp,xtest(ii),bwtest(ii));
                    assert(all(pdfs>=0),['got pdf <0. xcenter=',...
                        num2str(xtest(ii)),' bw=',num2str(bwtest(ii))])
                    assert(size(pdfs,1)==dMC,'pdf returned wrong size')
                    assert(size(pdfs,2)==1,'pdf returned wrong size')
                    ntot = ntot+dMC;
                    sumx = sumx + sum(pdfs);
                    sumxx = sumxx + sum(pdfs.^2);
                    e = sumxx/ntot-(sumx/ntot)^2;
                    e = sqrt(e/ntot);
                end
                integral = sumx*(UB-LB)/ntot;
                assert(abs(integral-1)<10*tol,'pdf integral might not be not 1; try rerunning')
            end
            disp('pdf probably integrates to 1')
        end
        function test_plot_kernel(obj,bwmin,bwmax)
            % do some plots
            % for each bandwidth in bandwidth_list
            %   make a subplot of kernels with center in centers_list
            %   (i.e. plot pdf)
            % end
            if nargin<2
                bwmin = 1e-8;
            end
            if nargin<3
                bwmax = 2;
            end
            nplots = 8;
            bwlist = exp(linspace(log(bwmin),log(bwmax),nplots));
            xcenters = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99];
            xcenters = xcenters*(obj.upper_bound-obj.lower_bound)+obj.lower_bound;
            xax = linspace(obj.lower_bound,obj.upper_bound,1000)';
            n = min(nplots,4); % number subplot cols
            m = ceil(nplots/n); % number subplot rows
            figure()
            for ii=1:nplots
                bw = bwlist(ii);
                p = obj.weight(xax,xcenters',bw,false);
                subplot(m,n,ii)
                hold on
                plot(xax,p)
                title(['bw=',num2str(bw)])
            end
        end
        function test_plot_kernel_cdf(obj,bwmin,bwmax)
            % do some plots showing the kernel, pdf (kernel normalized to
            % integrate to 1), and cdf
            if nargin<2
                bwmin = 1e-8;
            end
            if nargin<3
                bwmax = 2;
            end
            nfigs = 4;
            bwlist = exp(linspace(log(bwmin),log(bwmax),nfigs));
            xcenters = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99];
            xcenters = xcenters*(obj.upper_bound-obj.lower_bound)+obj.lower_bound;
            xax = linspace(obj.lower_bound,obj.upper_bound,1000)';
            n = min(length(xcenters),3); % number subplot cols
            m = ceil(length(xcenters)/n); % number subplot rows
            for ii=1:nfigs
                figure()
                bw = bwlist(ii);
                for jj=1:length(xcenters)
                    xc = xcenters(jj);
                    myweight = obj.weight(xax,xc,bw,false);
                    mypdf    = obj.weight(xax,xc,bw,true);
                    mycdf    = obj.cdf(xax,xc,bw);
                    subplot(m,n,jj)
                    hold on
                    plot(xax,myweight,'r')
                    plot(xax,mypdf,'b')
                    plot(xax,mycdf,'k')
                    title(['bw=',num2str(bw),',xc=',num2str(xc)])
                end
            end
        end
    end
    methods (Static)
        function y2 = myfastinterp1(deltax,Y,X)
            % a faster 1D interpolation (compared to interp1) on a
            % simplified case
            % inputs:
            %  - deltax: positive double
            %  - Y     : 1XN array
            %            the function being approximated is defined by
            %            X = 0:deltaX:(N-1)*deltax (which is 1XN)
            %  - X     : 1XM array, places to interpolate
            %            require 0<=X(ii)<=(N-1)*deltax
            % returns:
            %  - y2: 1XM array of linearly interpolated values
            assert(size(Y,1)==1)
            N = size(Y,2);
            assert(size(X,1)==1)
            M = size(X,2);
            assert(all(X>=0))
            assert(all(X<=(N-1)*deltax))
            inds = floor(X/deltax)+1;
            % if inds(ii)==N, then y2(ii)=Y(end)
            notN = inds<N;
            y2 = zeros(1,M);
            y2(~notN) = Y(end);
            inds2 = inds(notN);
            diff = X(notN) - (inds2-1)*deltax; % 
            y2(notN) = Y(inds2) + (Y(inds2+1)-Y(inds2)).*diff/deltax;
        end
        function test_myfastinterp1()
            N = 5;
            deltax = rand(1)+0.4;
            Y = rand(1,N);
            X = linspace(0,(N-1)*deltax,1000);
            y2 = KDE_KERNEL.myfastinterp1(deltax,Y,X);
            xax = 0:deltax:(N-1)*deltax;
            figure
            hold on
            plot(X,y2,'b')
            plot(xax,Y,'r*')
            y2 = KDE_KERNEL.myfastinterp1(deltax,Y,xax);
            assert(max(abs(y2-Y))<1e-12)
        end
    end
    methods (Abstract)
        p = cdf(obj,x,xcenters,bandwidth)
          % get cdf - cumulative distribution function of weight (i.e.
          % weight divided by integral of weight)
          % inputs:
          %  - x        : nX1 array
          %  - xcenters : mX1 array
          %  - bandwidth: double
          % returns:
          %  - p: nXm array
          %       p(ii,jj) is the cdf (between 0 and 1 inclusive) of x(ii)
          %       given the kernel centered at xcenters(jj) and the
          %       bandwidth
        w = weight(obj,x,xcenters,bandwidth,normalize)
          % get the kernel weight - must be non-negative (but doesn't have
          % to integrate to 1)
          % inputs:
          %  - x        : nX1 array
          %  - xcenters : mX1 array
          %  - bandwidth: double
          %  - normalize: boolean
          % returns:
          %  - w: nXm array
          %       w(ii,jj) is the weight of x(ii)
          %       given the kernel centered at xcenters(jj) and the
          %       bandwidth
          %       if normalize is true, then normalize weights so that
          %       weights are a pdf (i.e. integrates to 1)
        hlist = get_bandwidth_discretization(obj)
          % get the finite set of allowable bandwidths
          % inputs:
          % returns:
          %  - hlist: 1XN array, hlist(ii)<hlist(ii+1),
          %           bandwidth parameter domain
        KDEK = deep_copy(obj)
            % return a deep copy of the instance
            % only saved_kernel is not copied;  rather, the saved_kernel
            % instance is a property of the KDE_KERNEL copy
    end
    
end