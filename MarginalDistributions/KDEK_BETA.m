classdef KDEK_BETA < KDE_KERNEL
    % Kernel Density Estimation - Beta Kernel
    % "Time Adaptive Conditional Kernel Density Estimation for Wind Power
    % Forecasting", 2012
    % by R.J. Bessa, V. Miranda, A. Botterud, J. Wang and Emil M. Constantinescu 
    %
    % the kernel is
    % 
    % f_X(x) = K_{1+x/h,1+(1-x)/h}(X)
    % 
    % where 0<=X<=1 is a data point and f(x), 0<=x<=1, is the kernel
    % K_{p,q} is the probability density function of Beta(p,q)
    % this means that f_X(x)>=0 but does not integrate to 1 and there is 
    % no closed form solution for its cdf
    % so, f is discretized and is represented as a piecewise linear kernel
    %
    properties (Access=private)
        N % number of discretizations 
        Delta % = 1/(Nxc-1)
        xdis
        Nh % number of discretizations of bandwidth
        hdis % discretized h
    end
    methods
        function KDEK = deep_copy(obj)
            % return a deep copy
            % do not input SavedKernel
            KDEK = KDEK_BETA(obj.lower_bound,obj.upper_bound);
        end
        function obj = KDEK_BETA(ymin,ymax)
            obj.lower_bound = ymin;
            obj.upper_bound = ymax;
            obj.is_circular = false;
            % discretize x
            obj.N = 400;
            % Delta has +eps since otherwise rounding errors sometimes
            % gives ceil(1/Delta)=obj.N+1
            % adding eps ensures ceil(1/Delta)=obj.N
            obj.Delta = 1/(obj.N-1)+eps; 
            obj.xdis = linspace(0,1,obj.N);
            % h (bandwidth) discretization is only used for SavedKernel
            obj.Nh = 50;
            obj.hdis = exp(linspace(log(1e-6),log(1),obj.Nh));
        end
        function hlist = get_bandwidth_discretization(obj)
            hlist = obj.hdis;
        end
        function p = cdf(obj,x,xcenters,bandwidth)
            assert(numel(xcenters)==numel(bandwidth))
            % don't scale x because it's scaled in cdf_integrate_singleXi
            % don't scale xcenters because it's scaled in
            % get_discretized_weight
%             x = obj.scale(x);
            p = zeros(size(x,1),size(xcenters,1));
            for ii=1:size(xcenters,1)
                pdf = obj.get_discretized_weight(xcenters(ii),bandwidth(ii),true);
                p(:,ii) = obj.cdf_integrate_singleXi(x,pdf');
            end
        end
        function p = weight(obj,x,xcenters,bandwidth,normalize)
            assert(numel(xcenters)==numel(bandwidth))
            if ~all(x>=obj.lower_bound)
                figure
                plot(x-obj.lower_bound)
                if any(isnan(x))
                    disp('there are NaNs.  get rid of them')
                    tmp = find(isnan(x));
                    Ntmp = numel(tmp);
                    disp(tmp(min(Ntmp,10)))
                end
                assert(all(x>=obj.lower_bound))
            end
            if ~all(x<=obj.upper_bound)
                figure
                hold on
                plot(obj.upper_bound,'b')
                plot(x,'r')
                figure
                plot(obj.upper_bound-x)
                assert(all(x<=obj.upper_bound))
            end
            x = obj.scale(x);
%             xcenters = obj.scale(xcenters);
            p = zeros(size(x,1),size(xcenters,1));
            for ii=1:size(xcenters,1)
                pdf = obj.get_discretized_weight(xcenters(ii),bandwidth(ii),normalize);
                p(:,ii) = obj.myfastinterp1(obj.Delta,pdf,x')';
            end
        end
        function p = beta_kernel(obj,x,h)
            % inputs:
            %  - x: double, 0<=x<=1, center
            %  - h: double, 0<h, scaling
            % returns:
            %  - p: 1Xlength(obj.xdis), p>0 kernel
            assert(numel(x)==1)
            assert(numel(h)==1)
            p = pdf('beta',x,1+obj.xdis/h,1+(1-obj.xdis)/h);
            % p might be zero at points far away from x
            % this can cause troubles
            small1 = 1e-9;
            small2 = 1e-14;
            psmall = small1*exp(log(small2/small1)*abs(obj.xdis-x));
            p = max(p,psmall);
        end
        function generate_and_save_kernel(obj)
            % this function generates the discretized beta kernel
            % only call it once (like, ever)
            % if you ever change this code and run it you'll have to double
            % check KDE_SAVEDKERNEL
            beta_kernel_discretized = zeros(obj.N,obj.N+1,obj.Nh);
            for ii=1:obj.N % loop over xcenters
                for jj=1:obj.Nh % loop over bandwidths
                    h = obj.hdis(jj);
                    kernel = obj.beta_kernel(obj.xdis(ii),h);
%                     kernel = pdf('beta',obj.xdis(ii),1+obj.xdis/h,1+(1-obj.xdis)/h);
                    beta_kernel_discretized(ii,1:end-1,jj) = kernel;
                    % and get the normalization constant
                    integral = obj.cdf_integrate_singleXi(1,kernel');
                    beta_kernel_discretized(ii,end,jj) = integral;
                end
            end
            xcenter_delta = obj.Delta;
            bandwidth_discretization = obj.hdis;
            save('KDE_BETA_kernel.mat','beta_kernel_discretized','xcenter_delta','bandwidth_discretization')
        end
        function x = scale(obj,x)
            x = (x-obj.lower_bound)/(obj.upper_bound-obj.lower_bound);
        end
        function x = scale_inv(obj,x)
            x = obj.lower_bound+x*(obj.upper_bound-obj.lower_bound);
        end
        function N = get_n_kernel_dis(obj)
            N = obj.N;
        end
        function p = get_discretized_weight(obj,Xi,h,normalize)
            % lower_bound<=Xi<=upper_bound is a scalar. it's a center
            % h is the bandwidth scalar
            % p is (1 X obj.N)
            % first scale x according to bounds
            Xi = obj.scale(Xi);
            assert(numel(Xi)==1)
            assert(numel(h)==1)
            assert(0<=Xi)
            assert(Xi<=1)
            if ~obj.saved_kernel
                % when the bandwidth parameter goes to zero p goes towards a
                % delta function
                % to avoid the situation where p is 0 at all of the discretized
                % points obj.xdis (i.e. weight function is 0 everywhere), we'll
                % round Xi to the nearest point in obj.xdis
                Xi = obj.Delta*round(Xi/obj.Delta);
                p = obj.beta_kernel(Xi,h);
                if normalize
                    integral = obj.cdf_integrate_singleXi(obj.upper_bound,p');
                    p = p/integral;
                end
            else
                % also discretize h
                [~,hind] = min(abs(h-obj.hdis));
                h = obj.hdis(hind);
                Xi = obj.Delta*round(Xi/obj.Delta);
                p = obj.SavedKernel.get_kernel('beta',Xi,h,normalize);
            end
        end
        function p = cdf_integrate_singleXi(obj,x,pdf_Xi)
            % x is nX1
            % pdf_Xi is (obj.N X 1). it's the kernel pdf with center Xi
            % p is nX1. p(ii) is the cdf of x(ii)
            % this uses the trapezoid rule for integrating the pdf by
            % assuming the pdf is piecewise linear
            x = obj.scale(x);
            assert(size(x,2)==1)
            assert(size(pdf_Xi,1)==obj.N)
            assert(all(x>=0))
            assert(all(x<=1))
            integral = cumtrapz(pdf_Xi)*obj.Delta; % integral at linspace(0,1,obj.N)
            ind = ceil(x/obj.Delta); % in 0:(N-1)
            p = zeros(size(x,1),1);
            notzero = ind>0; % if ind==0 then cdf=0
            indnz = ind(notzero);
            p(notzero) = integral(indnz); % integral of pdf from 0 to floor(x/Delta)*Delta  
            xdiff = x(notzero) - (indnz-1)*obj.Delta;
            p0 = pdf_Xi(indnz);
            p1 = pdf_Xi(indnz+1);
            p(notzero) = p(notzero) + p0.*xdiff+(1/2/obj.Delta)*(p1-p0).*(xdiff.^2);
        end
        function y = inverse_cdf_integrate(obj,p,mypdfs,scale)
            % p is 1XM, 0<=p<=1
            % mypdfs is (obj.N X M). 
            % each mypdfs(:,ii) is a linear combination of kernel pdfs
            %     with different centers
            % y is 1XM s.t. integral_{x=0}^{y(ii)} mypdfs(:,ii)(x)dx= p(ii)
            assert(size(mypdfs,1)==obj.N);
            M = size(mypdfs,2);
            assert(size(p,1)==1)
            assert(size(p,2)==M)
            assert(all(all(mypdfs>=0)))
            assert(all(0<=p & p<=1))
            integral = cumtrapz(mypdfs,1)*obj.Delta; % integral at linspace(0,1,obj.N)
            assert(~any(any(isnan(integral))))
            assert(all(abs(integral(end,:)-1)<1e-8),'your linear combo pdf does not integrate to 1')
            % ind(ii) is largest index s.t. p(ii)>=integral(ind(ii))
            rowind = sum(ones(obj.N,1)*p>=integral,1); % 1XM
            notN = rowind<obj.N; % if ind(ii)=obj.N then y(ii)=1
            rowindnotN = rowind(notN);
            colind = 1:M;
            colindnotN = colind(notN);
            matind = (colindnotN-1)*obj.N+rowindnotN;
            p1 = mypdfs(matind);
            p2 = mypdfs(matind+1);
            c1 = integral(matind);
%             
            diff = p(notN) - c1;
            if ~all((p1.^2)+2*(p2-p1).*diff/obj.Delta>-1e-8)
                min((p1.^2)+2*(p2-p1).*diff/obj.Delta)
            end
            assert(all((p1.^2)+2*(p2-p1).*diff/obj.Delta>-1e-8))
            dtmp = real(((p1.^2)+2*(p2-p1).*diff/obj.Delta).^0.5);
            d1 = (-p1-dtmp)*obj.Delta./(p2-p1); % solution 1
            d2 = (-p1+dtmp)*obj.Delta./(p2-p1); % solution 2
            % we want solution d satisfying 0<=d<=obj.Delta
            myeps = 1e-4;
            sat1 = -myeps<=d1 & d1<=obj.Delta+myeps;
            sat2 = -myeps<=d2 & d2<=obj.Delta+myeps;
            if ~all(sat1 | sat2)
                figure
                hold on
                plot(d1/obj.Delta,'r*-')
                plot(d2/obj.Delta,'b*-')
            end
            assert(all(sat1 | sat2))
            d = zeros(size(d1));
            d(sat1) = d1(sat1);
            d(sat2) = d2(sat2);
            y = ones(1,M);
            y(notN) = (rowindnotN-1)*obj.Delta+d;
            assert(all(y>=-myeps))
            assert(all(y<=1+myeps))
            y = min(max(0,y),1);
            if scale
                y = obj.scale_inv(y);
            end
        end
        function y = inverse_cdf_integrate_single_pdf(obj,p,mypdf,scale)
            % p is MX1, 0<=p(ii)<=1
            % mypdf is (obj.N X 1). it's a linear combination of kernel
            % pdfs with different centers
            % y is MX1 s.t. integral_{x=0}^{y(ii)} mypdf(x) dx = p(ii)
            M = size(p,1);
            assert(size(mypdf,2)==1)
            assert(size(mypdf,1)==obj.N);
            assert(size(p,2)==1);
            assert(all(mypdf>=0))
            assert(all(p>=0))
            assert(all(p<=1))
            integral = cumtrapz(mypdf)*obj.Delta; % integral at linspace(0,1,obj.N)
            assert(abs(integral(end)-1)<1e-12,'your linear combo pdf does not integrate to 1')
            % ind(ii) is largest index s.t. p(ii)>=integral(ind(ii))
            ind = sum(p*ones(1,obj.N)>=ones(M,1)*(integral'),2);
            y = ones(M,1);
            notN = ind<obj.N; % if ind(ii)=obj.N then y(ii)=1
            indnotN = ind(notN);
            diff = p(notN,1) - integral(indnotN);
            p1 = mypdf(indnotN);
            p2 = mypdf(indnotN+1);
            % int_{x=0}^{d(ii)} p1(ii)+x*(p2(ii)-p1(ii))/obj.Delta=diff(ii)
            if ~all((p1.^2)-2*(p2-p1).*diff/obj.Delta>-1e-8)
                min((p1.^2)-2*(p2-p1).*diff/obj.Delta)
            end
            assert(all((p1.^2)-2*(p2-p1).*diff/obj.Delta>-1e-8))
            dtmp = real(((p1.^2)-2*(p2-p1).*diff/obj.Delta).^0.5);
            d1 = (-p1-dtmp)*obj.Delta./(p2-p1); % solution 1
            d2 = (-p1+dtmp)*obj.Delta./(p2-p1); % solution 2
            % we want solution d satisfying 0<=d<=obj.Delta
            sat1 = 0<=d1 & d1<=obj.Delta;
            sat2 = 0<=d2 & d2<=obj.Delta;
            if ~all(sat1 | sat2)
                figure
                hold on
                plot(d1/obj.Delta,'r*-')
                plot(d2/obj.Delta,'b*-')
            end
            assert(all(sat1 | sat2))
            d = zeros(size(d1));
            d(sat1) = d1(sat1);
            d(sat2) = d2(sat2);
            y(notN) = (indnotN-1)*obj.Delta+d;
            assert(all(y>=0))
            assert(all(y<=1))
            if scale
                y = obj.scale_inv(y);
            end
        end
    end
    methods (Static)
        
        function test()
            KB = KDEK_BETA(0,100);
%             KB.test_integral_bdd();
%             KB.test_plot_kernel();
            KB.test_plot_kernel_cdf();
        end
    end
end