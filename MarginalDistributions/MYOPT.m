classdef MYOPT
    % a class for some functions that solve discrete, computationally
    % expensive, noisy optimization problems
    % methods are of the following form:
    % function xstar = myopmethod(f,domain,args)
    % inputs:
    %  - f     : function handle, double = f(x), x is 1XM
    %  - domain: 1XM cell, each element is 1X(Mi) array of possible values
    %            for dimension i
    % returns:
    %  - xstar: 1XM array, minimizer (or best guess) of f
    
    methods (Static)
        function xstar = random(f,domain,maxiter)
            % random selection
            tic
            Mi = MYOPT.check_domain(domain);
            fx_list = zeros(maxiter,1);
            x_list  = zeros(maxiter,size(Mi,2));
            for ii=1:maxiter
                x = ceil(rand(1,length(Mi)).*Mi);
                x = MYOPT.convert_index2domain(domain,x);
                fx = f(x);
                fx_list(ii) = fx;
                x_list(ii,:) = x;
            end
            [~,ind] = min(fx_list);
            xstar = x_list(ind,:);
            toc
        end
        function [xstar,fstar] = SRBF(f,domain,maxiter)
            % Stochastic Radial Basis Function
            % do initialization
            Xcenter = zeros(maxiter,length(domain));
            ycenter = zeros(maxiter,1);
            Xcenter_inds = zeros(size(Xcenter));
            
            Xtmp = MYOPT.latin_hypercube(domain,length(domain)*2+1);
            N = size(Xtmp,1);
            Xcenter(1:N,:) = Xtmp;
            Xcenter_inds(1:N,:) = MYOPT.convert_domain2index(domain,Xtmp);
            for ii=1:N
                ycenter(ii) = f(Xcenter(ii,:));
            end
            dmin = zeros(1,size(Xcenter,2));
            dmax = MYOPT.check_domain(domain);
            
            % now do loop
            cand_weights = linspace(0,1,10); % candiate score weights
            ct = 0;
            for jj=N+1:maxiter
                ct = ct+1;
                % get the coefficients ---
                % use domain (which can lead to bad condition numbers):
%                 Xc = Xcenter(1:jj-1,:); 
                % use indexes as the domain
                Xc = Xcenter_inds(1:jj-1,:); % the indexes
                % scale to [0,1]
                Xc = (Xc-ones(jj-1,1)*dmin); 
                Xc = Xc./(ones(jj-1,1)*(dmax-dmin));
                yc = ycenter(1:jj-1);
                % do y cutoff
                yc = min(yc,median(yc));
                cc = MYOPT.RBF_coefs(Xc,yc);
                % figure out where to evaluate next ---
                % number of candidate points chosen uniformly randomly from 
                % whole domain
                M1 = 10; 
                % number of candidate points chosen uniformly randomly in
                % neighborhood of Xbest
                M2 = 10;
                % pick M points from domain that haven't been selected yet
                [~,ind] = min(ycenter(1:(jj-1)));
                cand_inds = MYOPT.RBF_gen_cand_points(domain,...
                    Xcenter_inds(1:(jj-1),:),M1,Xcenter_inds(ind,:),M2);
                cand = MYOPT.convert_index2domain(domain,cand_inds);
                cand = (cand-ones(size(cand,1),1)*dmin)./(ones(size(cand,1),1)*(dmax-dmin));
                cand_y = MYOPT.RBF_eval(cand,Xc,cc); % RBF estimate
                mincy = min(cand_y);
                maxcy = max(cand_y);
                if maxcy>mincy
                    cand_yscore = (cand_y-mincy)/(maxcy-mincy);
                else
                    cand_yscore = ones(size(cand_y,1),1);
                end
                cand_d = min(pdist2(cand,Xc),[],2); % distance to nearest neighbor
                mincd = min(cand_d);
                maxcd = max(cand_d);
                if maxcd>mincd
                    cand_dscore = 1 - (cand_d-mincd)/(maxcd-mincd);
                else
                    cand_dscore = 1 - ones(size(cand_d,1),1);
                end
                cw_ind = mod(ct,length(cand_weights))+1;
                weight = cand_weights(cw_ind);
                cand_score = cand_yscore*weight+cand_dscore*(1-weight);
                [~,bestind] = min(cand_score);
                best01 = cand(bestind,:); % best point in [0,1]
                % evaluate ---
                Xcenter(jj,:) = best01.*(dmax-dmin)+dmin;
                ycenter(jj) = f(Xcenter(jj,:));
                Xcenter_inds(jj,:) = MYOPT.convert_domain2index(domain,Xcenter(jj,:));

            end
            % now get the best point
            [fstar,ind] = min(ycenter);
            xstar = Xcenter(ind,:);
        end
        function [xstar,fstar] = gradient_descent(f,domain,x0,maxiter)
            % gradient descent
            % x0 is domain (not index)
            f0 = f(x0);
            Mi = MYOPT.check_domain(domain);
            M = length(Mi);
            converged = false;
            niters = 0;
            x0 = MYOPT.convert_domain2index(domain,x0);
            while ~converged && niters<maxiter
                f1 = f0;
                for ii=1:M
                    x2 = x0;
                    if x2(ii)>1
                        x2(ii) = x2(ii)-1;
                        x2d = MYOPT.convert_index2domain(domain,x2);
                        f2 = f(x2d);
                        niters = niters + 1;
                        if f2<f1
                            f1 = f2;
                            x1 = x2;
                        end
                    end
                    x2 = x0;
                    if x2(ii)<Mi(ii)
                        x2(ii) = x2(ii)+1;
                        x2d = MYOPT.convert_index2domain(domain,x2);
                        f2 = f(x2d);
                        niters = niters + 1;
                        if f2<f1
                            f1 = f2;
                            x1 = x2;
                        end
                    end
                end
                if f1==f0
                    converged = true;
                else
                    f0 = f1;
                    x0 = x1;
                end
            end
            xstar = MYOPT.convert_index2domain(domain,x0);
            fstar = f0;
        end
        function LHpts = latin_hypercube(domain,n)
            % select n points from domain in a latin hypercube
            % LHpts is nXlength(domain)
            Mi = MYOPT.check_domain(domain);
            n = min(n,max(Mi));
            LHpts = zeros(n,length(Mi));
            myrank = 0;
            while myrank<length(Mi)
                for ii=1:length(Mi)
                    inds = round(linspace(1,Mi(ii),n)');
                    LHpts(:,ii) = inds(randperm(n));
                end
                myrank = rank(LHpts);
            end
            for jj=1:n
                LHpts(jj,:) = MYOPT.convert_index2domain(domain,LHpts(jj,:));
            end
        end
    end
    methods (Static, Access=private)
        function c = RBF_coefs(Xcenter,y)
            % get the cubic RBF coefficients
            D = squareform(pdist(Xcenter)).^3; % cubic kernel
            P = [Xcenter,ones(size(Xcenter,1),1)];
            M = [ D P ; P' zeros(size(Xcenter,2)+1) ];
            if rcond(M)<1e-16
                disp(Xcenter)
                a=asdf;
            end
            c = M\[y ; zeros(size(Xcenter,2)+1,1)];
        end
        function y = RBF_eval(X,Xcenter,c)
            % evaluate the RBF
            D = pdist2(X,Xcenter).^3; % cubic kernel
            P = [X,ones(size(X,1),1)];
            y = [ D P ]*c;
        end
        function xrand = RBF_gen_cand_points(domain,Xinds,M1,Xindbest,M2)
            % get M1+M2 index points in domain that are not in
            % [Xinds;Xindbest].
            % first get M1 points uniformly randomly from the whole domain
            xrand1 = MYOPT.RBF_gen_cand_points_single(domain,Xinds,M1);
            % now get points uniformly randomly from the neighborhood of
            % Xindbest
            Mi = MYOPT.check_domain(domain);
            neighb = max(1,round(Mi/10));
            dom2 = cell(1,length(domain));
            for ii=1:length(domain)
                d = domain{ii};
                [~,ind] = min(abs(d-Xindbest(ii)));
                dom2{ii} = d(max(1,ind-neighb(ii)):min(Mi(ii),ind+neighb(ii)));
            end
            xrand2 = MYOPT.RBF_gen_cand_points_single(dom2,Xinds,M2);
            xrand = [xrand1;xrand2];
        end
        function xrand = RBF_gen_cand_points_single(domain,Xinds,M)
            % get M points drawn uniformly randomly from domain that are
            % not in Xinds
            Mi = MYOPT.check_domain(domain);
            xrand = ceil(rand(M,length(domain)).*(ones(M,1)*Mi));
            issame = MYOPT.RBF_check_same_points(xrand,Xinds);
            ct = 0;
            while sum(issame)>0 && ct<5
                xrand(issame,:) = ceil(rand(sum(issame),length(domain)).*...
                                       (ones(sum(issame),1)*Mi));
                issame = MYOPT.RBF_check_same_points(xrand,Xinds);
                ct = ct+1;
            end
            xrand = xrand(~issame,:);
            
        end
        function issame = RBF_check_same_points(X1,X2)
            % issame is boolean column vector with size(X1,1) rows
            % issame(ii)=true if X1(ii,:)=X2(jj,:) for any jj
            d = pdist2(X1,X2);
            issame = min(d,[],2)<1e-12;
        end
        function Mi = check_domain(domain)
            % make sure domain is right size
            % returns:
            %  - Mi: 1Xlength(domain), Mi(ii)=length(domain{ii})
            assert(size(domain,1)==1)
            Mi = zeros(1,length(domain));
            for ii=1:length(domain)
                assert(size(domain{ii},1)==1);
                Mi(ii) = size(domain{ii},2);
            end
        end
        function xd = convert_index2domain(domain,x)
            % x is 1Xn
            xd = zeros(size(x));
            for ii=1:length(domain)
                d = domain{ii};
                xd(:,ii) = d(x(:,ii));
            end
        end
        function id = convert_domain2index(domain,x)
            % x is 1Xn
            id = zeros(size(x));
            for ii=1:length(domain)
                d = domain{ii};
                [~,id(:,ii)] = min(pdist2(x(:,ii),d'),[],2);
            end
        end
    end
    methods (Static)
        function test_gradient_descent()
            domain = { 1:10, 1:10 };
            f = @(x) (x(1)-3)^2 + (x(2)-0)^2;
            for ii=1:10
                for jj=1:10
                    x0 = [ii jj];
                    xstar = MYOPT.gradient_descent(f,domain,x0,100);
                    assert(all(xstar == [3 1]))
                end
            end
            disp('passed test')
        end
    end
end