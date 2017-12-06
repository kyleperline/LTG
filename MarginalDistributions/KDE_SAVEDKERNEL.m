classdef KDE_SAVEDKERNEL
    % if kernels are computationally expensive to evaluate they can be
    % discretized and precomputed
    % this class loads in those precomputed kernels and provides an
    % interface
    % it's assumed the saved, precomputed discretized kernels K are in the
    % following format:
    % K is (M X N X H)
    % K
    properties
        % beta kernel
        beta_kernel    % beta kernel
        beta_hdis      % beta h discretization
        beta_xcdelta   % beta x center discretization level
        beta_is_loaded % boolean
    end
    methods
        function obj = KDE_SAVEDKERNEL()
            obj.beta_is_loaded = false;
        end
        function SK = deep_copy(obj)
            SK = KDE_SAVEDKERNEL();
            if obj.beta_is_loaded
                SK = SK.load_beta();
            end
        end
        function obj = load_beta(obj)
            if ~obj.beta_is_loaded
                load('KDE_BETA_kernel.mat');
                obj.beta_kernel  = beta_kernel_discretized;
                obj.beta_hdis    = bandwidth_discretization;
                obj.beta_xcdelta = xcenter_delta;
                obj.beta_is_loaded = true;
            end
        end
        function p = get_kernel(obj,kernel_name,xc,h,normalize)
            % get the discretized kernel
            % inputs:
            %  - kernel_name: string
            %  - xc         : double, kernel center
            %  - h          : positive kernel bandwidth
            %  - normalize  : boolean, whether to return a kernel that
            %                 integrates to 1
            % returns:
            %  - p: 1XN array of discretized kernel with center xc and
            %       bandwidth h. if normalize=true then p integrates to 1
            if strcmp(kernel_name,'beta')
                if ~obj.beta_is_loaded
                    error('Beta kernel was not loaded.')
                end
                % get xcenter index
                xcind = round(xc/obj.beta_xcdelta)+1;
                % get the index hind of the element of hdis that's closest
                % to h
                [~,hind] = min(abs(h-obj.beta_hdis));
                p = obj.beta_kernel(xcind,1:end-1,hind);
                if normalize
                    p = p/obj.beta_kernel(xcind,end,hind);
                end
            else
                error(['unrecognized kernel ',kernel_name])
            end
        end
    end
    
    
end