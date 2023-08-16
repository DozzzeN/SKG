function X = sils(B,y,p)
%
% X = sils(B,y,p) produces p optimal solutions to the standard integer
% least squares problem min_{x}||y-Bx||
%
% Inputs:
%    B - m-by-n real matrix with full column rank
%    y - m-dimensional real vector
%    p - number of optimal solutions and its default value is 1
%
% Output:
%    X - n-by-p integer matrix (in double precision), whose j-th column
%        is the j-th optimal solution, i.e., its residual is the j-th
%        smallest, ||y-B*X(:,1)|| <= ...<= ||y-B*X(:,p)||
%

% Subfunctions: sils_reduction, sils_search

% Main References:
% [1] X. Xie, X.-W. Chang, and M. Al Borno. Partial LLL reduction,
%     Proceedings of IEEE GLOBECOM 2011, 5 pages.
% [2] X.-W. Chang, X. Yang, and T. Zhou, MLAMBDA: A Modified LAMBDA Method
%     for Integer Least-squares Estimation, Journal of Geodesy, 79 (2005),
%     pp. 552-565.
% [3] A. Ghasemmehdi and E. Agrell, Faster Recursions in Sphere Decoding,
%     IEEE Transactions on Information Theory, 57 (2011), pp. 3530-3536.

% Authors: Xiao-Wen Chang, www.cs.mcgill.ca/~chang
%          Tianyang Zhou
% Copyright (c) 2006-2022. Scientific Computing Lab, McGill University.
% October 2006. Last revision: Nov 2022


    % Check input arguments
    if nargin < 2 % input error
        error('Not enough input arguments!')
    end

    if nargin < 3
        p = 1;
    end

    if p <= 0 % input error
        error('Third input argument must be an integer bigger than 0!')
    end

    [m,n] = size(B);

    if rank(B) < n
	    error('Matrix does not have full column rank!')
    end

    if m ~= size(y,1) || size(y,2) ~= 1  % Input error
        error('Input arguments have a matrix dimension error!')
    end


    % Reduction - reduce the problem to the triangular form
    [R,Z,y] = sils_reduction(B,y);

    % Search - find the p optimal solustions to the reduced problem
    Zhat = sils_search(R,y(1:n),p);

    % Perform the unimodual transformation to obtain the solutions to
    %   the original problem
    X = Z * Zhat;
end


%========================================================================
%====================== Subfunction of sils =============================
%========================================================================

function [R,Z,y] = sils_reduction(B,y)
%
% [R,Z,y] = sils_reduction(B,y) reduces the general standard integer
% least squares problem to an upper triangular one by the LLL-QRZ
% factorization Q'*B*Z = [R; 0]. The orthogonal matrix Q
% is not produced.
%
% Inputs:
%    B - m-by-n real matrix with full column rank
%    y - m-dimensional real vector to be transformed to Q'*y
%
% Outputs:
%    R - n-by-n LLL-reduced upper triangular matrix
%    Z - n-by-n unimodular matrix, i.e., an integer matrix with |det(Z)|=1
%    y - m-vector transformed from the input y by Q', i.e., y := Q'*y
%

% Subfunction: qrmcp

% Main Reference:
% X. Xie, X.-W. Chang, and M. Al Borno. Partial LLL Reduction,
% Proceedings of IEEE GLOBECOM 2011, 5 pages.

% Authors: Xiao-Wen Chang, www.cs.mcgill.ca/~chang
%          Xiaohu Xie, Tianyang Zhou
% Copyright (c) 2006-2016. Scientific Computing Lab, McGill University.
% October 2006. Last revision: June 2016


    [~,n] = size(B);

    % QR factorization with minimum-column pivoting
    [R,piv,y] = qrmcp(B,y);

    % Obtain the permutation matrix Z
    Z = zeros(n,n);
    for j = 1 : n
        Z(piv(j),j) = 1;
    end

    % ------------------------------------------------------------------
    % --------  Perfome the partial LLL reduction  ---------------------
    % ------------------------------------------------------------------

    k = 2;

    while k <= n

        k1 = k-1;
        zeta = round(R(k1,k) / R(k1,k1));
        alpha = R(k1,k) - zeta * R(k1,k1);

        if R(k1,k1)^2 > (1 + 1.e-10) * (alpha^2 + R(k,k)^2)
            if zeta ~= 0
                % Perform a size reduction on R(k-1,k)
                R(k1,k) = alpha;
                R(1:k-2,k) = R(1:k-2,k) - zeta * R(1:k-2,k-1);
                Z(:,k) = Z(:,k) - zeta * Z(:,k-1);

                % Perform size reductions on R(1:k-2,k)
                for i = k-2:-1:1
                    zeta = round(R(i,k)/R(i,i));
                    if zeta ~= 0
                        R(1:i,k) = R(1:i,k) - zeta * R(1:i,i);
                        Z(:,k) = Z(:,k) - zeta * Z(:,i);
                    end
                end
            end

            % Permute columns k-1 and k of R and Z
            R(1:k,[k1,k]) = R(1:k,[k,k1]);
            Z(:,[k1,k]) = Z(:,[k,k1]);

            % Bring R back to an upper triangular matrix by a Givens rotation
            [G,R([k1,k],k1)] = planerot(R([k1,k],k1));
            R([k1,k],k:n) = G * R([k1,k],k:n);

            % Apply the Givens rotation to y
            y([k1,k]) = G * y([k1,k]);

            if k > 2
                k = k - 1;
            end

        else
            k = k + 1;
        end
    end
end


%========================================================================
%====================== Subfunction of sils_reduction ===================
%========================================================================

function [R,piv,y] = qrmcp(B,y)
%
% [R,piv,y] = qrmcp(B,y) computes the QR factorization of B with
%             minimum-column pivoting:
%                  Q'BP = R (underdetermined B),
%                  Q'BP = [R; 0] (underdetermined B)
%             and computes Q'*y. The orthogonal matrix Q is not produced.
%
% Inputs:
%    B - m-by-n real matrix to be factorized
%    y - m-dimensional real vector to be transformed to Q'y
%
% Outputs:
%    R - m-by-n real upper trapezoidal matrix (m < n)
%        n-by-n real upper triangular matrix (m >= n)
%    piv - n-dimensional permutation vector representing P
%    y - m-vector transformed from the input y by Q, i.e., y := Q'*y

% Authors: Xiao-Wen Chang, www.cs.mcgill.ca/~chang
%          Xiaohu Xie, Tianyang Zhou
% Copyright (c) 2006-2016. Scientific Computing Lab, McGill University.
% October 2006; Last revision: June 2016


    [m,n] = size(B);

    % Initialization
    colNormB = zeros(2,n);
    piv = 1:n;

    % Compute the 2-norm squared of each column of B
    for j = 1:n
        colNormB(1,j) = (norm(B(:,j)))^2;
    end

    n_dim = min(m-1,n);

    for k = 1 : n_dim
        % Find the column with minimum 2-norm in B(k:m,k:n)
        [~, i] = min(colNormB(1,k:n) - colNormB(2,k:n));
        q = i + k - 1;

        % Column interchange
        if q > k
            piv([k,q]) = piv([q,k]);
            colNormB(:,[k,q]) = colNormB(:,[q,k]);
            B(:,[k,q]) = B(:,[q,k]);
        end

        % Compute and apply the Householder transformation  I-tau*v*v'
        if norm(B(k+1:m,k)) > 0 % A Householder transformation is needed
	        v = B(k:m,k);
            rho = norm(v);
            if v(1) >= 0
                rho = -rho;
            end
            v(1) = v(1) - rho; % B(k,k)+sgn(B(k,k))*norm(B(k:n,k))
            tao = -1 / (rho * v(1));
            B(k,k) = rho;
            if m < n
               B(k+1:m,k) = 0;
            end
            B(k:m,k+1:n) = B(k:m,k+1:n) - tao * v * (v' * B(k:m,k+1:n));
            % Update y by the Householder transformation
            y(k:m) = y(k:m,:) - tao * v * (v' * y(k:m));
        end

        % Update colnormB(2,k+1:n)
        colNormB(2,k+1:n) = colNormB(2,k+1:n) + B(k,k+1:n) .* B(k,k+1:n);
    end

    if m < n
       R = B;
    else
       R = triu(B(1:n,1:n));
    end
end

%========================================================================
%====================== Subfunction of sils =============================
%========================================================================

function Zhat = sils_search(R,y,p)
%
% Zhat = sils_search(R,y,p) produces p optimal solutions to
% the upper triangular integer least squares problem min_{z}||y-Rz||
% by a depth-first search algorithm.
%
% Inputs:
%    R - n-by-n real nonsingular upper triangular matrix
%    y - n-dimensional real vector
%    p - the number of optimal solutions with a default value of 1
%
% Output:
%    Zhat - n-by-p integer matrix (in double precision), whose j-th column
%           is the j-th optimal solution, i.e., its residual is the j-th
%           smallest, so ||y-R*Zhat(:,1)|| <= ...<= ||y-R*Zhat(:,p)||

% Main References:
% [1] X.-W. Chang, X. Yang, and T. Zhou, MLAMBDA: A Modified LAMBDA Method
%     for Integer Least-squares Estimation, Journal of Geodesy, 79 (2005),
%     pp. 552-565.
% [2] A. Ghasemmehdi and E. Agrell, Faster Recursions in Sphere Decoding,
%     IEEE Transactions on Information Theory, 57 (2011), pp. 3530-3536.
%

% Authors: Xiao-Wen Chang, www.cs.mcgill.ca/~chang
%          Tianyang Zhou, Xiangyu Ren
% Copyright (c) 2006-2016. Scientific Computing Lab, McGill University.
% October 2006. Last revision: June 2016.


    % ------------------------------------------------------------------
    % --------  Initialization  ----------------------------------------
    % ------------------------------------------------------------------

    n = size(R,1);

    % Current point
    z = zeros(n,1);

    % c(k)=(y(k)-R(k,k+1:n)*z(k+1:n))/R(k,k)
    c = zeros(n,1);

    % d(k): left or right search direction at level k
    d = zeros(n,1);

    % Partial squared residual norm for z
    % prsd(k) = (norm(y(k+1:n) - R(k+1:n,k+1:n)*z(k+1:n)))^2
    prsd = zeros(n,1);

    % Store some quantities for efficiently calculating c
    % S(k,n) = y(k),
    % S(k,j-1) = y(k) - R(k,j:n)*z(j:n) = S(k,j) - R(k,j)*z(j), j=k+1:n
    S = zeros(n,n);
    S(:,n) = y;

    % path(k): record information for updating S(k,k:path(k)-1)
    path = n*ones(n,1);

    % The level at which search starts to move up to a higher level
    ulevel = 0;

    % The p candidate solutions (or points)
    Zhat = zeros(n,p);

    % Squared residual norms of the p candidate solutions
    rsd = zeros(p,1);

    % Initial squared search radius
    beta = inf;

    % The initial number of candidate solutions
    ncand = 0;

    % ------------------------------------------------------------------
    % --------  Search process  ----------------------------------------
    % ------------------------------------------------------------------

    c(n) = y(n) / R(n,n);
    z(n) = round(c(n));
    gamma = R(n,n) * (c(n) - z(n));
    % Determine enumeration direction at level n
    if c(n) > z(n)
        d(n) = 1;
    else
        d(n) = -1;
    end

    k = n;

    while 1
        % Temporary partial squared residual norm at level k
        newprsd = prsd(k) + gamma * gamma;

        if newprsd < beta
            if k ~= 1 % move to level k-1
                % Update path
                if ulevel ~= 0
                    path(ulevel:k-1) = k;
                    for j = ulevel-1 : -1 : 1
                         if path(j) < k
                               path(j) = k;
                         else
                             break;  % Note path(1:j-1) >= path(j)
                         end
                    end
                end

                % Update S
                k = k - 1;
                for j = path(k) : -1 : k+1
                    S(k,j-1) = S(k,j) - R(k,j) * z(j);
                end

                % Update the partial squared residual norm
                prsd(k) = newprsd;

                % Find the initial integer
                c(k) = S(k,k) / R(k,k);
                z(k) = round(c(k));
                gamma = R(k,k) * (c(k) - z(k));
                if c(k) > z(k)
                    d(k) = 1;
                else
                    d(k) = -1;
                end

                ulevel = 0;

            else % A new point is found, update the set of candidate solutions
                if ncand < p % Add the new point
                    ncand = ncand + 1;
                    Zhat(:,ncand) = z;
                    rsd(ncand) = newprsd;
                    if ncand == p
                        beta = rsd(p);
                    end
                else % Insert the new point and remove the worst one
                    i = 1;
                    while i < p && rsd(i) <= newprsd
                        i = i + 1;
                    end
                    Zhat(:,i:p) = [z, Zhat(:,i:p-1)];
                    rsd(i:p) = [newprsd; rsd(i:p-1)];
                    beta = rsd(p);
                end

                z(1) = z(1) + d(1);
                gamma = R(1,1)*(c(1)-z(1));
                if d(1) > 0
                    d(1) = -d(1) - 1;
                else
                    d(1) = -d(1) + 1;
                end
            end
        else
            if k == n % The p optimal solutions have been found
                break
            else  % Move back to level k+1
                if ulevel == 0
                   ulevel = k;
                end
                k = k + 1;
                % Find a new integer at level k
                z(k) = z(k) + d(k);
                gamma = R(k,k) * (c(k) - z(k));
                if d(k) > 0
                    d(k) = -d(k) - 1;
                else
                    d(k) = -d(k) + 1;
                end
            end
        end
    end
end

