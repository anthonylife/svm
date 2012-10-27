function net = train(tutor, X, Y, C, kernel, alpha_init, bias_init)
% Train a support vector classification network, using the sequential minimal
% optimisation algorithm.
%
% net = train(tutor, x, y, net);
% net = train(tutor, x, y, C, kernel);
% net = train(tutor, X, Y, C, kernel, alpha_init, bias_init)
%
% where:
% tutor = tutor object
% x = training inputs
% y = training data
% C = Upper bound - non-separable case (optional, defaults C=Inf)
% kernel = kernel function (optional, defaults kernel=linear)
% net = svc object (optional)
% alpha_init =
% bias_init =
%
% if kernel, alpha_init or bias_init is 'NOBIAS' then no
% threshold, b, is used and it is set to 0

% File : @quadprogsvctutor/train.m
% Author : Diego Andres Alvarez Marin
% Description : Part of an object-oriented implementation of Vapnik's
% Support Vector Machine, as described in [1].
%
% References :
% V.N. VAPNIK, "The Nature of Statistical Learning Theory",
% Springer-Verlag, New York, ISBN 0-387-94559-8, 1995.
%
% PLATT, J.~C. (1998).
% Fast training of support vector machines using sequential minimal
% optimization. In SchÃ¶lkopf, B., Burges, C., and Smola, A.~J., editors,
% Advances in Kernel Methods: Support Vector Learning, chapter~12,
% pages 185--208. MIT Press, Cambridge, Massachusetts.

% History : May 15/2001 - v1.00

if size(Y, 2) ~= 1 | ~isreal(Y)
error('y must be a real double precision column vector');
end

n = size(Y, 1);

if n ~= size(X, 1)
error('x and y must have the same number of rows');
end

if (nargin<3 | nargin>7) % check correct number of arguments
help svc
return;
end;

if nargin == 4 & isa(C, 'svc')
net = C;
C = get(net,'C');
kernel = get(net,'kernel');
else
if nargin < 4, C = Inf; end;
if nargin < 5, kernel = linear; end;
end;

NOBIAS = 0;
switch nargin
case 5
if ischar(kernel) & strcmp(kernel,'NOBIAS')
NOBIAS = 1;
end;
case 6
if ischar(alpha_init) & strcmp(alpha_init,'NOBIAS')
NOBIAS = 1;
end;
case 7
if ischar(bias_init) & strcmp(bias_init,'NOBIAS')
NOBIAS = 1;
end;
end;

if nargin == 7
if n ~= size(alpha_init, 1)
error('alpha must be a real double precision column vector with the same size as y');
end
if any(alpha_init < 0)
error ('No pueden existir alphas negativos')
end;
else
alpha_init = zeros(n,1); %inicializo los pesos a zeros
bias_init = 0; %inicializo threshold a zero
end;

fprintf('\n\nSequential Minimal Optimization: SVMs for Classification\n')
fprintf( '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')

tic;
if NOBIAS
SMO = SMOTutorNOBIAS(X, Y, C, kernel, alpha_init, bias_init);
else
SMO = SMOTutor(X, Y, C, kernel, alpha_init, bias_init);
end;
fprintf('Execution time: %4.1f seconds\n',toc);

sv = X;
w = (SMO.alpha.*Y)'; %weight vector
net = svc(kernel, sv, w, SMO.bias, C);
fprintf('Epochs : %d\n',SMO.epochs);
w0_2 = w*SMO.Kcache*w';
fprintf('|w0|^2 : %f\n',w0_2);
fprintf('Margin : %f\n',1/sqrt(w0_2));
NUMSV = nonZeroLagrangeMultipliers;
fprintf('Support Vectors : %d (%3.1f%%)\n\n',NUMSV,100*NUMSV/n);
return;

function RESULT = SMOTutor(x,y,C,kernel,alpha_init,bias_init)
%Implementation of the Sequential Minimal Optimization (SMO)
%training algorithm for Vapnik's Support Vector Machine (SVM)

global SMO;

[ntp,d] = size(x);
%Inicializando las variables
SMO.epsilon = svtol(C); SMO.tolerance = KKTtol;
SMO.x = x; SMO.y = y;
SMO.C = C; SMO.kernel = kernel;
SMO.alpha = alpha_init; SMO.bias = bias_init;
SMO.ntp = ntp; %number of training points

%CACHES:
SMO.Kcache = evaluate(kernel,x,x); %kernel evaluations
SMO.error = zeros(SMO.ntp,1); %error

if ~any(SMO.alpha)
%Como todos los alpha(i) son zeros, entonces fwd(i), tambien es zero
SMO.error = -y;
else
SMO.error = fwd(1:ntp) - y;
end;

numChanged = 0; examineAll = 1;
epoch = 0;

%When all data were examined and no changes done the loop reachs its
%end. Otherwise, loops with all data and likely support vector are
%alternated until all support vector be found.
while (numChanged > 0) | examineAll
numChanged = 0;
if examineAll
%Loop sobre todos los puntos
for i = 1:ntp
numChanged = numChanged + examineExample(i);
end;
else
%Loop sobre KKT points
for i = 1:ntp
%Solo los puntos que violan las condiciones KKT
if (SMO.alpha(i)>SMO.epsilon) & (SMO.alpha(i)<(SMO.C-SMO.epsilon))
numChanged = numChanged + examineExample(i);
end;
end;
end;

if (examineAll == 1)
examineAll = 0;
elseif (numChanged == 0)
examineAll = 1;
end;

epoch = epoch+1;
% trerror = 1; %100*sum((error)<0)/ntp;
% fprintf('Epoch: %d, TR Error: %g%%, numChanged: %d, alpha>0: %d, 0<alpha<C: %d \n',...
% epoch,...
% trerror,...
% numChanged,...
% nonZeroLagrangeMultipliers,...
% nonBoundLagrangeMultipliers);

%WRITE RESULTADOS A DISCO, W, B, ERROR
end;
SMO.epochs = epoch;
RESULT = SMO;
return;

function RESULT = nonZeroLagrangeMultipliers;
global SMO;
RESULT = sum(SMO.alpha>SMO.epsilon);
return;

function RESULT = nonBoundLagrangeMultipliers;
global SMO;
RESULT = sum((SMO.alpha>SMO.epsilon) & (SMO.alpha<(SMO.C-SMO.epsilon)));
return;

function RESULT = fwd(n)
global SMO;
LN = length(n);
RESULT = -SMO.bias + sum(repmat(SMO.y,1,LN) .* repmat(SMO.alpha,1,LN) .* SMO.Kcache(:,n))';
return;

function RESULT = examineExample(i2)
%First heuristic selects i2 and asks to examineExample to find a
%second point (i1) in order to do an optimization step with two
%Lagrange multipliers

global SMO;
alpha2 = SMO.alpha(i2); y2 = SMO.y(i2);

if ((alpha2 > SMO.epsilon) & (alpha2 < (SMO.C-SMO.epsilon)))
e2 = SMO.error(i2);
else
e2 = fwd(i2) - y2;
end;

% r2 < 0 if point i2 is placed between margin (-1)-(+1)
% Otherwise r2 is > 0. r2 = f2*y2-1

r2 = e2*y2;
%KKT conditions:
% r2>0 and alpha2==0 (well classified)
% r2==0 and 0<alpha2<C (support vectors at margins)
% r2<0 and alpha2==C (support vectors between margins)
%
% Test the KKT conditions for the current i2 point.
%
% If a point is well classified its alpha must be 0 or if
% it is out of its margin its alpha must be C. If it is at margin
% its alpha must be between 0<alpha2<C.

%take action only if i2 violates Karush-Kuhn-Tucker conditions
if ((r2 < -SMO.tolerance) & (alpha2 < (SMO.C-SMO.epsilon))) | ...
((r2 > SMO.tolerance) & (alpha2 > SMO.epsilon))
% If it doens't violate KKT conditions then exit, otherwise continue.

%Try i2 by three ways; if successful, then immediately return 1;
RESULT = 1;
% First the routine tries to find an i1 lagrange multiplier that
% maximizes the measure |E1-E2|. As large this value is as bigger
% the dual objective function becames.
% In this first test, only support vectors will be tested.

POS = find((SMO.alpha > SMO.epsilon) & (SMO.alpha < (SMO.C-SMO.epsilon)));
[MAX,i1] = max(abs(e2 - SMO.error(POS)));
if ~isempty(i1)
if takeStep(i1, i2, e2), return; end;
end;

%The second heuristic choose any Lagrange Multiplier that is a SV and tries to optimize
for i1 = randperm(SMO.ntp)
if (SMO.alpha(i1) > SMO.epsilon) & (SMO.alpha(i1) < (SMO.C-SMO.epsilon))
%if a good i1 is found, optimise
if takeStep(i1, i2, e2), return; end;
end
end

%if both heuristc above fail, iterate over all data set
for i1 = randperm(SMO.ntp)
if ~((SMO.alpha(i1) > SMO.epsilon) & (SMO.alpha(i1) < (SMO.C-SMO.epsilon)))
if takeStep(i1, i2, e2), return; end;
end
end;
end;

%no progress possible
RESULT = 0;
return;


function RESULT = takeStep(i1, i2, e2)
% for a pair of alpha indexes, verify if it is possible to execute
% the optimisation described by Platt.

global SMO;
RESULT = 0;
if (i1 == i2), return; end;

% compute upper and lower constraints, L and H, on multiplier a2
alpha1 = SMO.alpha(i1); alpha2 = SMO.alpha(i2);
x1 = SMO.x(i1); x2 = SMO.x(i2);
y1 = SMO.y(i1); y2 = SMO.y(i2);
C = SMO.C; K = SMO.Kcache;

s = y1*y2;
if (y1 ~= y2)
L = max(0, alpha2-alpha1); H = min(C, alpha2-alpha1+C);
else
L = max(0, alpha1+alpha2-C); H = min(C, alpha1+alpha2);
end;

if (L == H), return; end;

if (alpha1 > SMO.epsilon) & (alpha1 < (C-SMO.epsilon))
e1 = SMO.error(i1);
else
e1 = fwd(i1) - y1;
end;

%if (alpha2 > SMO.epsilon) & (alpha2 < (C-SMO.epsilon))
% e2 = SMO.error(i2);
%else
% e2 = fwd(i2) - y2;
%end;

%compute eta
k11 = K(i1,i1); k12 = K(i1,i2); k22 = K(i2,i2);
eta = 2.0*k12-k11-k22;

%recompute Lagrange multiplier for pattern i2
if (eta < 0.0)
a2 = alpha2 - y2*(e1 - e2)/eta;

%constrain a2 to lie between L and H
if (a2 < L)
a2 = L;
elseif (a2 > H)
a2 = H;
end;
else
%When eta is not negative, the objective function W should be
%evaluated at each end of the line segment. Only those terms in the
%objective function that depend on alpha2 need be evaluated...

ind = find(SMO.alpha>0);

aa2 = L; aa1 = alpha1 + s*(alpha2-aa2);

Lobj = aa1 + aa2 + sum((-y1*aa1/2).*SMO.y(ind).*K(ind,i1) + (-y2*aa2/2).*SMO.y(ind).*K(ind,i2));

aa2 = H; aa1 = alpha1 + s*(alpha2-aa2);
Hobj = aa1 + aa2 + sum((-y1*aa1/2).*SMO.y(ind).*K(ind,i1) + (-y2*aa2/2).*SMO.y(ind).*K(ind,i2));

if (Lobj>Hobj+SMO.epsilon)
a2 = L;
elseif (Lobj<Hobj-SMO.epsilon)
a2 = H;
else
a2 = alpha2;
end;
end;

if (abs(a2-alpha2) < SMO.epsilon*(a2+alpha2+SMO.epsilon))
return;
end;

% recompute Lagrange multiplier for pattern i1
a1 = alpha1 + s*(alpha2-a2);

w1 = y1*(a1 - alpha1); w2 = y2*(a2 - alpha2);

%update threshold to reflect change in Lagrange multipliers
b1 = SMO.bias + e1 + w1*k11 + w2*k12;
bold = SMO.bias;

if (a1>SMO.epsilon) & (a1<(C-SMO.epsilon))
SMO.bias = b1;
else
b2 = SMO.bias + e2 + w1*k12 + w2*k22;
if (a2>SMO.epsilon) & (a2<(C-SMO.epsilon))
SMO.bias = b2;
else
SMO.bias = (b1 + b2)/2;
end;
end;

% update error cache using new Lagrange multipliers
SMO.error = SMO.error + w1*K(:,i1) + w2*K(:,i2) + bold - SMO.bias;
SMO.error(i1) = 0.0; SMO.error(i2) = 0.0;

% update vector of Lagrange multipliers
SMO.alpha(i1) = a1; SMO.alpha(i2) = a2;

%report progress made
RESULT = 1;
return;

%*********************************************************************

function RESULT = SMOTutorNOBIAS(x,y,C,kernel,alpha_init,bias_init);
%Implementation of the Sequential Minimal Optimization (SMO)
%training algorithm for Vapnik's Support Vector Machine (SVM)

global SMO;

[ntp,d] = size(x);
%Inicializando las variables
SMO.epsilon = svtol(C);
SMO.x = x; SMO.y = y;
SMO.C = C; SMO.kernel = kernel;
SMO.alpha = alpha_init;

fprintf('We will not use bias. Setting bias to zero.\n')
SMO.bias = 0; %Implicit bias = 0

SMO.ntp = ntp; %number of training points

%CACHES:
SMO.Kcache = evaluate(kernel,x,x); %kernel evaluations
SMO.error = zeros(SMO.ntp,1); %error

if ~any(SMO.alpha)
%Como todos los alpha(i) son zeros, entonces fwd(i), tambien es zero
SMO.error = -y;
else
SMO.error = fwd(1:ntp) - y;
end;

numChanged = 0; examineAll = 1;
epoch = 0;

%When all data were examined and no changes done the loop reachs its
%end. Otherwise, loops with all data and likely support vector are
%alternated until all support vector be found.
while (numChanged > 0) | examineAll
numChanged = 0;
%FIRST CHOICE HEURISTIC
%THE OUTER LOOP
if examineAll
%Loop sobre todos los puntos
for i = 1:ntp
numChanged = numChanged + examineExampleNOBIAS(i);
end;
else
%Loop sobre KKT points
for i = 1:ntp
%Solo los puntos que violan las condiciones KKT
if (SMO.alpha(i)>SMO.epsilon) & (SMO.alpha(i)<(SMO.C-SMO.epsilon))
numChanged = numChanged + examineExampleNOBIAS(i);
end;
end;
end;

if (examineAll == 1)
examineAll = 0;
elseif (numChanged == 0)
examineAll = 1;
end;

% epoch = epoch+1;
% trerror = 1; %100*sum((error)<0)/ntp;
% fprintf('Epoch: %d, TR Error: %g%%, numChanged: %d, alpha>0: %d, 0<alpha<C: %d \n',...
% epoch,...
% trerror,...
% numChanged,...
% nonZeroLagrangeMultipliers,...
% nonBoundLagrangeMultipliers);

%WRITE RESULTADOS A DISCO, W, B, ERROR
end;
SMO.epochs = epoch;
RESULT = SMO;
return;

function RESULT = examineExampleNOBIAS(i1)
RESULT = takeStepNOBIAS(i1);
return;

function RESULT = takeStepNOBIAS(i1)
global SMO;

alpha1 = SMO.alpha(i1);
x1 = SMO.x(i1);
y1 = SMO.y(i1);
C = SMO.C;
K = SMO.Kcache;

if (alpha1 > SMO.epsilon) & (alpha1 < (C-SMO.epsilon))
e1 = SMO.error(i1);
else
e1 = fwd(i1) - y1;
end;

%constrain a1 to lie between L and H
a1 = min(max(0,alpha1 - y1*e1/K(i1,i1)),C);

if (abs(a1-alpha1) < SMO.epsilon*(a1+alpha1+SMO.epsilon))
RESULT = 0;
return;
end;

% update error cache using new Lagrange multipliers
SMO.error = SMO.error + y1*(a1 - alpha1)*K(:,i1);
SMO.error(i1) = 0.0;

% update vector of Lagrange multipliers
SMO.alpha(i1) = a1;

%report progress made
RESULT = 1;
return;
