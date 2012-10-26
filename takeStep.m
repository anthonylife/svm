%TAKESTEP give analytic solution to the Quadratic programming
%  problems and determine whether to update the chosen 
%  variables.
%
%  Main procedure:
%    1.calculate value range for i2;
%    2.calculate new value for i2 and i1;
%    3.update threshold b;
%    4.update error value.
%
%  @date: 10/26/2012
%

function status = takeStep(i1, i2)
% define in other place
global train_set;
global C;
global tolerance;
global eps;
global alpha;
global error_cache;
global tr_ins_num;
global kernel_func;
global b;

y1 = train_set.tag(i1);
y2 = train_set.tag(i2);
alpha1_old = alpha(i1);
alpha2_old = alpha(i2);
if alpha1_old>0&&alpha1_old<C,
    E1 = error_cache(i1);
else,
    E1 = learned_func(i1) - y1;
end

if alpha2_old>0&&alpha2_old<C,
    E2 = error_cache(i2);
else,
    E2 = learned_func(i2) - y2;
end

disp('i1,i2');
i1,i2

% cal value range for i2
if y1 == y2,
    if (alpha1_old+alpha2_old)>C,
        L = alpha1_old+alpha2_old-C;
        H = C;
    else
        L = 0;
        H = alpha1_old+alpha2_old;
    end
else,
    temp_diff = alpha2_old-alpha1_old ;
    if temp_diff > 0,
        L = temp_diff;
        H = C;
    else
        L = 0;
        H = C + temp_diff;
    end
end

% (C,0) or (0,C)
if L == H,
    status = 0;
    return ;
end

% 2.calculate new value for i2 and i1;
k11 = kernel_func(train_set.fea(i1,:), train_set.fea(i2,:));
k22 = kernel_func(train_set.fea(i2,:), train_set.fea(i2,:));
k12 = kernel_func(train_set.fea(i1,:), train_set.fea(i2,:));

eta = 2*k12 - k11 - k22;

if eta < 0,
    alpha2_new = alpha2_old + y2*(E2-E1)/eta;
    if alpha2_new < L,
        alpha2_new = L;
    elseif alpha2_new > H,
        alpha2_new = H;
    end
else
    lobj_re = y2*(E1-E2)*L;
    hobj_re = y2*(E1-E2)*H;
    % variation of the objective function value
    if lobj_re > hobj_re + eps,
        alpha2_new = L;
    elseif hobj_re > lobj_re + eps,
        alpha2_new = H;
    else,
        alpha2_new = alpha2_old;
    end
end

% variatio of i2 value
if abs(alpha2_new-alpha2_old)<eps*(alpha2_old+alpha2_new+eps),
    status = 0;
    return ;
end

% update i1 value
s = train_set.tag(i1)*train_set.tag(i2);
alpha1_new = alpha1_old + s*(alpha2_old - alpha2_new);

if alpha1_new < 0,
    alpha2_new = alpha2_new + s*alpha1_new;
    alpha1_new = 0;
elseif alpha1_new > C,
    alpha2_new = alpha2_new + s*(alpha1_new-C);
    alpha1_new = C;
end

% 3.update threshold b (Note:wx-b);
if alpha1_new > 0 && alpha1_new < C,
    bnew = E1 + y1*k11*(alpha1_new-alpha1_old) + y2*k12*(alpha2_new...
        -alpha2_old) + b;
else
    if alpha2_new >0 && alpha2_new < C,
        bnew = E2 + y1*k12*(alpha1_new-alpha1_old) + ...
            y2*k22*(alpha2_new-alpha2_old) + b;
    else
        %Beause two inequalities all includes 1,so b1 and b2 satisfy KKT
        b1 = E1 + y1*k11*(alpha1_new-alpha1_old) + ...
            y2*k12*(alpha2_new-alpha2_old) + b;
        b2 = E2 + y1*k12*(alpha1_new-alpha1_old) + ...
            y2*k22*(alpha2_new-alpha2_old) + b;
        bnew = (b1+b2)/2;
    end
end
delta_b = bnew-b;
size(b)
size(bnew)
b = bnew;

% 4.update error value. use difference to update.
t1 = y1*(alpha1_new - alpha1_old);
t2 = y2*(alpha2_new - alpha2_old);
for i=1:tr_ins_num,
    if alpha(i)>0 && alpha(i)<C,
        error_cache(i) = error_cache(i) + t1*kernel_func(...
            train_set.fea(i1,:), train_set.fea(i,:)) + t2* ...
            kernel_func(train_set.fea(i2,:),train_set.fea(i,:))-delta_b;
    end
end
% force
error_cache(i1) = 0;
error_cache(i2) = 0;

alpha(i1) = alpha1_new;
alpha(i2) = alpha2_new;

disp('Seperating line');
%sp = sum((alpha.*train_set.tag)'*train_set.fea, 1)
sp = 0;
for i=1:tr_ins_num,
    sp = sp + alpha(i)*train_set.tag(i)*train_set.fea(i,:);
end
sp
b
pause;

status = 1;
return ;
