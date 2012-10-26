%EXAMINESAMPLE check whether current chosing variable violate
%  KKT condition. If satisfied, find another variable and 
%  test to see if they can make value of objective function
%  descending.
%
%  Main Procedures for finding another variable:
%    1.find one maximize |E2-E1|;
%    2.find one on the separating boundary;
%    3.loop all variables.
%
%  @date: 10/26/2012
%

function status = exampleSample(i1)
% defined in other place
global train_set;
global C;
global tolerance;
global alpha;
global error_cache;
global tr_ins_num;

y1 = train_set.tag(i1);
alpha1 = alpha(i1);
if alpha1>0&&alpha1<C,
    E1 = error_cache(i1);
else,
    E1 = learned_func(i1) - y1;
end

r1 = y1*E1;
disp('i1,y1,alpha1,')
i1,y1,alpha1
pause;

if (r1 < -tolerance & alpha1 < C) | (r1 > tolerance & alpha1 > 0),
    % 1.find one maximize |E2-E1|;
    i2 = -1;
    tmax = 0;
    for k=1:tr_ins_num,
        if alpha(k) > 0 && alpha(k) < C,
            E2 = error_cache(k);
            temp = abs(E1-E2);
            if temp > tmax,
                tmax = temp;
                i2 = k;
            end
        end
    end
    disp('i2, tmax');
    i2
    pause;
    if i2 > 0,
        if takeStep(i1, i2),
            status = 1;
            return; 
        end
    end

    % 2.find one on the separating boundary;
    k0 = unidrnd(tr_ins_num);
    for k=k0:tr_ins_num+k0-1,
        i2 = rem(k, tr_ins_num)+1;
        if alpha(i2) > 0 && alpha(i2) < C,
            if takeStep(i1, i2),
                status = 1;
                return;
            end
        end
    end

    % 3.loop all variables.
    k0 = unidrnd(tr_ins_num)
    for k=k0:tr_ins_num+k0-1,
        i2 = rem(k, tr_ins_num)+1;
        if takeStep(i1, i2),
            status = 1;
            return;
        end
    end
end

status = 0;
return ;
