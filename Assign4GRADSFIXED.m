clear all; clc;
% Read in data
book_fname = 'goblet_book.txt';
fid = fopen(book_fname,'r');
book_data = fscanf(fid,'%c');
fclose(fid);

% get unique characters in text, dim = 1 x 80 
book_chars = unique(book_data);

% map character to index
char_to_ind = containers.Map('KeyType','char','ValueType','int32');
ind_to_char = containers.Map('KeyType','int32','ValueType','char');
for i = 1:length(book_chars)
    char_to_ind(book_chars(1,i)) = i;
    ind_to_char(i) = book_chars(1,i);
end

% Set hyper-parameters and init RNN parameters
K = length(book_chars); % = 80
m = 5%100; for testing grads
sig = 0.01;
eta = 0.1;
seq_length = 25;

RNN = parameter_object;
RNN.b = zeros(m,1);
RNN.c = zeros(K,1);
RNN.U = randn(m, K)*sig;
RNN.W = randn(m, m)*sig;
RNN.V = randn(K, m)*sig;


% FOR SYNTHE TEXT
% h0 = m×1, xt = d×1, 
% xkk = char_to_ind(book_data(1,40)); % pick out a char
% x0 =double(bsxfun(@eq, (xkk), 1:K)'); % one hot it
% A = rand(m,1);
% h0=A./sum(A); % generate random hidden state vector
% n = 10;
% Y = Synthesize_text(RNN, h0, x0, n);
%  
%d= {''};
%for i = 1:n
%    g = ind_to_char(find(Y(:,i)));
%    d = strcat(d,{g}); 
%end
%disp(d)

% ENDS HERE

% Forward and backward pass of backprop
X_chars = book_data(1:seq_length);
Y_chars = book_data(2:seq_length+1); % next character after X
X1 = zeros(1, seq_length);
Y1 = zeros(1, seq_length);

for i = 1:seq_length % conv to indices
    X1(1,i) = char_to_ind(X_chars(1,i));
    Y1(1,i) = char_to_ind(Y_chars(1,i));
end

% make one hot
X = double(bsxfun(@eq, X1(:), 1:K))'; % one-hot representation for labels between 1 and 10
Y = double(bsxfun(@eq, Y1(:), 1:K))'; % 83x25
h0 = zeros(m,1);
[loss, out_vec, H_t, P, A_t] = forward_pass(RNN, h0, X, Y);
grads = backward_pass(RNN, X, Y, P, H_t, A_t);
% clip the gradients
% % for f = fieldnames(grads)'
% %     grads.(f{1}) = max(min(grads.(f{1}), 5), -5);
% % end


%Check grad calc
h = 1e-4;
num_grads = ComputeGradsNum(X, Y, RNN, h);

eps = 1e-6;  % Seems to be working fine
errorW = sqrt(sumsqr(grads.W-num_grads.W))/max(eps, sqrt(sumsqr(grads.W)) + sqrt(sumsqr(num_grads.W)))
errorU = sqrt(sumsqr(grads.U-num_grads.U))/max(eps, sqrt(sumsqr(grads.U)) + sqrt(sumsqr(num_grads.U)))
errorV = sqrt(sumsqr(grads.V-num_grads.V))/max(eps, sqrt(sumsqr(grads.V)) + sqrt(sumsqr(num_grads.V)))
errorb = sqrt(sumsqr(grads.b-num_grads.b))/max(eps, sqrt(sumsqr(grads.b)) + sqrt(sumsqr(num_grads.b)))
errorc = sqrt(sumsqr(grads.c-num_grads.c))/max(eps, sqrt(sumsqr(grads.c)) + sqrt(sumsqr(num_grads.c)))

% VARF�R BLIR W NOLL?!!! DET �R JU HENNES SOM BLIR FEL!


function [loss, out_vec, H_t, P, A_t] = forward_pass(RNN, h0, X, Y)

    get_size = size(X);
    n = get_size(1,2);
    get_size2 = size(h0);
    H_t = zeros(get_size2(1,1),n);
    A_t = zeros(get_size2(1,1),n);
    P = zeros(get_size(1,1),n);
    loss = zeros(1,n); % sum at end
    out_vec = zeros(get_size(1,1),n); % contains final and intermediary outp vect
    h_t1 = h0;
    for i = 1:n
        a_t = RNN.W*h_t1 + RNN.U*X(:,i) + RNN.b;
        h_t = tanh(a_t);
        o_t = RNN.V*h_t + RNN.c;
        p_t = softmax(o_t);
        loss(1,i) = -(log(Y(:,i)'*p_t));
        P(:,i) = p_t;
        H_t(:,i) = h_t;
        h_t1 = h_t; % Save previous
        A_t(:,i) = a_t;
        out_vec(:,i) = o_t;
    end
    loss = sum(loss);
end

function [grads] = backward_pass(RNN, X, Y, P, H, A)
    % Grads we want b, c, W, U, V
    get_size = size(RNN.U);
    batch_dim = size(X);
    m = get_size(1,1);
    K = get_size(1,2);
    dL_db = zeros(m,1); 
    dL_dc = zeros(K,1);
    dL_dV = zeros(K, m);
    dL_dW = zeros(m, m); 
    dL_dU = zeros(m, K);
    dL_datplus = zeros(1,m);
    
    for i = 1:batch_dim(2)%batch_dim(2):-1:1
        % init

        y = Y(:,i);
        p = P(:,i);

        h_t = H(:,i);
        
        % These only depend on current time step
        g_t = -(y-p)'; % dl_dot
        dL_dV = dL_dV + g_t'*h_t';
        dL_dc = dL_dc + g_t';
        
        % iterate back to first time steps
        if i == batch_dim(2)
            for tau = i:-1:1
            Keep_track_tau = tau
            if tau == 1
                h_tprev = zeros(m,1); % will probably cause error at start
            else
                h_tprev = H(:,tau-1); % will probably cause error at start
            end
            
            if tau == batch_dim(2) % for 'a' beyond range(a_n+1), consider it zero
                dldot = -(Y(:,tau)-P(:,tau))'; %dL_dO_t
                dldht = dldot*RNN.V;
            else
                dldot = -(Y(:,tau)-P(:,tau))';
                dldht = dldot*RNN.V + dL_datplus*RNN.W;
                
            end
            
            dL_datplus = dldht*diag(1-tanh(A(:,tau)).^2);%(dldht'*diag(1-tanh(A(:,tau)).^2))'; %(100x1 x 100x100 originally, flipped dlht to get result 1x100, potential error!

            % get dW  % H�r h�ller jag p� och joxar..
            g_t = dL_datplus;
            dL_dW = dL_dW + g_t'*h_tprev';
            % get dU
            dL_dU = dL_dU + g_t'*X(:,tau)';
            % dL_db
            dL_db = dL_db + g_t';
            end
        end
        
    end
    
    grads = grads_object;
    grads.b = dL_db;
    grads.c = dL_dc;
    grads.U = dL_dU;
    grads.W = dL_dW;
    grads.V = dL_dV;
    
end

% function [dL_ht , dL_datpl] = from_future(RNN, Y, P, A)
%     % Calculates future values of 'a' so that we can get h_t
%     get_size = size(RNN.U);
%     batch_dim = size(Y);
%     m = get_size(1,1);
%     n = get_size(1,2);
%     dL_datpl = zeros(m,n); 
%     dL_ht = zeros(m,n);
%     for i = batch_dim(2):-1:1 % starts backwards 
%         if i == batch_dim(2) % for 'a' beyond range(a_n+1), consider it zero
%             dldot = -(Y(:,i)-P(:,i))'; %dL_dO_t
%             dldht = (dldot*RNN.V)';
%             dL_ht(:,i) = dldht;
%         else
%             dldot = -(Y(:,i)-P(:,i))'; % Ser ju fel ut det h�r ocks�.. saknar lite grejs
%             dL_ht(:,i) = dldht;
%         end
%         dL_datpl(:,i) = (dldht'*diag(1-tanh(A(:,i)).^2))'; %(100x1 x 100x100 originally, flipped dlht to get result 1x100, potential error!
%     end
% end
% 
% % Synthesize text from RNN
% function Y = Synthesize_text(RNN, h0, x0, n)
%     h_t = h0;
%     x_t = x0;
%     sampled_indices = zeros(1,n);
%     for t = 1:n 
%        a_t = RNN.W*h_t + RNN.U*x_t + RNN.b;
%        h_t = tanh(a_t);
%        o_t = RNN.V*h_t + RNN.c;
%        p_t = softmax(o_t);
%        % generate next input x_t+1
%        cp = cumsum(p_t);
%        a = rand;
%        ixs = find(cp-a>0);
%        x_t = ixs(1);
%        sampled_indices(1,t) = x_t;
%     end
%    k = size(RNN.V);
%    Y = double(bsxfun(@eq, sampled_indices(:), 1:k(1,1)))'; 
% end

% function l1 = ComputeLoss(X, Y, RNN_try, hprev)
%         ComputeCost(X, Y, W1, W2, b1, b2, lambda)
% 
%         s1 = W1*X + b1;
% h = max(0, s1); % ReLU
% s = W2*h + b2;
% P = softmax(s);
%         
%     [P, ~, ~]= EvaluateClassifier(X, W1, W2, b1, b2);
%     get_dim = size(X);
%     normed_cr_entr_loss = -trace(log(Y'*P))/(get_dim(2)); % log(Y'*P) for a single sample  #CROSS ENTROPY LOSS
%     regul = lambda*(sum((sum(W1.^2,2)))+sum((sum(W2.^2,2))));
%     J =normed_cr_entr_loss+regul;
% 
% end


function num_grads = ComputeGradsNum(X, Y, RNN, h)

    for f = fieldnames(RNN)'
        disp('Computing numerical gradient for')
        disp(['Field name: ' f{1} ]);
        num_grads.(f{1}) = ComputeGradNum(X, Y, f{1}, RNN, h);
    end

    function grad = ComputeGradNum(X, Y, f, RNN, h)

        n = numel(RNN.(f));
        grad = zeros(size(RNN.(f)));
        hprev = zeros(size(RNN.W, 1), 1);
        for i=1:n
            RNN_try = RNN;
            RNN_try.(f)(i) = RNN.(f)(i) - h;
            %l1 = ComputeLoss(X, Y, RNN_try, hprev);
            [l1, out_vec, H_t, P, A_t] = forward_pass(RNN_try, hprev, X, Y); % Shooting sparrows with cannons
            RNN_try.(f)(i) = RNN.(f)(i) + h;
            %l2 = ComputeLoss(X, Y, RNN_try, hprev);
            [l2, out_vec, H_t, P, A_t] = forward_pass(RNN_try, hprev, X, Y);
            grad(i) = (l2-l1)/(2*h);
        end
    end
end


     
    







