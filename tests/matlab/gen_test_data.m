n = 11;
x = linspace(-3, 3, n);
y = linspace(-3, 3, n);
z = linspace(-3, 3, n);

%% simple
P1 = [0,0,0];
P2 = [1,0,0];
P3 = [0,1,0];
slip = [1.0, 0, 0];
filename = '../result_simple.mat';

[X,Y,Z] = meshgrid(x, y, z);
Xf = reshape(X, numel(X), 1);
Yf = reshape(Y, numel(Y), 1);
Zf = reshape(Z, numel(Z), 1);


% 1 million TDE evaluations takes 1 second on 1 core on meade03
tic;
[UEf, UNf, UVf] = TDdispFS(Xf, Yf, Zf, P1, P2, P3, slip(1), slip(2), slip(3), 0.25);
toc;
tic;
[Stress, Strain] = TDstressFS(Xf, Yf, Zf, P1, P2, P3, slip(1), slip(2), slip(3), 1.0, 1.0);
toc;
format long;
save(filename, 'UEf', 'UNf', 'UVf', 'Stress', 'Strain', '-v6');

%% complex
P1 = [0,0.1,0.1];
P2 = [1,-0.2,-0.2];
P3 = [1,1,0.3];
slip = [1.3,1.4,1.5];
filename = '../result_complex.mat';

[X,Y,Z] = meshgrid(x, y, z);
Xf = reshape(X, numel(X), 1);
Yf = reshape(Y, numel(Y), 1);
Zf = reshape(Z, numel(Z), 1);


% 1 million TDE evaluations takes 1 second on 1 core on meade03
tic;
[UEf, UNf, UVf] = TDdispFS(Xf, Yf, Zf, P1, P2, P3, slip(1), slip(2), slip(3), 0.25);
toc;
tic;
[Stress, Strain] = TDstressFS(Xf, Yf, Zf, P1, P2, P3, slip(1), slip(2), slip(3), 1.0, 1.0);
toc;
format long;
save(filename, 'UEf', 'UNf', 'UVf', 'Stress', 'Strain', '-v6');


%% halfspace
P1 = [0,0.1,-0.1];
P2 = [1,-0.2,-0.2];
P3 = [1,1,-0.3];
slip = [1.3,1.4,1.5];
filename = '../result_halfspace.mat';

x = linspace(-3, 3, n);
y = linspace(-3, 3, n);
z = linspace(-6, 0, n);

[X,Y,Z] = meshgrid(x, y, z);
Xf = reshape(X, numel(X), 1);
Yf = reshape(Y, numel(Y), 1);
Zf = reshape(Z, numel(Z), 1);


% 1 million TDE evaluations takes 1 second on 1 core on meade03
tic;
[UEf, UNf, UVf] = TDdispHS(Xf, Yf, Zf, P1, P2, P3, slip(1), slip(2), slip(3), 0.25);
toc;
tic;
[Stress, Strain] = TDstressHS(Xf, Yf, Zf, P1, P2, P3, slip(1), slip(2), slip(3), 1.0, 1.0);
toc;
format long;
save(filename, 'UEf', 'UNf', 'UVf', 'Stress', 'Strain', '-v6');

%% HORIZONTAL
% halfspace
P1 = [-1, 0, -1];
P2 = [1, 0., -1];
P3 = [0, 1, -1];
slip = [1.,0.5,-0.3];

x = linspace(-3, 3, n);
y = linspace(-3, 3, n);
z = linspace(-6, 0, n);

[X,Y,Z] = meshgrid(x, y, z);
Xf = reshape(X, numel(X), 1);
Yf = reshape(Y, numel(Y), 1);
Zf = reshape(Z, numel(Z), 1);

% 1 million TDE evaluations takes 1 second on 1 core on meade03
filename = '../result_halfspace_horizontal.mat';
tic;
[UEf, UNf, UVf] = TDdispHS(Xf, Yf, Zf, P1, P2, P3, slip(1), slip(2), slip(3), 0.25);
toc;
tic;
[Stress, Strain] = TDstressHS(Xf, Yf, Zf, P1, P2, P3, slip(1), slip(2), slip(3), 1.0, 1.0);
toc;
format long;
save(filename, 'UEf', 'UNf', 'UVf', 'Stress', 'Strain', '-v6');


z = linspace(-3, 3, n);

[X,Y,Z] = meshgrid(x, y, z);
Xf = reshape(X, numel(X), 1);
Yf = reshape(Y, numel(Y), 1);
Zf = reshape(Z, numel(Z), 1);

filename = '../result_fullspace_horizontal.mat';
tic;
[UEf, UNf, UVf] = TDdispFS(Xf, Yf, Zf, P1, P2, P3, slip(1), slip(2), slip(3), 0.25);
toc;
tic;
[Stress, Strain] = TDstressFS(Xf, Yf, Zf, P1, P2, P3, slip(1), slip(2), slip(3), 1.0, 1.0);
toc;
format long;
save(filename, 'UEf', 'UNf', 'UVf', 'Stress', 'Strain', '-v6');
