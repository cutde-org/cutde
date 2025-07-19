function [Stress,Strain]=TDstressFS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,mu,lambda)
% TDstressFS
% calculates stresses and strains associated with a triangular dislocation
% in an elastic full-space.
%
% TD: Triangular Dislocation
% EFCS: Earth-Fixed Coordinate System
% TDCS: Triangular Dislocation Coordinate System
% ADCS: Angular Dislocation Coordinate System
%
% INPUTS
% X, Y and Z:
% Coordinates of calculation points in EFCS (East, North, Up). X, Y and Z
% must have the same size.
%
% P1,P2 and P3:
% Coordinates of TD vertices in EFCS.
%
% Ss, Ds and Ts:
% TD slip vector components (Strike-slip, Dip-slip, Tensile-slip).
%
% mu and lambda:
% Lame constants.
%
% OUTPUTS
% Stress:
% Calculated stress tensor components in EFCS. The six columns of Stress
% are Sxx, Syy, Szz, Sxy, Sxz and Syz, respectively. The stress components
% have the same unit as Lame constants.
%
% Strain:
% Calculated strain tensor components in EFCS. The six columns of Strain
% are Exx, Eyy, Ezz, Exy, Exz and Eyz, respectively. The strain components
% are dimensionless.
%
%
% Example: Calculate and plot the first component of stress tensor on a
% regular grid.
%
% [X,Y,Z] = meshgrid(-3:.02:3,-3:.02:3,2);
% [Stress,Strain] = TDstressFS(X,Y,Z,[-1 0 0],[1 -1 -1],[0 1.5 .5],...
% -1,2,3,.33e11,.33e11);
% h = surf(X,Y,reshape(Stress(:,1),size(X)),'edgecolor','none');
% view(2)
% axis equal
% axis tight
% set(gcf,'renderer','painters')

% Reference journal article:
% Nikkhoo, M., Walter, T. R. (2015): Triangular dislocation: an analytical,
% artefact-free solution. - Geophysical Journal International, 201,
% 1117-1139. doi: 10.1093/gji/ggv035

% Copyright (c) 2014 Mehdi Nikkhoo
%
% Permission is hereby granted, free of charge, to any person obtaining a
% copy of this software and associated documentation files
% (the "Software"), to deal in the Software without restriction, including
% without limitation the rights to use, copy, modify, merge, publish,
% distribute, sublicense, and/or sell copies of the Software, and to permit
% persons to whom the Software is furnished to do so, subject to the
% following conditions:
%
% The above copyright notice and this permission notice shall be included
% in all copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
% OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
% MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
% NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
% DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
% OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
% USE OR OTHER DEALINGS IN THE SOFTWARE.

% I appreciate any comments or bug reports.

% Update report No. 1:
% 1) Bug description: The equation for Poisson's ratio (Line 91 in the
% previous version) was valid only for a Poisson solid with mu = lambda.
% Bug fixed: The issue has been fixed and the correct equation has been
% replaced (Line 99 in this version).

% Mehdi Nikkhoo
% created: 2012.5.14
% Last modified: 2016.10.18
%
% Section 2.1, Physics of Earthquakes and Volcanoes
% Department 2, Geophysics
% Helmholtz Centre Potsdam
% German Research Centre for Geosciences (GFZ)
%
% Email:
% mehdi.nikkhoo@gfz-potsdam.de
% mehdi.nikkhoo@gmail.com
%
% website:
% http://www.volcanodeformation.com

nu = lambda/(lambda+mu)/2; % Poisson's ratio

bx = Ts; % Tensile-slip
by = Ss; % Strike-slip
bz = Ds; % Dip-slip

X = X(:);
Y = Y(:);
Z = Z(:);

P1 = P1(:);
P2 = P2(:);
P3 = P3(:);

% Calculate unit strike, dip and normal to TD vectors: For a horizontal TD
% as an exception, if the normal vector points upward, the strike and dip
% vectors point Northward and Westward, whereas if the normal vector points
% downward, the strike and dip vectors point Southward and Westward,
% respectively.
Vnorm = cross(P2-P1,P3-P1);
Vnorm = Vnorm/norm(Vnorm);

eY = [0 1 0]';
eZ = [0 0 1]';
Vstrike = cross(eZ,Vnorm);

if norm(Vstrike)==0
    Vstrike = eY*Vnorm(3);
end
Vstrike = Vstrike/norm(Vstrike);
Vdip = cross(Vnorm,Vstrike);

% Transform coordinates from EFCS into TDCS
p1 = zeros(3,1);
p2 = zeros(3,1);
p3 = zeros(3,1);
A = [Vnorm Vstrike Vdip]';
[x,y,z] = CoordTrans(X'-P2(1),Y'-P2(2),Z'-P2(3),A);
[p1(1),p1(2),p1(3)] = CoordTrans(P1(1)-P2(1),P1(2)-P2(2),P1(3)-P2(3),A);
[p3(1),p3(2),p3(3)] = CoordTrans(P3(1)-P2(1),P3(2)-P2(2),P3(3)-P2(3),A);

% Calculate the unit vectors along TD sides in TDCS
e12 = (p2-p1)/norm(p2-p1);
e13 = (p3-p1)/norm(p3-p1);
e23 = (p3-p2)/norm(p3-p2);

% Calculate the TD angles
A = acos(e12'*e13);
B = acos(-e12'*e23);
C = acos(e23'*e13);

% Determine the best arteact-free configuration for each calculation point
Trimode = trimodefinder(y,z,x,p1(2:3),p2(2:3),p3(2:3));
casepLog = Trimode==1;
casenLog = Trimode==-1;
casezLog = Trimode==0;
xp = x(casepLog);
yp = y(casepLog);
zp = z(casepLog);
xn = x(casenLog);
yn = y(casenLog);
zn = z(casenLog);

% Configuration I
if nnz(casepLog)~=0
    % Calculate first angular dislocation contribution
    [Exx1Tp,Eyy1Tp,Ezz1Tp,Exy1Tp,Exz1Tp,Eyz1Tp] = TDSetupS(xp,yp,zp,A,...
        bx,by,bz,nu,p1,-e13);
    % Calculate second angular dislocation contribution
    [Exx2Tp,Eyy2Tp,Ezz2Tp,Exy2Tp,Exz2Tp,Eyz2Tp] = TDSetupS(xp,yp,zp,B,...
        bx,by,bz,nu,p2,e12);
    % Calculate third angular dislocation contribution
    [Exx3Tp,Eyy3Tp,Ezz3Tp,Exy3Tp,Exz3Tp,Eyz3Tp] = TDSetupS(xp,yp,zp,C,...
        bx,by,bz,nu,p3,e23);
end

% Configuration II
if nnz(casenLog)~=0
    % Calculate first angular dislocation contribution
    [Exx1Tn,Eyy1Tn,Ezz1Tn,Exy1Tn,Exz1Tn,Eyz1Tn] = TDSetupS(xn,yn,zn,A,...
        bx,by,bz,nu,p1,e13);
    % Calculate second angular dislocation contribution
    [Exx2Tn,Eyy2Tn,Ezz2Tn,Exy2Tn,Exz2Tn,Eyz2Tn] = TDSetupS(xn,yn,zn,B,...
        bx,by,bz,nu,p2,-e12);
    % Calculate third angular dislocation contribution
    [Exx3Tn,Eyy3Tn,Ezz3Tn,Exy3Tn,Exz3Tn,Eyz3Tn] = TDSetupS(xn,yn,zn,C,...
        bx,by,bz,nu,p3,-e23);
end

% Calculate the strain tensor components in TDCS
if nnz(casepLog)~=0
    exx(casepLog,1) = Exx1Tp+Exx2Tp+Exx3Tp;
    eyy(casepLog,1) = Eyy1Tp+Eyy2Tp+Eyy3Tp;
    ezz(casepLog,1) = Ezz1Tp+Ezz2Tp+Ezz3Tp;
    exy(casepLog,1) = Exy1Tp+Exy2Tp+Exy3Tp;
    exz(casepLog,1) = Exz1Tp+Exz2Tp+Exz3Tp;
    eyz(casepLog,1) = Eyz1Tp+Eyz2Tp+Eyz3Tp;
end
if nnz(casenLog)~=0
    exx(casenLog,1) = Exx1Tn+Exx2Tn+Exx3Tn;
    eyy(casenLog,1) = Eyy1Tn+Eyy2Tn+Eyy3Tn;
    ezz(casenLog,1) = Ezz1Tn+Ezz2Tn+Ezz3Tn;
    exy(casenLog,1) = Exy1Tn+Exy2Tn+Exy3Tn;
    exz(casenLog,1) = Exz1Tn+Exz2Tn+Exz3Tn;
    eyz(casenLog,1) = Eyz1Tn+Eyz2Tn+Eyz3Tn;
end
if nnz(casezLog)~=0
    exx(casezLog,1) = nan;
    eyy(casezLog,1) = nan;
    ezz(casezLog,1) = nan;
    exy(casezLog,1) = nan;
    exz(casezLog,1) = nan;
    eyz(casezLog,1) = nan;
end

% Transform the strain tensor components from TDCS into EFCS
[Exx,Eyy,Ezz,Exy,Exz,Eyz] = TensTrans(exx,eyy,ezz,exy,exz,eyz,...
    [Vnorm,Vstrike,Vdip]);

% Calculate the stress tensor components in EFCS
Sxx = 2*mu*Exx+lambda*(Exx+Eyy+Ezz);
Syy = 2*mu*Eyy+lambda*(Exx+Eyy+Ezz);
Szz = 2*mu*Ezz+lambda*(Exx+Eyy+Ezz);
Sxy = 2*mu*Exy;
Sxz = 2*mu*Exz;
Syz = 2*mu*Eyz;

Strain = [Exx,Eyy,Ezz,Exy,Exz,Eyz];
Stress = [Sxx,Syy,Szz,Sxy,Sxz,Syz];

function [Txx2,Tyy2,Tzz2,Txy2,Txz2,Tyz2]=TensTrans(Txx1,Tyy1,Tzz1,Txy1,...
    Txz1,Tyz1,A)
% TensTrans Transforms the coordinates of tensors,from x1y1z1 coordinate
% system to x2y2z2 coordinate system. "A" is the transformation matrix,
% whose columns e1,e2 and e3 are the unit base vectors of the x1y1z1. The
% coordinates of e1,e2 and e3 in A must be given in x2y2z2. The transpose
% of A (i.e., A') does the transformation from x2y2z2 into x1y1z1.
Txx2 = A(1)^2*Txx1+2*A(1)*A(4)*Txy1+2*A(1)*A(7)*Txz1+2*A(4)*A(7)*Tyz1+...
    A(4)^2*Tyy1+A(7)^2*Tzz1;
Tyy2 = A(2)^2*Txx1+2*A(2)*A(5)*Txy1+2*A(2)*A(8)*Txz1+2*A(5)*A(8)*Tyz1+...
    A(5)^2*Tyy1+A(8)^2*Tzz1;
Tzz2 = A(3)^2*Txx1+2*A(3)*A(6)*Txy1+2*A(3)*A(9)*Txz1+2*A(6)*A(9)*Tyz1+...
    A(6)^2*Tyy1+A(9)^2*Tzz1;
Txy2 = A(1)*A(2)*Txx1+(A(1)*A(5)+A(2)*A(4))*Txy1+(A(1)*A(8)+...
    A(2)*A(7))*Txz1+(A(8)*A(4)+A(7)*A(5))*Tyz1+A(5)*A(4)*Tyy1+...
    A(7)*A(8)*Tzz1;
Txz2 = A(1)*A(3)*Txx1+(A(1)*A(6)+A(3)*A(4))*Txy1+(A(1)*A(9)+...
    A(3)*A(7))*Txz1+(A(9)*A(4)+A(7)*A(6))*Tyz1+A(6)*A(4)*Tyy1+...
    A(7)*A(9)*Tzz1;
Tyz2 = A(2)*A(3)*Txx1+(A(3)*A(5)+A(2)*A(6))*Txy1+(A(3)*A(8)+...
    A(2)*A(9))*Txz1+(A(8)*A(6)+A(9)*A(5))*Tyz1+A(5)*A(6)*Tyy1+...
    A(8)*A(9)*Tzz1;

function [X1,X2,X3]=CoordTrans(x1,x2,x3,A)
% CoordTrans transforms the coordinates of the vectors, from
% x1x2x3 coordinate system to X1X2X3 coordinate system. "A" is the
% transformation matrix, whose columns e1,e2 and e3 are the unit base
% vectors of the x1x2x3. The coordinates of e1,e2 and e3 in A must be given
% in X1X2X3. The transpose of A (i.e., A') will transform the coordinates
% from X1X2X3 into x1x2x3.

x1 = x1(:);
x2 = x2(:);
x3 = x3(:);
r = A*[x1';x2';x3'];
X1 = r(1,:)';
X2 = r(2,:)';
X3 = r(3,:)';

function [trimode]=trimodefinder(x,y,z,p1,p2,p3)
% trimodefinder calculates the normalized barycentric coordinates of
% the points with respect to the TD vertices and specifies the appropriate
% artefact-free configuration of the angular dislocations for the
% calculations. The input matrices x, y and z share the same size and
% correspond to the y, z and x coordinates in the TDCS, respectively. p1,
% p2 and p3 are two-component matrices representing the y and z coordinates
% of the TD vertices in the TDCS, respectively.
% The components of the output (trimode) corresponding to each calculation
% points, are 1 for the first configuration, -1 for the second
% configuration and 0 for the calculation point that lie on the TD sides.

x = x(:);
y = y(:);
z = z(:);

a = ((p2(2)-p3(2)).*(x-p3(1))+(p3(1)-p2(1)).*(y-p3(2)))./...
    ((p2(2)-p3(2)).*(p1(1)-p3(1))+(p3(1)-p2(1)).*(p1(2)-p3(2)));
b = ((p3(2)-p1(2)).*(x-p3(1))+(p1(1)-p3(1)).*(y-p3(2)))./...
    ((p2(2)-p3(2)).*(p1(1)-p3(1))+(p3(1)-p2(1)).*(p1(2)-p3(2)));
c = 1-a-b;

trimode = ones(length(x),1);
trimode(a<=0 & b>c & c>a) = -1;
trimode(b<=0 & c>a & a>b) = -1;
trimode(c<=0 & a>b & b>c) = -1;
trimode(a==0 & b>=0 & c>=0) = 0;
trimode(a>=0 & b==0 & c>=0) = 0;
trimode(a>=0 & b>=0 & c==0) = 0;
trimode(trimode==0 & z~=0) = 1;

function [exx,eyy,ezz,exy,exz,eyz]=TDSetupS(x,y,z,alpha,bx,by,bz,nu,...
    TriVertex,SideVec)
% TDSetupS transforms coordinates of the calculation points as well as
% slip vector components from ADCS into TDCS. It then calculates the
% strains in ADCS and transforms them into TDCS.

% Transformation matrix
A = [[SideVec(3);-SideVec(2)] SideVec(2:3)]';

% Transform coordinates of the calculation points from TDCS into ADCS
r1 = A*[y'-TriVertex(2);z'-TriVertex(3)];
y1 = r1(1,:)';
z1 = r1(2,:)';

% Transform the in-plane slip vector components from TDCS into ADCS
r2 = A*[by;bz];
by1 = r2(1,:)';
bz1 = r2(2,:)';

% Calculate strains associated with an angular dislocation in ADCS
[exx,eyy,ezz,exy,exz,eyz] = AngDisStrain(x,y1,z1,-pi+alpha,bx,by1,bz1,nu);

% Transform strains from ADCS into TDCS
B = [[1 0 0];[zeros(2,1),A']]; % 3x3 Transformation matrix
[exx,eyy,ezz,exy,exz,eyz] = TensTrans(exx,eyy,ezz,exy,exz,eyz,B);

function [Exx,Eyy,Ezz,Exy,Exz,Eyz]=AngDisStrain(x,y,z,alpha,bx,by,bz,nu)
% AngDisStrain calculates the strains associated with an angular
% dislocation in an elastic full-space.

sinA = sin(alpha);
cosA = cos(alpha);
eta = y.*cosA-z.*sinA;
zeta = y.*sinA+z.*cosA;

x2 = x.^2;
y2 = y.^2;
z2 = z.^2;
r2 = x2+y2+z2;
r = sqrt(r2);
r3 = r.*r2;
rz = r.*(r-z);
r2z2 = r2.*(r-z).^2;
r3z = r3.*(r-z);

W = zeta-r;
W2 = W.^2;
Wr = W.*r;
W2r = W2.*r;
Wr3 = W.*r3;
W2r2 = W2.*r2;

C = (r*cosA-z)./Wr;
S = (r*sinA-y)./Wr;

% Partial derivatives of the Burgers' function
rFi_rx = (eta./r./(r-zeta)-y./r./(r-z))/4/pi;
rFi_ry = (x./r./(r-z)-cosA*x./r./(r-zeta))/4/pi;
rFi_rz = (sinA*x./r./(r-zeta))/4/pi;

Exx = bx.*(rFi_rx)+...
    bx/8/pi/(1-nu)*(eta./Wr+eta.*x2./W2r2-eta.*x2./Wr3+y./rz-...
    x2.*y./r2z2-x2.*y./r3z)-...
    by*x/8/pi/(1-nu).*(((2*nu+1)./Wr+x2./W2r2-x2./Wr3)*cosA+...
    (2*nu+1)./rz-x2./r2z2-x2./r3z)+...
    bz*x*sinA/8/pi/(1-nu).*((2*nu+1)./Wr+x2./W2r2-x2./Wr3);

Eyy = by.*(rFi_ry)+...
    bx/8/pi/(1-nu)*((1./Wr+S.^2-y2./Wr3).*eta+(2*nu+1)*y./rz-y.^3./r2z2-...
    y.^3./r3z-2*nu*cosA*S)-...
    by*x/8/pi/(1-nu).*(1./rz-y2./r2z2-y2./r3z+...
    (1./Wr+S.^2-y2./Wr3)*cosA)+...
    bz*x*sinA/8/pi/(1-nu).*(1./Wr+S.^2-y2./Wr3);

Ezz = bz.*(rFi_rz)+...
    bx/8/pi/(1-nu)*(eta./W./r+eta.*C.^2-eta.*z2./Wr3+y.*z./r3+...
    2*nu*sinA*C)-...
    by*x/8/pi/(1-nu).*((1./Wr+C.^2-z2./Wr3)*cosA+z./r3)+...
    bz*x*sinA/8/pi/(1-nu).*(1./Wr+C.^2-z2./Wr3);

Exy = bx.*(rFi_ry)./2+by.*(rFi_rx)./2-...
    bx/8/pi/(1-nu).*(x.*y2./r2z2-nu*x./rz+x.*y2./r3z-nu*x*cosA./Wr+...
    eta.*x.*S./Wr+eta.*x.*y./Wr3)+...
    by/8/pi/(1-nu)*(x2.*y./r2z2-nu*y./rz+x2.*y./r3z+nu*cosA*S+...
    x2.*y*cosA./Wr3+x2*cosA.*S./Wr)-...
    bz*sinA/8/pi/(1-nu).*(nu*S+x2.*S./Wr+x2.*y./Wr3);

Exz = bx.*(rFi_rz)./2+bz.*(rFi_rx)./2-...
    bx/8/pi/(1-nu)*(-x.*y./r3+nu*x*sinA./Wr+eta.*x.*C./Wr+...
    eta.*x.*z./Wr3)+...
    by/8/pi/(1-nu)*(-x2./r3+nu./r+nu*cosA*C+x2.*z*cosA./Wr3+...
    x2*cosA.*C./Wr)-...
    bz*sinA/8/pi/(1-nu).*(nu*C+x2.*C./Wr+x2.*z./Wr3);

Eyz = by.*(rFi_rz)./2+bz.*(rFi_ry)./2+...
    bx/8/pi/(1-nu).*(y2./r3-nu./r-nu*cosA*C+nu*sinA*S+eta*sinA*cosA./W2-...
    eta.*(y*cosA+z*sinA)./W2r+eta.*y.*z./W2r2-eta.*y.*z./Wr3)-...
    by*x/8/pi/(1-nu).*(y./r3+sinA*cosA^2./W2-cosA*(y*cosA+z*sinA)./...
    W2r+y.*z*cosA./W2r2-y.*z*cosA./Wr3)-...
    bz*x*sinA/8/pi/(1-nu).*(y.*z./Wr3-sinA*cosA./W2+(y*cosA+z*sinA)./...
    W2r-y.*z./W2r2);
