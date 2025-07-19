function [ue,un,uv]=TDdispHS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,nu)
% TDdispHS
% calculates displacements associated with a triangular dislocation in an
% elastic half-space.
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
% nu:
% Poisson's ratio.
%
% OUTPUTS
% ue, un and uv:
% Calculated displacement vector components in EFCS. ue, un and uv have
% the same unit as Ss, Ds and Ts in the inputs.
%
%
% Example: Calculate and plot the first component of displacement vector
% on a regular grid.
%
% [X,Y,Z] = meshgrid(-3:.02:3,-3:.02:3,-5);
% [ue,un,uv] = TDdispHS(X,Y,Z,[-1 0 0],[1 -1 -1],[0 1.5 -2],-1,2,3,.25);
% h = surf(X,Y,reshape(ue,size(X)),'edgecolor','none');
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
% 1) The solid angle calculation was modified to account for the "sign bit"
% of the absolute zero values in the numerator of the "atan2" function.
% These zeros correspond to the points that are exactly on the TD plane
% (see Lines 240-245).
%
% 2) The three vertices of a TD must not be located on the free surface
% simultaneously. The displacements corresponding to such a case are set to
% zero (see Lines 99-103 here). Accordingly, the two "if" expressions at
% Lines 107-109 and 116-120 in the previous version were removed.

% Mehdi Nikkhoo
% created: 2013.1.24
% Last modified: 2016.10.18
%
% Section 2.1, Physics of Earthquakes and Volcanoes
% Department 2, Geophysics
% Helmholtz Centre Potsdam
% German Research Centre for Geosciences (GFZ)
%
% email:
% mehdi.nikkhoo@gfz-potsdam.de
% mehdi.nikkhoo@gmail.com
%
% website:
% http://www.volcanodeformation.com

if any(Z>0 | P1(3)>0 | P2(3)>0 | P3(3)>0)
    error('Half-space solution: Z coordinates must be negative!')
elseif P1(3)==0 && P2(3)==0 && P3(3)==0
    ue = zeros(numel(X),1);
    un = zeros(numel(X),1);
    uv = zeros(numel(X),1);
    return
end

X = X(:);
Y = Y(:);
Z = Z(:);

P1 = P1(:);
P2 = P2(:);
P3 = P3(:);

% Calculate main dislocation contribution to displacements
[ueMS,unMS,uvMS] = TDdispFS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,nu);

% Calculate harmonic function contribution to displacements
[ueFSC,unFSC,uvFSC] = TDdisp_HarFunc(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,nu);

% Calculate image dislocation contribution to displacements
P1(3) = -P1(3);
P2(3) = -P2(3);
P3(3) = -P3(3);
[ueIS,unIS,uvIS] = TDdispFS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,nu);

% Calculate the complete displacement vector components in EFCS
ue = ueMS+ueIS+ueFSC;
un = unMS+unIS+unFSC;
uv = uvMS+uvIS+uvFSC;


function [ue,un,uv]=TDdispFS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,nu)
% TDdispFS
% Calculates displacements associated with a triangular dislocation in an
% elastic full-space.

bx = Ts; % Tensile-slip
by = Ss; % Strike-slip
bz = Ds; % Dip-slip

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
    % For horizontal elements in case of half-space calculation!!!
    % Correct the strike vector of image dislocation only
    if P1(3)>0
        Vstrike = -Vstrike;
    end
end
Vstrike = Vstrike/norm(Vstrike);
Vdip = cross(Vnorm,Vstrike);

% Transform coordinates and slip vector components from EFCS into TDCS
p1 = zeros(3,1);
p2 = zeros(3,1);
p3 = zeros(3,1);
At = [Vnorm Vstrike Vdip]';
[x,y,z] = CoordTrans(X'-P2(1),Y'-P2(2),Z'-P2(3),At);
[p1(1),p1(2),p1(3)] = CoordTrans(P1(1)-P2(1),P1(2)-P2(2),P1(3)-P2(3),At);
[p3(1),p3(2),p3(3)] = CoordTrans(P3(1)-P2(1),P3(2)-P2(2),P3(3)-P2(3),At);

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
    [u1Tp,v1Tp,w1Tp] = TDSetupD(xp,yp,zp,A,bx,by,bz,nu,p1,-e13);
    % Calculate second angular dislocation contribution
    [u2Tp,v2Tp,w2Tp] = TDSetupD(xp,yp,zp,B,bx,by,bz,nu,p2,e12);
    % Calculate third angular dislocation contribution
    [u3Tp,v3Tp,w3Tp] = TDSetupD(xp,yp,zp,C,bx,by,bz,nu,p3,e23);
end

% Configuration II
if nnz(casenLog)~=0
    % Calculate first angular dislocation contribution
    [u1Tn,v1Tn,w1Tn] = TDSetupD(xn,yn,zn,A,bx,by,bz,nu,p1,e13);
    % Calculate second angular dislocation contribution
    [u2Tn,v2Tn,w2Tn] = TDSetupD(xn,yn,zn,B,bx,by,bz,nu,p2,-e12);
    % Calculate third angular dislocation contribution
    [u3Tn,v3Tn,w3Tn] = TDSetupD(xn,yn,zn,C,bx,by,bz,nu,p3,-e23);
end

% Calculate the "incomplete" displacement vector components in TDCS
if nnz(casepLog)~=0
    u(casepLog,1) = u1Tp+u2Tp+u3Tp;
    v(casepLog,1) = v1Tp+v2Tp+v3Tp;
    w(casepLog,1) = w1Tp+w2Tp+w3Tp;
end
if nnz(casenLog)~=0
    u(casenLog,1) = u1Tn+u2Tn+u3Tn;
    v(casenLog,1) = v1Tn+v2Tn+v3Tn;
    w(casenLog,1) = w1Tn+w2Tn+w3Tn;
end
if nnz(casezLog)~=0
    u(casezLog,1) = nan;
    v(casezLog,1) = nan;
    w(casezLog,1) = nan;
end

% Calculate the Burgers' function contribution corresponding to the TD
a = [-x p1(2)-y p1(3)-z];
b = [-x -y -z];
c = [-x p3(2)-y p3(3)-z];
na = sqrt(sum(a.^2,2));
nb = sqrt(sum(b.^2,2));
nc = sqrt(sum(c.^2,2));

FiN = (a(:,1).*(b(:,2).*c(:,3)-b(:,3).*c(:,2))-...
    a(:,2).*(b(:,1).*c(:,3)-b(:,3).*c(:,1))+...
    a(:,3).*(b(:,1).*c(:,2)-b(:,2).*c(:,1)));
FiD = (na.*nb.*nc+sum(a.*b,2).*nc+sum(a.*c,2).*nb+sum(b.*c,2).*na);
FiN(FiN==0) = -0; % Fix the "sign bit" of FiN for x = 0
Fi = -2*atan2(FiN,FiD)/4/pi;

% Calculate the complete displacement vector components in TDCS
u = bx.*Fi+u;
v = by.*Fi+v;
w = bz.*Fi+w;

% Transform the complete displacement vector components from TDCS into EFCS
[ue,un,uv] = CoordTrans(u,v,w,[Vnorm Vstrike Vdip]);

function [ue,un,uv]=TDdisp_HarFunc(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,nu)
% TDdisp_HarFunc calculates the harmonic function contribution to the
% displacements associated with a triangular dislocation in a half-space.
% The function cancels the surface normal tractions induced by the main and
% image dislocations.

bx = Ts; % Tensile-slip
by = Ss; % Strike-slip
bz = Ds; % Dip-slip

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

% Transform slip vector components from TDCS into EFCS
A = [Vnorm Vstrike Vdip];
[bX,bY,bZ] = CoordTrans(bx,by,bz,A);

% Calculate contribution of angular dislocation pair on each TD side
[u1,v1,w1] = AngSetupFSC(X,Y,Z,bX,bY,bZ,P1,P2,nu); % Side P1P2
[u2,v2,w2] = AngSetupFSC(X,Y,Z,bX,bY,bZ,P2,P3,nu); % Side P2P3
[u3,v3,w3] = AngSetupFSC(X,Y,Z,bX,bY,bZ,P3,P1,nu); % Side P3P1

% Calculate total harmonic function contribution to displacements
ue = u1+u2+u3;
un = v1+v2+v3;
uv = w1+w2+w3;

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

function [u,v,w]=TDSetupD(x,y,z,alpha,bx,by,bz,nu,TriVertex,SideVec)
% TDSetupD transforms coordinates of the calculation points as well as
% slip vector components from ADCS into TDCS. It then calculates the
% displacements in ADCS and transforms them into TDCS.

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

% Calculate displacements associated with an angular dislocation in ADCS
[u,v0,w0] = AngDisDisp(x,y1,z1,-pi+alpha,bx,by1,bz1,nu);

% Transform displacements from ADCS into TDCS
r3 = A'*[v0';w0'];
v = r3(1,:)';
w = r3(2,:)';

function [ue,un,uv]=AngSetupFSC(X,Y,Z,bX,bY,bZ,PA,PB,nu)
% AngSetupFSC calculates the Free Surface Correction to displacements
% associated with angular dislocation pair on each TD side.

% Calculate TD side vector and the angle of the angular dislocation pair
SideVec = PB-PA;
eZ = [0 0 1]';
beta = acos(-SideVec'*eZ/norm(SideVec));

if abs(beta)<eps || abs(pi-beta)<eps
    ue = zeros(length(X),1);
    un = zeros(length(X),1);
    uv = zeros(length(X),1);
else
    ey1 = [SideVec(1:2);0];
    ey1 = ey1/norm(ey1);
    ey3 = -eZ;
    ey2 = cross(ey3,ey1);
    A = [ey1,ey2,ey3]; % Transformation matrix

    % Transform coordinates from EFCS to the first ADCS
    [y1A,y2A,y3A] = CoordTrans(X-PA(1),Y-PA(2),Z-PA(3),A);
    % Transform coordinates from EFCS to the second ADCS
    [y1AB,y2AB,y3AB] = CoordTrans(SideVec(1),SideVec(2),SideVec(3),A);
    y1B = y1A-y1AB;
    y2B = y2A-y2AB;
    y3B = y3A-y3AB;

    % Transform slip vector components from EFCS to ADCS
    [b1,b2,b3] = CoordTrans(bX,bY,bZ,A);

    % Determine the best artefact-free configuration for the calculation
    % points near the free surface
    I = (beta*y1A)>=0;

    % Configuration I
    [v1A(I),v2A(I),v3A(I)] = AngDisDispFSC(y1A(I),y2A(I),y3A(I),...
        -pi+beta,b1,b2,b3,nu,-PA(3));
    [v1B(I),v2B(I),v3B(I)] = AngDisDispFSC(y1B(I),y2B(I),y3B(I),...
        -pi+beta,b1,b2,b3,nu,-PB(3));

    % Configuration II
    [v1A(~I),v2A(~I),v3A(~I)] = AngDisDispFSC(y1A(~I),y2A(~I),y3A(~I),...
        beta,b1,b2,b3,nu,-PA(3));
    [v1B(~I),v2B(~I),v3B(~I)] = AngDisDispFSC(y1B(~I),y2B(~I),y3B(~I),...
        beta,b1,b2,b3,nu,-PB(3));

    % Calculate total Free Surface Correction to displacements in ADCS
    v1 = v1B-v1A;
    v2 = v2B-v2A;
    v3 = v3B-v3A;

    % Transform total Free Surface Correction to displacements from ADCS
    % to EFCS
    [ue,un,uv] = CoordTrans(v1,v2,v3,A');
end

function [u,v,w]=AngDisDisp(x,y,z,alpha,bx,by,bz,nu)
% AngDisDisp calculates the "incomplete" displacements (without the
% Burgers' function contribution) associated with an angular dislocation in
% an elastic full-space.

cosA = cos(alpha);
sinA = sin(alpha);
eta = y*cosA-z*sinA;
zeta = y*sinA+z*cosA;
r = sqrt(x.^2+y.^2+z.^2);

% Avoid complex results for the logarithmic terms
zeta(zeta>r) = r(zeta>r);
z(z>r) = r(z>r);

ux = bx/8/pi/(1-nu)*(x.*y./r./(r-z)-x.*eta./r./(r-zeta));
vx = bx/8/pi/(1-nu)*(eta*sinA./(r-zeta)-y.*eta./r./(r-zeta)+...
    y.^2./r./(r-z)+(1-2*nu)*(cosA*log(r-zeta)-log(r-z)));
wx = bx/8/pi/(1-nu)*(eta*cosA./(r-zeta)-y./r-eta.*z./r./(r-zeta)-...
    (1-2*nu)*sinA*log(r-zeta));

uy = by/8/pi/(1-nu)*(x.^2*cosA./r./(r-zeta)-x.^2./r./(r-z)-...
    (1-2*nu)*(cosA*log(r-zeta)-log(r-z)));
vy = by*x/8/pi/(1-nu).*(y.*cosA./r./(r-zeta)-...
    sinA*cosA./(r-zeta)-y./r./(r-z));
wy = by*x/8/pi/(1-nu).*(z*cosA./r./(r-zeta)-cosA^2./(r-zeta)+1./r);

uz = bz*sinA/8/pi/(1-nu).*((1-2*nu)*log(r-zeta)-x.^2./r./(r-zeta));
vz = bz*x*sinA/8/pi/(1-nu).*(sinA./(r-zeta)-y./r./(r-zeta));
wz = bz*x*sinA/8/pi/(1-nu).*(cosA./(r-zeta)-z./r./(r-zeta));

u = ux+uy+uz;
v = vx+vy+vz;
w = wx+wy+wz;

function [v1,v2,v3] = AngDisDispFSC(y1,y2,y3,beta,b1,b2,b3,nu,a)
% AngDisDispFSC calculates the harmonic function contribution to the
% displacements associated with an angular dislocation in an elastic
% half-space.

sinB = sin(beta);
cosB = cos(beta);
cotB = cot(beta);
y3b = y3+2*a;
z1b = y1*cosB+y3b*sinB;
z3b = -y1*sinB+y3b*cosB;
r2b = y1.^2+y2.^2+y3b.^2;
rb = sqrt(r2b);

Fib = 2*atan(-y2./(-(rb+y3b)*cot(beta/2)+y1)); % The Burgers' function

v1cb1 = b1/4/pi/(1-nu)*(-2*(1-nu)*(1-2*nu)*Fib*cotB.^2+(1-2*nu)*y2./...
    (rb+y3b).*((1-2*nu-a./rb)*cotB-y1./(rb+y3b).*(nu+a./rb))+(1-2*nu).*...
    y2.*cosB*cotB./(rb+z3b).*(cosB+a./rb)+a*y2.*(y3b-a)*cotB./rb.^3+y2.*...
    (y3b-a)./(rb.*(rb+y3b)).*(-(1-2*nu)*cotB+y1./(rb+y3b).*(2*nu+a./rb)+...
    a*y1./rb.^2)+y2.*(y3b-a)./(rb.*(rb+z3b)).*(cosB./(rb+z3b).*((rb*...
    cosB+y3b).*((1-2*nu)*cosB-a./rb).*cotB+2*(1-nu)*(rb*sinB-y1)*cosB)-...
    a.*y3b*cosB*cotB./rb.^2));

v2cb1 = b1/4/pi/(1-nu)*((1-2*nu)*((2*(1-nu)*cotB^2-nu)*log(rb+y3b)-(2*...
    (1-nu)*cotB^2+1-2*nu)*cosB*log(rb+z3b))-(1-2*nu)./(rb+y3b).*(y1*...
    cotB.*(1-2*nu-a./rb)+nu*y3b-a+y2.^2./(rb+y3b).*(nu+a./rb))-(1-2*...
    nu).*z1b*cotB./(rb+z3b).*(cosB+a./rb)-a*y1.*(y3b-a)*cotB./rb.^3+...
    (y3b-a)./(rb+y3b).*(-2*nu+1./rb.*((1-2*nu).*y1*cotB-a)+y2.^2./(rb.*...
    (rb+y3b)).*(2*nu+a./rb)+a*y2.^2./rb.^3)+(y3b-a)./(rb+z3b).*(cosB^2-...
    1./rb.*((1-2*nu).*z1b*cotB+a*cosB)+a*y3b.*z1b*cotB./rb.^3-1./(rb.*...
    (rb+z3b)).*(y2.^2*cosB^2-a*z1b*cotB./rb.*(rb*cosB+y3b))));

v3cb1 = b1/4/pi/(1-nu)*(2*(1-nu)*(((1-2*nu)*Fib*cotB)+(y2./(rb+y3b).*(2*...
    nu+a./rb))-(y2*cosB./(rb+z3b).*(cosB+a./rb)))+y2.*(y3b-a)./rb.*(2*...
    nu./(rb+y3b)+a./rb.^2)+y2.*(y3b-a)*cosB./(rb.*(rb+z3b)).*(1-2*nu-...
    (rb*cosB+y3b)./(rb+z3b).*(cosB+a./rb)-a*y3b./rb.^2));

v1cb2 = b2/4/pi/(1-nu)*((1-2*nu)*((2*(1-nu)*cotB^2+nu)*log(rb+y3b)-(2*...
    (1-nu)*cotB^2+1)*cosB*log(rb+z3b))+(1-2*nu)./(rb+y3b).*(-(1-2*nu).*...
    y1*cotB+nu*y3b-a+a*y1*cotB./rb+y1.^2./(rb+y3b).*(nu+a./rb))-(1-2*...
    nu)*cotB./(rb+z3b).*(z1b*cosB-a*(rb*sinB-y1)./(rb*cosB))-a*y1.*...
    (y3b-a)*cotB./rb.^3+(y3b-a)./(rb+y3b).*(2*nu+1./rb.*((1-2*nu).*y1*...
    cotB+a)-y1.^2./(rb.*(rb+y3b)).*(2*nu+a./rb)-a*y1.^2./rb.^3)+(y3b-a)*...
    cotB./(rb+z3b).*(-cosB*sinB+a*y1.*y3b./(rb.^3*cosB)+(rb*sinB-y1)./...
    rb.*(2*(1-nu)*cosB-(rb*cosB+y3b)./(rb+z3b).*(1+a./(rb*cosB)))));

v2cb2 = b2/4/pi/(1-nu)*(2*(1-nu)*(1-2*nu)*Fib*cotB.^2+(1-2*nu)*y2./...
    (rb+y3b).*(-(1-2*nu-a./rb)*cotB+y1./(rb+y3b).*(nu+a./rb))-(1-2*nu)*...
    y2*cotB./(rb+z3b).*(1+a./(rb*cosB))-a*y2.*(y3b-a)*cotB./rb.^3+y2.*...
    (y3b-a)./(rb.*(rb+y3b)).*((1-2*nu)*cotB-2*nu*y1./(rb+y3b)-a*y1./rb.*...
    (1./rb+1./(rb+y3b)))+y2.*(y3b-a)*cotB./(rb.*(rb+z3b)).*(-2*(1-nu)*...
    cosB+(rb*cosB+y3b)./(rb+z3b).*(1+a./(rb*cosB))+a*y3b./(rb.^2*cosB)));

v3cb2 = b2/4/pi/(1-nu)*(-2*(1-nu)*(1-2*nu)*cotB*(log(rb+y3b)-cosB*...
    log(rb+z3b))-2*(1-nu)*y1./(rb+y3b).*(2*nu+a./rb)+2*(1-nu)*z1b./(rb+...
    z3b).*(cosB+a./rb)+(y3b-a)./rb.*((1-2*nu)*cotB-2*nu*y1./(rb+y3b)-a*...
    y1./rb.^2)-(y3b-a)./(rb+z3b).*(cosB*sinB+(rb*cosB+y3b)*cotB./rb.*...
    (2*(1-nu)*cosB-(rb*cosB+y3b)./(rb+z3b))+a./rb.*(sinB-y3b.*z1b./...
    rb.^2-z1b.*(rb*cosB+y3b)./(rb.*(rb+z3b)))));

v1cb3 = b3/4/pi/(1-nu)*((1-2*nu)*(y2./(rb+y3b).*(1+a./rb)-y2*cosB./(rb+...
    z3b).*(cosB+a./rb))-y2.*(y3b-a)./rb.*(a./rb.^2+1./(rb+y3b))+y2.*...
    (y3b-a)*cosB./(rb.*(rb+z3b)).*((rb*cosB+y3b)./(rb+z3b).*(cosB+a./...
    rb)+a.*y3b./rb.^2));

v2cb3 = b3/4/pi/(1-nu)*((1-2*nu)*(-sinB*log(rb+z3b)-y1./(rb+y3b).*(1+a./...
    rb)+z1b./(rb+z3b).*(cosB+a./rb))+y1.*(y3b-a)./rb.*(a./rb.^2+1./(rb+...
    y3b))-(y3b-a)./(rb+z3b).*(sinB*(cosB-a./rb)+z1b./rb.*(1+a.*y3b./...
    rb.^2)-1./(rb.*(rb+z3b)).*(y2.^2*cosB*sinB-a*z1b./rb.*(rb*cosB+y3b))));

v3cb3 = b3/4/pi/(1-nu)*(2*(1-nu)*Fib+2*(1-nu)*(y2*sinB./(rb+z3b).*(cosB+...
    a./rb))+y2.*(y3b-a)*sinB./(rb.*(rb+z3b)).*(1+(rb*cosB+y3b)./(rb+...
    z3b).*(cosB+a./rb)+a.*y3b./rb.^2));

v1 = v1cb1+v1cb2+v1cb3;
v2 = v2cb1+v2cb2+v2cb3;
v3 = v3cb1+v3cb2+v3cb3;
