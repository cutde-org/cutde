<%!
def comp(d):
    return ["x", "y", "z", "a", "b", "c"][d]
%>

<%def name="binop(dim, name, op, b_scalar = False)">
<%
b_type = 'Real' if b_scalar else 'Real' + str(dim)
%>
WITHIN_KERNEL Real${dim} ${name}${dim}(Real${dim} a, ${b_type} b) {
    Real${dim} out;
    % for d in range(dim):
        <%
            if b_scalar:
                b_str = 'b';
            else:
                b_str = 'b.' + comp(d);
        %>
        out.${comp(d)} = a.${comp(d)} ${op} ${b_str};
    % endfor
    return out;
}
</%def>

<%def name="LOCAL_BARRIER()">
% if backend == 'cuda':
__syncthreads();
% elif backend == 'opencl':
barrier(CLK_LOCAL_MEM_FENCE)
% endif
</%def>

<%def name="defs(preamble, float_type)">

#ifndef CUTDE_COMMON
#define CUTDE_COMMON
${preamble}

<%
import numpy as np
%>
#define Real ${float_type}
% if float_type == 'double':
#define EPS ${np.finfo(np.float64).eps}
% else:
#define EPS ${np.finfo(np.float32).eps}
% endif

#ifndef M_PI
  #define M_PI   3.14159265358979323846264338327950288
#endif

typedef struct Real2 {
    Real x;
    Real y;
} Real2;

typedef struct Real3 {
    Real x;
    Real y;
    Real z;
} Real3;

typedef struct Real6 {
    Real x;
    Real y;
    Real z;
    Real a;
    Real b;
    Real c;
} Real6;

WITHIN_KERNEL void print(Real x) {
    printf("%f \n", x);
}

% for dim in [2,3,6]:
    ${binop(dim, 'add','+')}
    ${binop(dim, 'sub','-')}
    ${binop(dim, 'mul','*')}
    ${binop(dim, 'div','/')}

    ${binop(dim, 'add_scalar','+',b_scalar = True)}
    ${binop(dim, 'sub_scalar','-',b_scalar = True)}
    ${binop(dim, 'mul_scalar','*',b_scalar = True)}
    ${binop(dim, 'div_scalar','/',b_scalar = True)}

    WITHIN_KERNEL void print${dim}(Real${dim} x) {
        <%
        format_str = "%f " * dim
        var_str = ','.join(['x.' + comp(d) for d in range(dim)])
        %>
        printf("${format_str} \n", ${var_str});
    }

    WITHIN_KERNEL Real sum${dim}(Real${dim} x) {
        Real out = 0.0;
        % for d in range(dim):
            out += x.${comp(d)};
        % endfor
        return out;
    }

    WITHIN_KERNEL Real dot${dim}(Real${dim} x, Real${dim} y) {
        return sum${dim}(mul${dim}(x, y));
    }

    WITHIN_KERNEL Real length${dim}(Real${dim} x) {
        return sqrt(dot${dim}(x, x));
    }

    WITHIN_KERNEL Real${dim} negate${dim}(Real${dim} x) {
        return mul_scalar${dim}(x, -1);
    }

    WITHIN_KERNEL Real${dim} normalize${dim}(Real${dim} x) {
        return div_scalar${dim}(x, length${dim}(x));
    }

    WITHIN_KERNEL Real${dim} make${dim}(
    % for d in range(dim):
        Real ${comp(d)}
        % if d != dim - 1:
            ,
        % endif
    % endfor
    ) {
        Real${dim} out;
        % for d in range(dim):
            out.${comp(d)} = ${comp(d)};
        % endfor
        return out;
    }
% endfor


WITHIN_KERNEL Real2 transform2(Real2 a, Real2 b, Real2 v) {
    Real2 out;
    out.x = dot2(a,v);
    out.y = dot2(b,v);
    return out;
}

WITHIN_KERNEL Real2 inv_transform2(Real2 a, Real2 b, Real2 v) {
    Real2 out;
    out.x = a.x * v.x + b.x * v.y;
    out.y = a.y * v.x + b.y * v.y;
    return out;
}

WITHIN_KERNEL Real3 transform3(Real3 a, Real3 b, Real3 c, Real3 v) {
    Real3 out;
    out.x = dot3(a,v);
    out.y = dot3(b,v);
    out.z = dot3(c,v);
    return out;
}

WITHIN_KERNEL Real3 inv_transform3(Real3 a, Real3 b, Real3 c, Real3 v) {
    Real3 out;
    out.x = a.x * v.x + b.x * v.y + c.x * v.z;
    out.y = a.y * v.x + b.y * v.y + c.y * v.z;
    out.z = a.z * v.x + b.z * v.y + c.z * v.z;
    return out;
}

WITHIN_KERNEL Real3 cross3(Real3 x, Real3 y) {
    Real3 out;
    out.x = x.y * y.z - x.z * y.y;
    out.y = x.z * y.x - x.x * y.z;
    out.z = x.x * y.y - x.y * y.x;
    return out;
}

WITHIN_KERNEL Real triangle_area(Real3 tri0, Real3 tri1, Real3 tri2) {
    Real3 edge1 = sub3(tri1, tri0);
    Real3 edge2 = sub3(tri2, tri0);
    Real3 cross_prod = cross3(edge1, edge2);
    return 0.5 * length3(cross_prod);
}

WITHIN_KERNEL int is_degenerate_triangle(Real3 tri0, Real3 tri1, Real3 tri2) {
    Real area = triangle_area(tri0, tri1, tri2);
    return area == 0;
}

WITHIN_KERNEL Real6 tensor_transform3(Real3 a, Real3 b, Real3 c, Real6 tensor) {
    Real A[9];
    A[0] = a.x; A[1] = a.y; A[2] = a.z;
    A[3] = b.x; A[4] = b.y; A[5] = b.z;
    A[6] = c.x; A[7] = c.y; A[8] = c.z;
    Real6 out;
    out.x = A[0]*A[0]*tensor.x+2*A[0]*A[3]*tensor.a+2*A[0]*A[6]*tensor.b+
        2*A[3]*A[6]*tensor.c+ A[3]*A[3]*tensor.y+A[6]*A[6]*tensor.z;
    out.y = A[1]*A[1]*tensor.x+2*A[1]*A[4]*tensor.a+2*A[1]*A[7]*tensor.b+
        2*A[4]*A[7]*tensor.c+A[4]*A[4]*tensor.y+A[7]*A[7]*tensor.z;
    out.z = A[2]*A[2]*tensor.x+2*A[2]*A[5]*tensor.a+2*A[2]*A[8]*tensor.b+
        2*A[5]*A[8]*tensor.c+A[5]*A[5]*tensor.y+A[8]*A[8]*tensor.z;
    out.a = A[0]*A[1]*tensor.x+(A[0]*A[4]+A[1]*A[3])*tensor.a+(A[0]*A[7]+
        A[1]*A[6])*tensor.b+(A[7]*A[3]+A[6]*A[4])*tensor.c+A[4]*A[3]*tensor.y+
        A[6]*A[7]*tensor.z;
    out.b = A[0]*A[2]*tensor.x+(A[0]*A[5]+A[2]*A[3])*tensor.a+(A[0]*A[8]+
        A[2]*A[6])*tensor.b+(A[8]*A[3]+A[6]*A[5])*tensor.c+A[5]*A[3]*tensor.y+
        A[6]*A[8]*tensor.z;
    out.c = A[1]*A[2]*tensor.x+(A[2]*A[4]+A[1]*A[5])*tensor.a+(A[2]*A[7]+
        A[1]*A[8])*tensor.b+(A[7]*A[5]+A[8]*A[4])*tensor.c+A[4]*A[5]*tensor.y+
        A[7]*A[8]*tensor.z;
    return out;
}

WITHIN_KERNEL int trimodefinder(Real3 obs, Real3 tri0, Real3 tri1, Real3 tri2) {
    // trimodefinder calculates the normalized barycentric coordinates of
    // the points with respect to the TD vertices and specifies the appropriate
    // artefact-free configuration of the angular dislocations for the
    // calculations. The input matrices x, y and z share the same size and
    // correspond to the y, z and x coordinates in the TDCS, respectively. p1,
    // p2 and p3 are two-component matrices representing the y and z coordinates
    // of the TD vertices in the TDCS, respectively.
    // The components of the output (trimode) corresponding to each calculation
    // points, are 1 for the first configuration, -1 for the second
    // configuration and 0 for the calculation point that lie on the TD sides.

    Real a = ((tri1.z-tri2.z)*(obs.y-tri2.y)+(tri2.y-tri1.y)*(obs.z-tri2.z))/
        ((tri1.z-tri2.z)*(tri0.y-tri2.y)+(tri2.y-tri1.y)*(tri0.z-tri2.z));
    Real b = ((tri2.z-tri0.z)*(obs.y-tri2.y)+(tri0.y-tri2.y)*(obs.z-tri2.z))/
        ((tri1.z-tri2.z)*(tri0.y-tri2.y)+(tri2.y-tri1.y)*(tri0.z-tri2.z));
    Real c = 1-a-b;

    int result = 1;
    if ((a<=0 && b>c && c>a) ||
            (b<=0 && c>a && a>b) ||
            (c<=0 && a>b && b>c)) {
        result = -1;
    }

    if ((a==0 && b>=0 && c>=0) ||
            (a>=0 && b==0 && c>=0) ||
            (a>=0 && b>=0 && c==0)) {
        result = 0;
    }
    if (result == 0 && obs.x != 0) {
        result = 1;
    }

    return result;
}

WITHIN_KERNEL Real3 AngDisDisp(Real x, Real y, Real z, Real alpha, Real bx, Real by, Real bz, Real nu) {

    Real cosA = cos(alpha);
    Real sinA = sin(alpha);
    Real eta = y*cosA-z*sinA;
    Real zeta = y*sinA+z*cosA;
    Real r = sqrt(x * x + y * y + z * z);

    Real ux = bx/8/M_PI/(1-nu)*(x*y/r/(r-z)-x*eta/r/(r-zeta));
    Real vx = bx/8/M_PI/(1-nu)*(eta*sinA/(r-zeta)-y*eta/r/(r-zeta)+\
        y*y/r/(r-z)+(1-2*nu)*(cosA*log(r-zeta)-log(r-z)));
    Real wx = bx/8/M_PI/(1-nu)*(eta*cosA/(r-zeta)-y/r-eta*z/r/(r-zeta)-\
        (1-2*nu)*sinA*log(r-zeta));

    Real uy = by/8/M_PI/(1-nu)*(x*x*cosA/r/(r-zeta)-x*x/r/(r-z)-\
         (1-2*nu)*(cosA*log(r-zeta)-log(r-z)));
    Real vy = by*x/8/M_PI/(1-nu)*(y*cosA/r/(r-zeta)-sinA*cosA/(r-zeta)-y/r/(r-z));
    Real wy = by*x/8/M_PI/(1-nu)*(z*cosA/r/(r-zeta)-cosA*cosA/(r-zeta)+1/r);

    Real uz = bz*sinA/8/M_PI/(1-nu)*((1-2*nu)*log(r-zeta)-x*x/r/(r-zeta));
    Real vz = bz*x*sinA/8/M_PI/(1-nu)*(sinA/(r-zeta)-y/r/(r-zeta));
    Real wz = bz*x*sinA/8/M_PI/(1-nu)*(cosA/(r-zeta)-z/r/(r-zeta));
    return make3(ux+uy+uz, vx+vy+vz, wx+wy+wz);
}

WITHIN_KERNEL Real3 TDSetupD(Real3 obs, Real alpha, Real3 slip, Real nu, Real3 TriVertex, Real3 SideVec) {
    // TDSetupD transforms coordinates of the calculation points as well as
    // slip vector components from ADCS into TDCS. It then calculates the
    // displacements in ADCS and transforms them into TDCS.

    Real2 A1 = make2(SideVec.z, -SideVec.y);
    Real2 A2 = make2(SideVec.y, SideVec.z);
    Real2 r1 = transform2(A1, A2, make2(obs.y - TriVertex.y, obs.z - TriVertex.z));
    Real y1 = r1.x;
    Real z1 = r1.y;

    Real2 r2 = transform2(A1, A2, make2(slip.y, slip.z));
    Real by1 = r2.x;
    Real bz1 = r2.y;

    Real3 uvw = AngDisDisp(obs.x, y1, z1, -M_PI + alpha, slip.x, by1, bz1, nu);

    Real2 r3 = inv_transform2(A1, A2, make2(uvw.y, uvw.z));
    Real v = r3.x;
    Real w = r3.y;
    return make3(uvw.x, v, w);
}

WITHIN_KERNEL Real6 AngDisStrain(Real x, Real y, Real z, Real alpha, Real bx, Real by, Real bz, Real nu) {
    // AngDisStrain calculates the strains associated with an angular
    // dislocation in an elastic full-space.

    Real cosA = cos(alpha);
    Real sinA = sin(alpha);
    Real eta = y*cosA-z*sinA;
    Real zeta = y*sinA+z*cosA;

    Real x2 = x*x;
    Real y2 = y*y;
    Real z2 = z*z;
    Real r2 = x2+y2+z2;
    Real r = sqrt(r2);
    Real r3 = r*r2;
    Real rz = r*(r-z);
    Real rmz = (r-z);
    Real r2z2 = r2*rmz*rmz;
    Real r3z = r3*rmz;

    Real W = zeta-r;
    Real W2 = W*W;
    Real Wr = W*r;
    Real W2r = W2*r;
    Real Wr3 = W*r3;
    Real W2r2 = W2*r2;

    Real C = (r*cosA-z)/Wr;
    Real S = (r*sinA-y)/Wr;

    // Partial derivatives of the Burgers' function
    Real rFi_rx = (eta/r/(r-zeta)-y/r/(r-z))/4/M_PI;
    Real rFi_ry = (x/r/(r-z)-cosA*x/r/(r-zeta))/4/M_PI;
    Real rFi_rz = (sinA*x/r/(r-zeta))/4/M_PI;

    Real6 out;
    out.x = bx*(rFi_rx)+
        bx/8/M_PI/(1-nu)*(eta/Wr+eta*x2/W2r2-eta*x2/Wr3+y/rz-
        x2*y/r2z2-x2*y/r3z)-
        by*x/8/M_PI/(1-nu)*(((2*nu+1)/Wr+x2/W2r2-x2/Wr3)*cosA+
        (2*nu+1)/rz-x2/r2z2-x2/r3z)+
        bz*x*sinA/8/M_PI/(1-nu)*((2*nu+1)/Wr+x2/W2r2-x2/Wr3);

    out.y = by*(rFi_ry)+
        bx/8/M_PI/(1-nu)*((1/Wr+S*S-y2/Wr3)*eta+(2*nu+1)*y/rz-y*y*y/r2z2-
        y*y*y/r3z-2*nu*cosA*S)-
        by*x/8/M_PI/(1-nu)*(1/rz-y2/r2z2-y2/r3z+
        (1/Wr+S*S-y2/Wr3)*cosA)+
        bz*x*sinA/8/M_PI/(1-nu)*(1/Wr+S*S-y2/Wr3);

    out.z = bz*(rFi_rz)+
        bx/8/M_PI/(1-nu)*(eta/W/r+eta*C*C-eta*z2/Wr3+y*z/r3+
        2*nu*sinA*C)-
        by*x/8/M_PI/(1-nu)*((1/Wr+C*C-z2/Wr3)*cosA+z/r3)+
        bz*x*sinA/8/M_PI/(1-nu)*(1/Wr+C*C-z2/Wr3);

    out.a = bx*(rFi_ry)/2+by*(rFi_rx)/2-
        bx/8/M_PI/(1-nu)*(x*y2/r2z2-nu*x/rz+x*y2/r3z-nu*x*cosA/Wr+
        eta*x*S/Wr+eta*x*y/Wr3)+
        by/8/M_PI/(1-nu)*(x2*y/r2z2-nu*y/rz+x2*y/r3z+nu*cosA*S+
        x2*y*cosA/Wr3+x2*cosA*S/Wr)-
        bz*sinA/8/M_PI/(1-nu)*(nu*S+x2*S/Wr+x2*y/Wr3);

    out.b = bx*(rFi_rz)/2+bz*(rFi_rx)/2-
        bx/8/M_PI/(1-nu)*(-x*y/r3+nu*x*sinA/Wr+eta*x*C/Wr+
        eta*x*z/Wr3)+
        by/8/M_PI/(1-nu)*(-x2/r3+nu/r+nu*cosA*C+x2*z*cosA/Wr3+
        x2*cosA*C/Wr)-
        bz*sinA/8/M_PI/(1-nu)*(nu*C+x2*C/Wr+x2*z/Wr3);

    out.c = by*(rFi_rz)/2+bz*(rFi_ry)/2+
        bx/8/M_PI/(1-nu)*(y2/r3-nu/r-nu*cosA*C+nu*sinA*S+eta*sinA*cosA/W2-
        eta*(y*cosA+z*sinA)/W2r+eta*y*z/W2r2-eta*y*z/Wr3)-
        by*x/8/M_PI/(1-nu)*(y/r3+sinA*cosA*cosA/W2-cosA*(y*cosA+z*sinA)/
        W2r+y*z*cosA/W2r2-y*z*cosA/Wr3)-
        bz*x*sinA/8/M_PI/(1-nu)*(y*z/Wr3-sinA*cosA/W2+(y*cosA+z*sinA)/
        W2r-y*z/W2r2);
    return out;
}


WITHIN_KERNEL Real6 TDSetupS(Real3 obs, Real alpha, Real3 slip, Real nu,
    Real3 TriVertex, Real3 SideVec)
{
    // TDSetupS transforms coordinates of the calculation points as well as
    // slip vector components from ADCS into TDCS. It then calculates the
    // strains in ADCS and transforms them into TDCS.

    // Transformation matrix
    Real2 A1 = make2(SideVec.z, -SideVec.y);
    Real2 A2 = make2(SideVec.y, SideVec.z);

    // Transform coordinates of the calculation points from TDCS into ADCS
    Real2 r1 = transform2(A1, A2, make2(obs.y - TriVertex.y, obs.z - TriVertex.z));
    Real y1 = r1.x;
    Real z1 = r1.y;

    // Transform the in-plane slip vector components from TDCS into ADCS
    Real2 r2 = transform2(A1, A2, make2(slip.y, slip.z));
    Real by1 = r2.x;
    Real bz1 = r2.y;

    // Calculate strains associated with an angular dislocation in ADCS
    Real6 out_adcs = AngDisStrain(obs.x,y1,z1,-M_PI+alpha,slip.x,by1,bz1,nu);

    // Transform strains from ADCS into TDCS
    Real3 B0 = make3(1.0, 0.0, 0.0);
    Real3 B1 = make3(0.0, A1.x, A1.y);
    Real3 B2 = make3(0.0, A2.x, A2.y);
    return tensor_transform3(B0, B1, B2, out_adcs);
}

WITHIN_KERNEL Real3 AngDisDispFSC(Real y1, Real y2, Real y3, Real beta,
                                  Real b1, Real b2, Real b3, Real nu, Real a) {

    Real sinB = sin(beta);
    Real cosB = cos(beta);
    if (cosB == 0.) {
        cosB = EPS;
    }
    Real cotB = cosB / sinB;
    Real y3b = y3+2*a;
    Real z1b = y1*cosB+y3b*sinB;
    Real z3b = -y1*sinB+y3b*cosB;
    Real r2b = y1*y1+y2*y2+y3b*y3b;
    Real rb = sqrt(r2b);

    Real Fib = 2*atan(-y2/(-(rb+y3b)*(1.0 / tan(beta/2))+y1)); // The Burgers' function

    Real v1cb1 = b1/4/M_PI/(1-nu)*(-2*(1-nu)*(1-2*nu)*Fib*(cotB*cotB)+(1-2*nu)*y2/
        (rb+y3b)*((1-2*nu-a/rb)*cotB-y1/(rb+y3b)*(nu+a/rb))+(1-2*nu)*
        y2*cosB*cotB/(rb+z3b)*(cosB+a/rb)+a*y2*(y3b-a)*cotB/(rb*rb*rb)+y2*
        (y3b-a)/(rb*(rb+y3b))*(-(1-2*nu)*cotB+y1/(rb+y3b)*(2*nu+a/rb)+
        a*y1/(rb*rb))+y2*(y3b-a)/(rb*(rb+z3b))*(cosB/(rb+z3b)*((rb*
        cosB+y3b)*((1-2*nu)*cosB-a/rb)*cotB+2*(1-nu)*(rb*sinB-y1)*cosB)-
        a*y3b*cosB*cotB/(rb*rb)));

    Real v2cb1 = b1/4/M_PI/(1-nu)*((1-2*nu)*((2*(1-nu)*(cotB*cotB)-nu)*log(rb+y3b)-(2*
        (1-nu)*(cotB*cotB)+1-2*nu)*cosB*log(rb+z3b))-(1-2*nu)/(rb+y3b)*(y1*
        cotB*(1-2*nu-a/rb)+nu*y3b-a+(y2*y2)/(rb+y3b)*(nu+a/rb))-(1-2*
        nu)*z1b*cotB/(rb+z3b)*(cosB+a/rb)-a*y1*(y3b-a)*cotB/(rb*rb*rb)+
        (y3b-a)/(rb+y3b)*(-2*nu+1/rb*((1-2*nu)*y1*cotB-a)+(y2*y2)/(rb*
        (rb+y3b))*(2*nu+a/rb)+a*(y2*y2)/(rb*rb*rb))+(y3b-a)/(rb+z3b)*((cosB*cosB)-
        1/rb*((1-2*nu)*z1b*cotB+a*cosB)+a*y3b*z1b*cotB/(rb*rb*rb)-1/(rb*
        (rb+z3b))*((y2*y2)*(cosB*cosB)-a*z1b*cotB/rb*(rb*cosB+y3b))));

    Real v3cb1 = b1/4/M_PI/(1-nu)*(2*(1-nu)*(((1-2*nu)*Fib*cotB)+(y2/(rb+y3b)*(2*
        nu+a/rb))-(y2*cosB/(rb+z3b)*(cosB+a/rb)))+y2*(y3b-a)/rb*(2*
        nu/(rb+y3b)+a/(rb*rb))+y2*(y3b-a)*cosB/(rb*(rb+z3b))*(1-2*nu-
        (rb*cosB+y3b)/(rb+z3b)*(cosB+a/rb)-a*y3b/(rb*rb)));

    Real v1cb2 = b2/4/M_PI/(1-nu)*((1-2*nu)*((2*(1-nu)*(cotB*cotB)+nu)*log(rb+y3b)-(2*
        (1-nu)*(cotB*cotB)+1)*cosB*log(rb+z3b))+(1-2*nu)/(rb+y3b)*(-(1-2*nu)*
        y1*cotB+nu*y3b-a+a*y1*cotB/rb+(y1*y1)/(rb+y3b)*(nu+a/rb))-(1-2*
        nu)*cotB/(rb+z3b)*(z1b*cosB-a*(rb*sinB-y1)/(rb*cosB))-a*y1*
        (y3b-a)*cotB/(rb*rb*rb)+(y3b-a)/(rb+y3b)*(2*nu+1/rb*((1-2*nu)*y1*
        cotB+a)-(y1*y1)/(rb*(rb+y3b))*(2*nu+a/rb)-a*(y1*y1)/(rb*rb*rb))+(y3b-a)*
        cotB/(rb+z3b)*(-cosB*sinB+a*y1*y3b/((rb*rb*rb)*cosB)+(rb*sinB-y1)/
        rb*(2*(1-nu)*cosB-(rb*cosB+y3b)/(rb+z3b)*(1+a/(rb*cosB)))));

    Real v2cb2 = b2/4/M_PI/(1-nu)*(2*(1-nu)*(1-2*nu)*Fib*(cotB*cotB)+(1-2*nu)*y2/
        (rb+y3b)*(-(1-2*nu-a/rb)*cotB+y1/(rb+y3b)*(nu+a/rb))-(1-2*nu)*
        y2*cotB/(rb+z3b)*(1+a/(rb*cosB))-a*y2*(y3b-a)*cotB/(rb*rb*rb)+y2*
        (y3b-a)/(rb*(rb+y3b))*((1-2*nu)*cotB-2*nu*y1/(rb+y3b)-a*y1/rb*
        (1/rb+1/(rb+y3b)))+y2*(y3b-a)*cotB/(rb*(rb+z3b))*(-2*(1-nu)*
        cosB+(rb*cosB+y3b)/(rb+z3b)*(1+a/(rb*cosB))+a*y3b/((rb*rb)*cosB)));

    Real v3cb2 = b2/4/M_PI/(1-nu)*(-2*(1-nu)*(1-2*nu)*cotB*(log(rb+y3b)-cosB*
        log(rb+z3b))-2*(1-nu)*y1/(rb+y3b)*(2*nu+a/rb)+2*(1-nu)*z1b/(rb+
        z3b)*(cosB+a/rb)+(y3b-a)/rb*((1-2*nu)*cotB-2*nu*y1/(rb+y3b)-a*
        y1/(rb*rb))-(y3b-a)/(rb+z3b)*(cosB*sinB+(rb*cosB+y3b)*cotB/rb*
        (2*(1-nu)*cosB-(rb*cosB+y3b)/(rb+z3b))+a/rb*(sinB-y3b*z1b/
        (rb*rb)-z1b*(rb*cosB+y3b)/(rb*(rb+z3b)))));

    Real v1cb3 = b3/4/M_PI/(1-nu)*((1-2*nu)*(y2/(rb+y3b)*(1+a/rb)-y2*cosB/(rb+
        z3b)*(cosB+a/rb))-y2*(y3b-a)/rb*(a/(rb*rb)+1/(rb+y3b))+y2*
        (y3b-a)*cosB/(rb*(rb+z3b))*((rb*cosB+y3b)/(rb+z3b)*(cosB+a/
        rb)+a*y3b/(rb*rb)));

    Real v2cb3 = b3/4/M_PI/(1-nu)*((1-2*nu)*(-sinB*log(rb+z3b)-y1/(rb+y3b)*(1+a/
        rb)+z1b/(rb+z3b)*(cosB+a/rb))+y1*(y3b-a)/rb*(a/(rb*rb)+1/(rb+
        y3b))-(y3b-a)/(rb+z3b)*(sinB*(cosB-a/rb)+z1b/rb*(1+a*y3b/
        (rb*rb))-1/(rb*(rb+z3b))*((y2*y2)*cosB*sinB-a*z1b/rb*(rb*cosB+y3b))));

    Real v3cb3 = b3/4/M_PI/(1-nu)*(2*(1-nu)*Fib+2*(1-nu)*(y2*sinB/(rb+z3b)*(cosB+
        a/rb))+y2*(y3b-a)*sinB/(rb*(rb+z3b))*(1+(rb*cosB+y3b)/(rb+
        z3b)*(cosB+a/rb)+a*y3b/(rb*rb)));

    return make3(
        v1cb1+v1cb2+v1cb3,
        v2cb1+v2cb2+v2cb3,
        v3cb1+v3cb2+v3cb3
    );
}

WITHIN_KERNEL Real6 AngDisStrainFSC(Real y1, Real y2, Real y3, Real beta,
                                    Real b1, Real b2, Real b3, Real nu, Real a) {

    Real sinB = sin(beta);
    Real cosB = cos(beta);
    if (cosB == 0.) {
        cosB = EPS;
    }
    Real cotB = cosB / sinB;
    Real y3b = y3+2*a;
    Real z1b = y1*cosB+y3b*sinB;
    Real z3b = -y1*sinB+y3b*cosB;
    Real rb2 = y1*y1+y2*y2+y3b*y3b;
    Real rb = sqrt(rb2);

    Real W1 = rb*cosB+y3b;
    Real W2 = cosB+a/rb;
    Real W3 = cosB+y3b/rb;
    Real W4 = nu+a/rb;
    Real W5 = 2*nu+a/rb;
    Real W6 = rb+y3b;
    Real W7 = rb+z3b;
    Real W8 = y3+a;
    Real W9 = 1+a/rb/cosB;

    Real N1 = 1-2*nu;

    Real rFib_ry2 = z1b/rb/(rb+z3b)-y1/rb/(rb+y3b);
    Real rFib_ry1 = y2/rb/(rb+y3b)-cosB*y2/rb/(rb+z3b);
    Real rFib_ry3 = -sinB*y2/rb/(rb+z3b);

    Real6 out;
    out.x = b1*(1.0/4.0*((-2+2*nu)*N1*rFib_ry1*(cotB*cotB)-N1*y2/(W6*W6)*((1-W5)*cotB- y1/W6*W4)/rb*y1+N1*y2/W6*(a/(rb*rb*rb)*y1*cotB-1/W6*W4+(y1*y1)/ (W6*W6)*W4/rb+(y1*y1)/W6*a/(rb*rb*rb))-N1*y2*cosB*cotB/(W7*W7)*W2*(y1/ rb-sinB)-N1*y2*cosB*cotB/W7*a/(rb*rb*rb)*y1-3*a*y2*W8*cotB/(rb*rb2*rb2)* y1-y2*W8/(rb*rb*rb)/W6*(-N1*cotB+y1/W6*W5+a*y1/rb2)*y1-y2*W8/ rb2/(W6*W6)*(-N1*cotB+y1/W6*W5+a*y1/rb2)*y1+y2*W8/rb/W6* (1/W6*W5-(y1*y1)/(W6*W6)*W5/rb-(y1*y1)/W6*a/(rb*rb*rb)+a/rb2-2*a*(y1*y1) /(rb2*rb2))-y2*W8/(rb*rb*rb)/W7*(cosB/W7*(W1*(N1*cosB-a/rb)*cotB+ (2-2*nu)*(rb*sinB-y1)*cosB)-a*y3b*cosB*cotB/rb2)*y1-y2*W8/rb/ (W7*W7)*(cosB/W7*(W1*(N1*cosB-a/rb)*cotB+(2-2*nu)*(rb*sinB-y1)* cosB)-a*y3b*cosB*cotB/rb2)*(y1/rb-sinB)+y2*W8/rb/W7*(-cosB/ (W7*W7)*(W1*(N1*cosB-a/rb)*cotB+(2-2*nu)*(rb*sinB-y1)*cosB)*(y1/ rb-sinB)+cosB/W7*(1/rb*cosB*y1*(N1*cosB-a/rb)*cotB+W1*a/(rb*rb2) *y1*cotB+(2-2*nu)*(1/rb*sinB*y1-1)*cosB)+2*a*y3b*cosB*cotB/ (rb2*rb2)*y1))/M_PI/(1-nu))+ b2*(1.0/4.0*(N1*(((2-2*nu)*(cotB*cotB)+nu)/rb*y1/W6-((2-2*nu)*(cotB*cotB)+1)* cosB*(y1/rb-sinB)/W7)-N1/(W6*W6)*(-N1*y1*cotB+nu*y3b-a+a*y1* cotB/rb+(y1*y1)/W6*W4)/rb*y1+N1/W6*(-N1*cotB+a*cotB/rb-a* (y1*y1)*cotB/(rb*rb*rb)+2*y1/W6*W4-(y1*y1*y1)/(W6*W6)*W4/rb-(y1*y1*y1)/W6*a/ (rb*rb*rb))+N1*cotB/(W7*W7)*(z1b*cosB-a*(rb*sinB-y1)/rb/cosB)*(y1/ rb-sinB)-N1*cotB/W7*((cosB*cosB)-a*(1/rb*sinB*y1-1)/rb/cosB+a* (rb*sinB-y1)/(rb*rb*rb)/cosB*y1)-a*W8*cotB/(rb*rb*rb)+3*a*(y1*y1)*W8* cotB/(rb*rb2*rb2)-W8/(W6*W6)*(2*nu+1/rb*(N1*y1*cotB+a)-(y1*y1)/rb/W6* W5-a*(y1*y1)/(rb*rb*rb))/rb*y1+W8/W6*(-1/(rb*rb*rb)*(N1*y1*cotB+a)*y1+ 1/rb*N1*cotB-2*y1/rb/W6*W5+(y1*y1*y1)/(rb*rb*rb)/W6*W5+(y1*y1*y1)/rb2/ (W6*W6)*W5+(y1*y1*y1)/(rb2*rb2)/W6*a-2*a/(rb*rb*rb)*y1+3*a*(y1*y1*y1)/(rb*rb2*rb2))-W8* cotB/(W7*W7)*(-cosB*sinB+a*y1*y3b/(rb*rb*rb)/cosB+(rb*sinB-y1)/rb* ((2-2*nu)*cosB-W1/W7*W9))*(y1/rb-sinB)+W8*cotB/W7*(a*y3b/ (rb*rb*rb)/cosB-3*a*(y1*y1)*y3b/(rb*rb2*rb2)/cosB+(1/rb*sinB*y1-1)/rb* ((2-2*nu)*cosB-W1/W7*W9)-(rb*sinB-y1)/(rb*rb*rb)*((2-2*nu)*cosB-W1/ W7*W9)*y1+(rb*sinB-y1)/rb*(-1/rb*cosB*y1/W7*W9+W1/(W7*W7)* W9*(y1/rb-sinB)+W1/W7*a/(rb*rb*rb)/cosB*y1)))/M_PI/(1-nu))+ b3*(1.0/4.0*(N1*(-y2/(W6*W6)*(1+a/rb)/rb*y1-y2/W6*a/(rb*rb*rb)*y1+y2* cosB/(W7*W7)*W2*(y1/rb-sinB)+y2*cosB/W7*a/(rb*rb*rb)*y1)+y2*W8/ (rb*rb*rb)*(a/rb2+1/W6)*y1-y2*W8/rb*(-2*a/(rb2*rb2)*y1-1/(W6*W6)/ rb*y1)-y2*W8*cosB/(rb*rb*rb)/W7*(W1/W7*W2+a*y3b/rb2)*y1-y2*W8* cosB/rb/(W7*W7)*(W1/W7*W2+a*y3b/rb2)*(y1/rb-sinB)+y2*W8* cosB/rb/W7*(1/rb*cosB*y1/W7*W2-W1/(W7*W7)*W2*(y1/rb-sinB)- W1/W7*a/(rb*rb*rb)*y1-2*a*y3b/(rb2*rb2)*y1))/M_PI/(1-nu));

    out.y = b1*(1.0/4.0*(N1*(((2-2*nu)*(cotB*cotB)-nu)/rb*y2/W6-((2-2*nu)*(cotB*cotB)+1- 2*nu)*cosB/rb*y2/W7)+N1/(W6*W6)*(y1*cotB*(1-W5)+nu*y3b-a+(y2*y2) /W6*W4)/rb*y2-N1/W6*(a*y1*cotB/(rb*rb*rb)*y2+2*y2/W6*W4-(y2*y2*y2) /(W6*W6)*W4/rb-(y2*y2*y2)/W6*a/(rb*rb*rb))+N1*z1b*cotB/(W7*W7)*W2/rb* y2+N1*z1b*cotB/W7*a/(rb*rb*rb)*y2+3*a*y2*W8*cotB/(rb*rb2*rb2)*y1-W8/ (W6*W6)*(-2*nu+1/rb*(N1*y1*cotB-a)+(y2*y2)/rb/W6*W5+a*(y2*y2)/ (rb*rb*rb))/rb*y2+W8/W6*(-1/(rb*rb*rb)*(N1*y1*cotB-a)*y2+2*y2/rb/ W6*W5-(y2*y2*y2)/(rb*rb*rb)/W6*W5-(y2*y2*y2)/rb2/(W6*W6)*W5-(y2*y2*y2)/(rb2*rb2)/W6* a+2*a/(rb*rb*rb)*y2-3*a*(y2*y2*y2)/(rb*rb2*rb2))-W8/(W7*W7)*((cosB*cosB)-1/rb*(N1* z1b*cotB+a*cosB)+a*y3b*z1b*cotB/(rb*rb*rb)-1/rb/W7*((y2*y2)*(cosB*cosB)- a*z1b*cotB/rb*W1))/rb*y2+W8/W7*(1/(rb*rb*rb)*(N1*z1b*cotB+a* cosB)*y2-3*a*y3b*z1b*cotB/(rb*rb2*rb2)*y2+1/(rb*rb*rb)/W7*((y2*y2)*(cosB*cosB)- a*z1b*cotB/rb*W1)*y2+1/rb2/(W7*W7)*((y2*y2)*(cosB*cosB)-a*z1b*cotB/ rb*W1)*y2-1/rb/W7*(2*y2*(cosB*cosB)+a*z1b*cotB/(rb*rb*rb)*W1*y2-a* z1b*cotB/rb2*cosB*y2)))/M_PI/(1-nu))+ b2*(1.0/4.0*((2-2*nu)*N1*rFib_ry2*(cotB*cotB)+N1/W6*((W5-1)*cotB+y1/W6* W4)-N1*(y2*y2)/(W6*W6)*((W5-1)*cotB+y1/W6*W4)/rb+N1*y2/W6*(-a/ (rb*rb*rb)*y2*cotB-y1/(W6*W6)*W4/rb*y2-y2/W6*a/(rb*rb*rb)*y1)-N1*cotB/ W7*W9+N1*(y2*y2)*cotB/(W7*W7)*W9/rb+N1*(y2*y2)*cotB/W7*a/(rb*rb*rb)/ cosB-a*W8*cotB/(rb*rb*rb)+3*a*(y2*y2)*W8*cotB/(rb*rb2*rb2)+W8/rb/W6*(N1* cotB-2*nu*y1/W6-a*y1/rb*(1/rb+1/W6))-(y2*y2)*W8/(rb*rb*rb)/W6* (N1*cotB-2*nu*y1/W6-a*y1/rb*(1/rb+1/W6))-(y2*y2)*W8/rb2/(W6*W6) *(N1*cotB-2*nu*y1/W6-a*y1/rb*(1/rb+1/W6))+y2*W8/rb/W6* (2*nu*y1/(W6*W6)/rb*y2+a*y1/(rb*rb*rb)*(1/rb+1/W6)*y2-a*y1/rb* (-1/(rb*rb*rb)*y2-1/(W6*W6)/rb*y2))+W8*cotB/rb/W7*((-2+2*nu)*cosB+ W1/W7*W9+a*y3b/rb2/cosB)-(y2*y2)*W8*cotB/(rb*rb*rb)/W7*((-2+2*nu)* cosB+W1/W7*W9+a*y3b/rb2/cosB)-(y2*y2)*W8*cotB/rb2/(W7*W7)*((-2+ 2*nu)*cosB+W1/W7*W9+a*y3b/rb2/cosB)+y2*W8*cotB/rb/W7*(1/ rb*cosB*y2/W7*W9-W1/(W7*W7)*W9/rb*y2-W1/W7*a/(rb*rb*rb)/cosB*y2- 2*a*y3b/(rb2*rb2)/cosB*y2))/M_PI/(1-nu))+ b3*(1.0/4.0*(N1*(-sinB/rb*y2/W7+y2/(W6*W6)*(1+a/rb)/rb*y1+y2/W6* a/(rb*rb*rb)*y1-z1b/(W7*W7)*W2/rb*y2-z1b/W7*a/(rb*rb*rb)*y2)-y2*W8/ (rb*rb*rb)*(a/rb2+1/W6)*y1+y1*W8/rb*(-2*a/(rb2*rb2)*y2-1/(W6*W6)/ rb*y2)+W8/(W7*W7)*(sinB*(cosB-a/rb)+z1b/rb*(1+a*y3b/rb2)-1/ rb/W7*((y2*y2)*cosB*sinB-a*z1b/rb*W1))/rb*y2-W8/W7*(sinB*a/ (rb*rb*rb)*y2-z1b/(rb*rb*rb)*(1+a*y3b/rb2)*y2-2*z1b/(rb*rb2*rb2)*a*y3b*y2+ 1/(rb*rb*rb)/W7*((y2*y2)*cosB*sinB-a*z1b/rb*W1)*y2+1/rb2/(W7*W7)* ((y2*y2)*cosB*sinB-a*z1b/rb*W1)*y2-1/rb/W7*(2*y2*cosB*sinB+a* z1b/(rb*rb*rb)*W1*y2-a*z1b/rb2*cosB*y2)))/M_PI/(1-nu));

    out.z = b1*(1.0/4.0*((2-2*nu)*(N1*rFib_ry3*cotB-y2/(W6*W6)*W5*(y3b/rb+1)- 1.0/2.0*y2/W6*a/(rb*rb*rb)*2*y3b+y2*cosB/(W7*W7)*W2*W3+1.0/2.0*y2*cosB/W7* a/(rb*rb*rb)*2*y3b)+y2/rb*(2*nu/W6+a/rb2)-1.0/2.0*y2*W8/(rb*rb*rb)*(2* nu/W6+a/rb2)*2*y3b+y2*W8/rb*(-2*nu/(W6*W6)*(y3b/rb+1)-a/ (rb2*rb2)*2*y3b)+y2*cosB/rb/W7*(1-2*nu-W1/W7*W2-a*y3b/rb2)- 1.0/2.0*y2*W8*cosB/(rb*rb*rb)/W7*(1-2*nu-W1/W7*W2-a*y3b/rb2)*2* y3b-y2*W8*cosB/rb/(W7*W7)*(1-2*nu-W1/W7*W2-a*y3b/rb2)*W3+y2* W8*cosB/rb/W7*(-(cosB*y3b/rb+1)/W7*W2+W1/(W7*W7)*W2*W3+1.0/2.0* W1/W7*a/(rb*rb*rb)*2*y3b-a/rb2+a*y3b/(rb2*rb2)*2*y3b))/M_PI/(1-nu))+ b2*(1.0/4.0*((-2+2*nu)*N1*cotB*((y3b/rb+1)/W6-cosB*W3/W7)+(2-2*nu)* y1/(W6*W6)*W5*(y3b/rb+1)+1.0/2.0*(2-2*nu)*y1/W6*a/(rb*rb*rb)*2*y3b+(2- 2*nu)*sinB/W7*W2-(2-2*nu)*z1b/(W7*W7)*W2*W3-1.0/2.0*(2-2*nu)*z1b/ W7*a/(rb*rb*rb)*2*y3b+1/rb*(N1*cotB-2*nu*y1/W6-a*y1/rb2)-1.0/2.0* W8/(rb*rb*rb)*(N1*cotB-2*nu*y1/W6-a*y1/rb2)*2*y3b+W8/rb*(2*nu* y1/(W6*W6)*(y3b/rb+1)+a*y1/(rb2*rb2)*2*y3b)-1/W7*(cosB*sinB+W1* cotB/rb*((2-2*nu)*cosB-W1/W7)+a/rb*(sinB-y3b*z1b/rb2-z1b* W1/rb/W7))+W8/(W7*W7)*(cosB*sinB+W1*cotB/rb*((2-2*nu)*cosB-W1/ W7)+a/rb*(sinB-y3b*z1b/rb2-z1b*W1/rb/W7))*W3-W8/W7*((cosB* y3b/rb+1)*cotB/rb*((2-2*nu)*cosB-W1/W7)-1.0/2.0*W1*cotB/(rb*rb*rb)* ((2-2*nu)*cosB-W1/W7)*2*y3b+W1*cotB/rb*(-(cosB*y3b/rb+1)/W7+ W1/(W7*W7)*W3)-1.0/2.0*a/(rb*rb*rb)*(sinB-y3b*z1b/rb2-z1b*W1/rb/W7)* 2*y3b+a/rb*(-z1b/rb2-y3b*sinB/rb2+y3b*z1b/(rb2*rb2)*2*y3b- sinB*W1/rb/W7-z1b*(cosB*y3b/rb+1)/rb/W7+1.0/2.0*z1b*W1/(rb*rb*rb)/ W7*2*y3b+z1b*W1/rb/(W7*W7)*W3)))/M_PI/(1-nu))+ b3*(1.0/4.0*((2-2*nu)*rFib_ry3-(2-2*nu)*y2*sinB/(W7*W7)*W2*W3-1.0/2.0* (2-2*nu)*y2*sinB/W7*a/(rb*rb*rb)*2*y3b+y2*sinB/rb/W7*(1+W1/W7* W2+a*y3b/rb2)-1.0/2.0*y2*W8*sinB/(rb*rb*rb)/W7*(1+W1/W7*W2+a*y3b/ rb2)*2*y3b-y2*W8*sinB/rb/(W7*W7)*(1+W1/W7*W2+a*y3b/rb2)*W3+ y2*W8*sinB/rb/W7*((cosB*y3b/rb+1)/W7*W2-W1/(W7*W7)*W2*W3- 1.0/2.0*W1/W7*a/(rb*rb*rb)*2*y3b+a/rb2-a*y3b/(rb2*rb2)*2*y3b))/M_PI/(1-nu));

    out.a = b1/2*(1.0/4.0*((-2+2*nu)*N1*rFib_ry2*(cotB*cotB)+N1/W6*((1-W5)*cotB-y1/ W6*W4)-N1*(y2*y2)/(W6*W6)*((1-W5)*cotB-y1/W6*W4)/rb+N1*y2/W6* (a/(rb*rb*rb)*y2*cotB+y1/(W6*W6)*W4/rb*y2+y2/W6*a/(rb*rb*rb)*y1)+N1* cosB*cotB/W7*W2-N1*(y2*y2)*cosB*cotB/(W7*W7)*W2/rb-N1*(y2*y2)*cosB* cotB/W7*a/(rb*rb*rb)+a*W8*cotB/(rb*rb*rb)-3*a*(y2*y2)*W8*cotB/(rb*rb2*rb2)+W8/ rb/W6*(-N1*cotB+y1/W6*W5+a*y1/rb2)-(y2*y2)*W8/(rb*rb*rb)/W6*(-N1* cotB+y1/W6*W5+a*y1/rb2)-(y2*y2)*W8/rb2/(W6*W6)*(-N1*cotB+y1/ W6*W5+a*y1/rb2)+y2*W8/rb/W6*(-y1/(W6*W6)*W5/rb*y2-y2/W6* a/(rb*rb*rb)*y1-2*a*y1/(rb2*rb2)*y2)+W8/rb/W7*(cosB/W7*(W1*(N1* cosB-a/rb)*cotB+(2-2*nu)*(rb*sinB-y1)*cosB)-a*y3b*cosB*cotB/ rb2)-(y2*y2)*W8/(rb*rb*rb)/W7*(cosB/W7*(W1*(N1*cosB-a/rb)*cotB+(2- 2*nu)*(rb*sinB-y1)*cosB)-a*y3b*cosB*cotB/rb2)-(y2*y2)*W8/rb2/ (W7*W7)*(cosB/W7*(W1*(N1*cosB-a/rb)*cotB+(2-2*nu)*(rb*sinB-y1)* cosB)-a*y3b*cosB*cotB/rb2)+y2*W8/rb/W7*(-cosB/(W7*W7)*(W1* (N1*cosB-a/rb)*cotB+(2-2*nu)*(rb*sinB-y1)*cosB)/rb*y2+cosB/ W7*(1/rb*cosB*y2*(N1*cosB-a/rb)*cotB+W1*a/(rb*rb*rb)*y2*cotB+(2-2* nu)/rb*sinB*y2*cosB)+2*a*y3b*cosB*cotB/(rb2*rb2)*y2))/M_PI/(1-nu))+ b2/2*(1.0/4.0*(N1*(((2-2*nu)*(cotB*cotB)+nu)/rb*y2/W6-((2-2*nu)*(cotB*cotB)+1)* cosB/rb*y2/W7)-N1/(W6*W6)*(-N1*y1*cotB+nu*y3b-a+a*y1*cotB/rb+ (y1*y1)/W6*W4)/rb*y2+N1/W6*(-a*y1*cotB/(rb*rb*rb)*y2-(y1*y1)/(W6*W6) *W4/rb*y2-(y1*y1)/W6*a/(rb*rb*rb)*y2)+N1*cotB/(W7*W7)*(z1b*cosB-a* (rb*sinB-y1)/rb/cosB)/rb*y2-N1*cotB/W7*(-a/rb2*sinB*y2/ cosB+a*(rb*sinB-y1)/(rb*rb*rb)/cosB*y2)+3*a*y2*W8*cotB/(rb*rb2*rb2)*y1- W8/(W6*W6)*(2*nu+1/rb*(N1*y1*cotB+a)-(y1*y1)/rb/W6*W5-a*(y1*y1)/ (rb*rb*rb))/rb*y2+W8/W6*(-1/(rb*rb*rb)*(N1*y1*cotB+a)*y2+(y1*y1)/(rb*rb2) /W6*W5*y2+(y1*y1)/rb2/(W6*W6)*W5*y2+(y1*y1)/(rb2*rb2)/W6*a*y2+3* a*(y1*y1)/(rb*rb2*rb2)*y2)-W8*cotB/(W7*W7)*(-cosB*sinB+a*y1*y3b/(rb*rb*rb)/ cosB+(rb*sinB-y1)/rb*((2-2*nu)*cosB-W1/W7*W9))/rb*y2+W8*cotB/ W7*(-3*a*y1*y3b/(rb*rb2*rb2)/cosB*y2+1/rb2*sinB*y2*((2-2*nu)*cosB- W1/W7*W9)-(rb*sinB-y1)/(rb*rb*rb)*((2-2*nu)*cosB-W1/W7*W9)*y2+(rb* sinB-y1)/rb*(-1/rb*cosB*y2/W7*W9+W1/(W7*W7)*W9/rb*y2+W1/W7* a/(rb*rb*rb)/cosB*y2)))/M_PI/(1-nu))+ b3/2*(1.0/4.0*(N1*(1/W6*(1+a/rb)-(y2*y2)/(W6*W6)*(1+a/rb)/rb-(y2*y2)/ W6*a/(rb*rb*rb)-cosB/W7*W2+(y2*y2)*cosB/(W7*W7)*W2/rb+(y2*y2)*cosB/W7* a/(rb*rb*rb))-W8/rb*(a/rb2+1/W6)+(y2*y2)*W8/(rb*rb*rb)*(a/rb2+1/W6)- y2*W8/rb*(-2*a/(rb2*rb2)*y2-1/(W6*W6)/rb*y2)+W8*cosB/rb/W7* (W1/W7*W2+a*y3b/rb2)-(y2*y2)*W8*cosB/(rb*rb*rb)/W7*(W1/W7*W2+a* y3b/rb2)-(y2*y2)*W8*cosB/rb2/(W7*W7)*(W1/W7*W2+a*y3b/rb2)+y2* W8*cosB/rb/W7*(1/rb*cosB*y2/W7*W2-W1/(W7*W7)*W2/rb*y2-W1/ W7*a/(rb*rb*rb)*y2-2*a*y3b/(rb2*rb2)*y2))/M_PI/(1-nu))+ b1/2*(1.0/4.0*(N1*(((2-2*nu)*(cotB*cotB)-nu)/rb*y1/W6-((2-2*nu)*(cotB*cotB)+1- 2*nu)*cosB*(y1/rb-sinB)/W7)+N1/(W6*W6)*(y1*cotB*(1-W5)+nu*y3b- a+(y2*y2)/W6*W4)/rb*y1-N1/W6*((1-W5)*cotB+a*(y1*y1)*cotB/(rb*rb*rb)- (y2*y2)/(W6*W6)*W4/rb*y1-(y2*y2)/W6*a/(rb*rb*rb)*y1)-N1*cosB*cotB/W7* W2+N1*z1b*cotB/(W7*W7)*W2*(y1/rb-sinB)+N1*z1b*cotB/W7*a/(rb*rb2) *y1-a*W8*cotB/(rb*rb*rb)+3*a*(y1*y1)*W8*cotB/(rb*rb2*rb2)-W8/(W6*W6)*(-2* nu+1/rb*(N1*y1*cotB-a)+(y2*y2)/rb/W6*W5+a*(y2*y2)/(rb*rb*rb))/rb* y1+W8/W6*(-1/(rb*rb*rb)*(N1*y1*cotB-a)*y1+1/rb*N1*cotB-(y2*y2)/ (rb*rb*rb)/W6*W5*y1-(y2*y2)/rb2/(W6*W6)*W5*y1-(y2*y2)/(rb2*rb2)/W6*a*y1- 3*a*(y2*y2)/(rb*rb2*rb2)*y1)-W8/(W7*W7)*((cosB*cosB)-1/rb*(N1*z1b*cotB+a* cosB)+a*y3b*z1b*cotB/(rb*rb*rb)-1/rb/W7*((y2*y2)*(cosB*cosB)-a*z1b*cotB/ rb*W1))*(y1/rb-sinB)+W8/W7*(1/(rb*rb*rb)*(N1*z1b*cotB+a*cosB)* y1-1/rb*N1*cosB*cotB+a*y3b*cosB*cotB/(rb*rb*rb)-3*a*y3b*z1b*cotB/ (rb*rb2*rb2)*y1+1/(rb*rb*rb)/W7*((y2*y2)*(cosB*cosB)-a*z1b*cotB/rb*W1)*y1+1/ rb/(W7*W7)*((y2*y2)*(cosB*cosB)-a*z1b*cotB/rb*W1)*(y1/rb-sinB)-1/rb/ W7*(-a*cosB*cotB/rb*W1+a*z1b*cotB/(rb*rb*rb)*W1*y1-a*z1b*cotB/ rb2*cosB*y1)))/M_PI/(1-nu))+ b2/2*(1.0/4.0*((2-2*nu)*N1*rFib_ry1*(cotB*cotB)-N1*y2/(W6*W6)*((W5-1)*cotB+ y1/W6*W4)/rb*y1+N1*y2/W6*(-a/(rb*rb*rb)*y1*cotB+1/W6*W4-(y1*y1) /(W6*W6)*W4/rb-(y1*y1)/W6*a/(rb*rb*rb))+N1*y2*cotB/(W7*W7)*W9*(y1/ rb-sinB)+N1*y2*cotB/W7*a/(rb*rb*rb)/cosB*y1+3*a*y2*W8*cotB/(rb*rb2*rb2) *y1-y2*W8/(rb*rb*rb)/W6*(N1*cotB-2*nu*y1/W6-a*y1/rb*(1/rb+1/ W6))*y1-y2*W8/rb2/(W6*W6)*(N1*cotB-2*nu*y1/W6-a*y1/rb*(1/ rb+1/W6))*y1+y2*W8/rb/W6*(-2*nu/W6+2*nu*(y1*y1)/(W6*W6)/rb-a/ rb*(1/rb+1/W6)+a*(y1*y1)/(rb*rb*rb)*(1/rb+1/W6)-a*y1/rb*(-1/ (rb*rb*rb)*y1-1/(W6*W6)/rb*y1))-y2*W8*cotB/(rb*rb*rb)/W7*((-2+2*nu)* cosB+W1/W7*W9+a*y3b/rb2/cosB)*y1-y2*W8*cotB/rb/(W7*W7)*((-2+ 2*nu)*cosB+W1/W7*W9+a*y3b/rb2/cosB)*(y1/rb-sinB)+y2*W8* cotB/rb/W7*(1/rb*cosB*y1/W7*W9-W1/(W7*W7)*W9*(y1/rb-sinB)- W1/W7*a/(rb*rb*rb)/cosB*y1-2*a*y3b/(rb2*rb2)/cosB*y1))/M_PI/(1-nu))+ b3/2*(1.0/4.0*(N1*(-sinB*(y1/rb-sinB)/W7-1/W6*(1+a/rb)+(y1*y1)/(W6*W6) *(1+a/rb)/rb+(y1*y1)/W6*a/(rb*rb*rb)+cosB/W7*W2-z1b/(W7*W7)*W2* (y1/rb-sinB)-z1b/W7*a/(rb*rb*rb)*y1)+W8/rb*(a/rb2+1/W6)-(y1*y1)* W8/(rb*rb*rb)*(a/rb2+1/W6)+y1*W8/rb*(-2*a/(rb2*rb2)*y1-1/(W6*W6)/ rb*y1)+W8/(W7*W7)*(sinB*(cosB-a/rb)+z1b/rb*(1+a*y3b/rb2)-1/ rb/W7*((y2*y2)*cosB*sinB-a*z1b/rb*W1))*(y1/rb-sinB)-W8/W7* (sinB*a/(rb*rb*rb)*y1+cosB/rb*(1+a*y3b/rb2)-z1b/(rb*rb*rb)*(1+a*y3b/ rb2)*y1-2*z1b/(rb*rb2*rb2)*a*y3b*y1+1/(rb*rb*rb)/W7*((y2*y2)*cosB*sinB-a* z1b/rb*W1)*y1+1/rb/(W7*W7)*((y2*y2)*cosB*sinB-a*z1b/rb*W1)* (y1/rb-sinB)-1/rb/W7*(-a*cosB/rb*W1+a*z1b/(rb*rb*rb)*W1*y1-a* z1b/rb2*cosB*y1)))/M_PI/(1-nu));

    out.b = b1/2*(1.0/4.0*((-2+2*nu)*N1*rFib_ry3*(cotB*cotB)-N1*y2/(W6*W6)*((1-W5)* cotB-y1/W6*W4)*(y3b/rb+1)+N1*y2/W6*(1.0/2.0*a/(rb*rb*rb)*2*y3b*cotB+ y1/(W6*W6)*W4*(y3b/rb+1)+1.0/2.0*y1/W6*a/(rb*rb*rb)*2*y3b)-N1*y2*cosB* cotB/(W7*W7)*W2*W3-1.0/2.0*N1*y2*cosB*cotB/W7*a/(rb*rb*rb)*2*y3b+a/ (rb*rb*rb)*y2*cotB-3.0/2.0*a*y2*W8*cotB/(rb*rb2*rb2)*2*y3b+y2/rb/W6*(-N1* cotB+y1/W6*W5+a*y1/rb2)-1.0/2.0*y2*W8/(rb*rb*rb)/W6*(-N1*cotB+y1/ W6*W5+a*y1/rb2)*2*y3b-y2*W8/rb/(W6*W6)*(-N1*cotB+y1/W6*W5+ a*y1/rb2)*(y3b/rb+1)+y2*W8/rb/W6*(-y1/(W6*W6)*W5*(y3b/rb+ 1)-1.0/2.0*y1/W6*a/(rb*rb*rb)*2*y3b-a*y1/(rb2*rb2)*2*y3b)+y2/rb/W7* (cosB/W7*(W1*(N1*cosB-a/rb)*cotB+(2-2*nu)*(rb*sinB-y1)*cosB)- a*y3b*cosB*cotB/rb2)-1.0/2.0*y2*W8/(rb*rb*rb)/W7*(cosB/W7*(W1*(N1* cosB-a/rb)*cotB+(2-2*nu)*(rb*sinB-y1)*cosB)-a*y3b*cosB*cotB/ rb2)*2*y3b-y2*W8/rb/(W7*W7)*(cosB/W7*(W1*(N1*cosB-a/rb)*cotB+ (2-2*nu)*(rb*sinB-y1)*cosB)-a*y3b*cosB*cotB/rb2)*W3+y2*W8/rb/ W7*(-cosB/(W7*W7)*(W1*(N1*cosB-a/rb)*cotB+(2-2*nu)*(rb*sinB-y1)* cosB)*W3+cosB/W7*((cosB*y3b/rb+1)*(N1*cosB-a/rb)*cotB+1.0/2.0*W1* a/(rb*rb*rb)*2*y3b*cotB+1.0/2.0*(2-2*nu)/rb*sinB*2*y3b*cosB)-a*cosB* cotB/rb2+a*y3b*cosB*cotB/(rb2*rb2)*2*y3b))/M_PI/(1-nu))+ b2/2*(1.0/4.0*(N1*(((2-2*nu)*(cotB*cotB)+nu)*(y3b/rb+1)/W6-((2-2*nu)*(cotB*cotB) +1)*cosB*W3/W7)-N1/(W6*W6)*(-N1*y1*cotB+nu*y3b-a+a*y1*cotB/ rb+(y1*y1)/W6*W4)*(y3b/rb+1)+N1/W6*(nu-1.0/2.0*a*y1*cotB/(rb*rb*rb)*2* y3b-(y1*y1)/(W6*W6)*W4*(y3b/rb+1)-1.0/2.0*(y1*y1)/W6*a/(rb*rb*rb)*2*y3b)+ N1*cotB/(W7*W7)*(z1b*cosB-a*(rb*sinB-y1)/rb/cosB)*W3-N1*cotB/ W7*(cosB*sinB-1.0/2.0*a/rb2*sinB*2*y3b/cosB+1.0/2.0*a*(rb*sinB-y1)/ (rb*rb*rb)/cosB*2*y3b)-a/(rb*rb*rb)*y1*cotB+3.0/2.0*a*y1*W8*cotB/(rb*rb2*rb2)*2* y3b+1/W6*(2*nu+1/rb*(N1*y1*cotB+a)-(y1*y1)/rb/W6*W5-a*(y1*y1)/ (rb*rb*rb))-W8/(W6*W6)*(2*nu+1/rb*(N1*y1*cotB+a)-(y1*y1)/rb/W6*W5-a* (y1*y1)/(rb*rb*rb))*(y3b/rb+1)+W8/W6*(-1.0/2.0/(rb*rb*rb)*(N1*y1*cotB+a)*2* y3b+1.0/2.0*(y1*y1)/(rb*rb*rb)/W6*W5*2*y3b+(y1*y1)/rb/(W6*W6)*W5*(y3b/rb+ 1)+1.0/2.0*(y1*y1)/(rb2*rb2)/W6*a*2*y3b+3.0/2.0*a*(y1*y1)/(rb*rb2*rb2)*2*y3b)+ cotB/W7*(-cosB*sinB+a*y1*y3b/(rb*rb*rb)/cosB+(rb*sinB-y1)/rb*((2- 2*nu)*cosB-W1/W7*W9))-W8*cotB/(W7*W7)*(-cosB*sinB+a*y1*y3b/(rb*rb2) /cosB+(rb*sinB-y1)/rb*((2-2*nu)*cosB-W1/W7*W9))*W3+W8*cotB/ W7*(a/(rb*rb*rb)/cosB*y1-3.0/2.0*a*y1*y3b/(rb*rb2*rb2)/cosB*2*y3b+1.0/2.0/ rb2*sinB*2*y3b*((2-2*nu)*cosB-W1/W7*W9)-1.0/2.0*(rb*sinB-y1)/(rb*rb2) *((2-2*nu)*cosB-W1/W7*W9)*2*y3b+(rb*sinB-y1)/rb*(-(cosB*y3b/ rb+1)/W7*W9+W1/(W7*W7)*W9*W3+1.0/2.0*W1/W7*a/(rb*rb*rb)/cosB*2* y3b)))/M_PI/(1-nu))+ b3/2*(1.0/4.0*(N1*(-y2/(W6*W6)*(1+a/rb)*(y3b/rb+1)-1.0/2.0*y2/W6*a/ (rb*rb*rb)*2*y3b+y2*cosB/(W7*W7)*W2*W3+1.0/2.0*y2*cosB/W7*a/(rb*rb*rb)*2* y3b)-y2/rb*(a/rb2+1/W6)+1.0/2.0*y2*W8/(rb*rb*rb)*(a/rb2+1/W6)*2* y3b-y2*W8/rb*(-a/(rb2*rb2)*2*y3b-1/(W6*W6)*(y3b/rb+1))+y2*cosB/ rb/W7*(W1/W7*W2+a*y3b/rb2)-1.0/2.0*y2*W8*cosB/(rb*rb*rb)/W7*(W1/ W7*W2+a*y3b/rb2)*2*y3b-y2*W8*cosB/rb/(W7*W7)*(W1/W7*W2+a* y3b/rb2)*W3+y2*W8*cosB/rb/W7*((cosB*y3b/rb+1)/W7*W2-W1/ (W7*W7)*W2*W3-1.0/2.0*W1/W7*a/(rb*rb*rb)*2*y3b+a/rb2-a*y3b/(rb2*rb2)*2* y3b))/M_PI/(1-nu))+ b1/2.0*(1.0/4.0*((2-2*nu)*(N1*rFib_ry1*cotB-y1/(W6*W6)*W5/rb*y2-y2/W6* a/(rb*rb*rb)*y1+y2*cosB/(W7*W7)*W2*(y1/rb-sinB)+y2*cosB/W7*a/(rb*rb2) *y1)-y2*W8/(rb*rb*rb)*(2*nu/W6+a/rb2)*y1+y2*W8/rb*(-2*nu/(W6*W6) /rb*y1-2*a/(rb2*rb2)*y1)-y2*W8*cosB/(rb*rb*rb)/W7*(1-2*nu-W1/W7* W2-a*y3b/rb2)*y1-y2*W8*cosB/rb/(W7*W7)*(1-2*nu-W1/W7*W2-a* y3b/rb2)*(y1/rb-sinB)+y2*W8*cosB/rb/W7*(-1/rb*cosB*y1/W7* W2+W1/(W7*W7)*W2*(y1/rb-sinB)+W1/W7*a/(rb*rb*rb)*y1+2*a*y3b/(rb2*rb2) *y1))/M_PI/(1-nu))+ b2/2*(1.0/4.0*((-2+2*nu)*N1*cotB*(1/rb*y1/W6-cosB*(y1/rb-sinB)/W7)- (2-2*nu)/W6*W5+(2-2*nu)*(y1*y1)/(W6*W6)*W5/rb+(2-2*nu)*(y1*y1)/W6* a/(rb*rb*rb)+(2-2*nu)*cosB/W7*W2-(2-2*nu)*z1b/(W7*W7)*W2*(y1/rb- sinB)-(2-2*nu)*z1b/W7*a/(rb*rb*rb)*y1-W8/(rb*rb*rb)*(N1*cotB-2*nu*y1/ W6-a*y1/rb2)*y1+W8/rb*(-2*nu/W6+2*nu*(y1*y1)/(W6*W6)/rb-a/rb2+ 2*a*(y1*y1)/(rb2*rb2))+W8/(W7*W7)*(cosB*sinB+W1*cotB/rb*((2-2*nu)* cosB-W1/W7)+a/rb*(sinB-y3b*z1b/rb2-z1b*W1/rb/W7))*(y1/rb- sinB)-W8/W7*(1/rb2*cosB*y1*cotB*((2-2*nu)*cosB-W1/W7)-W1* cotB/(rb*rb*rb)*((2-2*nu)*cosB-W1/W7)*y1+W1*cotB/rb*(-1/rb*cosB* y1/W7+W1/(W7*W7)*(y1/rb-sinB))-a/(rb*rb*rb)*(sinB-y3b*z1b/rb2- z1b*W1/rb/W7)*y1+a/rb*(-y3b*cosB/rb2+2*y3b*z1b/(rb2*rb2)*y1- cosB*W1/rb/W7-z1b/rb2*cosB*y1/W7+z1b*W1/(rb*rb*rb)/W7*y1+z1b* W1/rb/(W7*W7)*(y1/rb-sinB))))/M_PI/(1-nu))+ b3/2*(1.0/4.0*((2-2*nu)*rFib_ry1-(2-2*nu)*y2*sinB/(W7*W7)*W2*(y1/rb- sinB)-(2-2*nu)*y2*sinB/W7*a/(rb*rb*rb)*y1-y2*W8*sinB/(rb*rb*rb)/W7*(1+ W1/W7*W2+a*y3b/rb2)*y1-y2*W8*sinB/rb/(W7*W7)*(1+W1/W7*W2+ a*y3b/rb2)*(y1/rb-sinB)+y2*W8*sinB/rb/W7*(1/rb*cosB*y1/ W7*W2-W1/(W7*W7)*W2*(y1/rb-sinB)-W1/W7*a/(rb*rb*rb)*y1-2*a*y3b/ (rb2*rb2)*y1))/M_PI/(1-nu));

    out.c = b1/2*(1.0/4.0*(N1*(((2-2*nu)*(cotB*cotB)-nu)*(y3b/rb+1)/W6-((2-2*nu)* (cotB*cotB)+1-2*nu)*cosB*W3/W7)+N1/(W6*W6)*(y1*cotB*(1-W5)+nu*y3b-a+ (y2*y2)/W6*W4)*(y3b/rb+1)-N1/W6*(1.0/2.0*a*y1*cotB/(rb*rb*rb)*2*y3b+ nu-(y2*y2)/(W6*W6)*W4*(y3b/rb+1)-1.0/2.0*(y2*y2)/W6*a/(rb*rb*rb)*2*y3b)-N1* sinB*cotB/W7*W2+N1*z1b*cotB/(W7*W7)*W2*W3+1.0/2.0*N1*z1b*cotB/W7* a/(rb*rb*rb)*2*y3b-a/(rb*rb*rb)*y1*cotB+3.0/2.0*a*y1*W8*cotB/(rb*rb2*rb2)*2*y3b+ 1/W6*(-2*nu+1/rb*(N1*y1*cotB-a)+(y2*y2)/rb/W6*W5+a*(y2*y2)/ (rb*rb*rb))-W8/(W6*W6)*(-2*nu+1/rb*(N1*y1*cotB-a)+(y2*y2)/rb/W6*W5+ a*(y2*y2)/(rb*rb*rb))*(y3b/rb+1)+W8/W6*(-1.0/2.0/(rb*rb*rb)*(N1*y1*cotB-a)* 2*y3b-1.0/2.0*(y2*y2)/(rb*rb*rb)/W6*W5*2*y3b-(y2*y2)/rb/(W6*W6)*W5*(y3b/ rb+1)-1.0/2.0*(y2*y2)/(rb2*rb2)/W6*a*2*y3b-3.0/2.0*a*(y2*y2)/(rb*rb2*rb2)*2*y3b)+ 1/W7*((cosB*cosB)-1/rb*(N1*z1b*cotB+a*cosB)+a*y3b*z1b*cotB/(rb*rb2) -1/rb/W7*((y2*y2)*(cosB*cosB)-a*z1b*cotB/rb*W1))-W8/(W7*W7)*((cosB*cosB)- 1/rb*(N1*z1b*cotB+a*cosB)+a*y3b*z1b*cotB/(rb*rb*rb)-1/rb/W7* ((y2*y2)*(cosB*cosB)-a*z1b*cotB/rb*W1))*W3+W8/W7*(1.0/2.0/(rb*rb*rb)*(N1* z1b*cotB+a*cosB)*2*y3b-1/rb*N1*sinB*cotB+a*z1b*cotB/(rb*rb*rb)+a* y3b*sinB*cotB/(rb*rb*rb)-3.0/2.0*a*y3b*z1b*cotB/(rb*rb2*rb2)*2*y3b+1.0/2.0/(rb*rb2) /W7*((y2*y2)*(cosB*cosB)-a*z1b*cotB/rb*W1)*2*y3b+1/rb/(W7*W7)*((y2*y2) *(cosB*cosB)-a*z1b*cotB/rb*W1)*W3-1/rb/W7*(-a*sinB*cotB/rb*W1+ 1.0/2.0*a*z1b*cotB/(rb*rb*rb)*W1*2*y3b-a*z1b*cotB/rb*(cosB*y3b/rb+ 1))))/M_PI/(1-nu))+ b2/2*(1.0/4.0*((2-2*nu)*N1*rFib_ry3*(cotB*cotB)-N1*y2/(W6*W6)*((W5-1)*cotB+ y1/W6*W4)*(y3b/rb+1)+N1*y2/W6*(-1.0/2.0*a/(rb*rb*rb)*2*y3b*cotB-y1/ (W6*W6)*W4*(y3b/rb+1)-1.0/2.0*y1/W6*a/(rb*rb*rb)*2*y3b)+N1*y2*cotB/ (W7*W7)*W9*W3+1.0/2.0*N1*y2*cotB/W7*a/(rb*rb*rb)/cosB*2*y3b-a/(rb*rb*rb)* y2*cotB+3.0/2.0*a*y2*W8*cotB/(rb*rb2*rb2)*2*y3b+y2/rb/W6*(N1*cotB-2* nu*y1/W6-a*y1/rb*(1/rb+1/W6))-1.0/2.0*y2*W8/(rb*rb*rb)/W6*(N1* cotB-2*nu*y1/W6-a*y1/rb*(1/rb+1/W6))*2*y3b-y2*W8/rb/(W6*W6) *(N1*cotB-2*nu*y1/W6-a*y1/rb*(1/rb+1/W6))*(y3b/rb+1)+y2* W8/rb/W6*(2*nu*y1/(W6*W6)*(y3b/rb+1)+1.0/2.0*a*y1/(rb*rb*rb)*(1/rb+ 1/W6)*2*y3b-a*y1/rb*(-1.0/2.0/(rb*rb*rb)*2*y3b-1/(W6*W6)*(y3b/rb+ 1)))+y2*cotB/rb/W7*((-2+2*nu)*cosB+W1/W7*W9+a*y3b/rb2/cosB)- 1.0/2.0*y2*W8*cotB/(rb*rb*rb)/W7*((-2+2*nu)*cosB+W1/W7*W9+a*y3b/ rb2/cosB)*2*y3b-y2*W8*cotB/rb/(W7*W7)*((-2+2*nu)*cosB+W1/W7* W9+a*y3b/rb2/cosB)*W3+y2*W8*cotB/rb/W7*((cosB*y3b/rb+1)/ W7*W9-W1/(W7*W7)*W9*W3-1.0/2.0*W1/W7*a/(rb*rb*rb)/cosB*2*y3b+a/rb2/ cosB-a*y3b/(rb2*rb2)/cosB*2*y3b))/M_PI/(1-nu))+ b3/2*(1.0/4.0*(N1*(-sinB*W3/W7+y1/(W6*W6)*(1+a/rb)*(y3b/rb+1)+ 1.0/2.0*y1/W6*a/(rb*rb*rb)*2*y3b+sinB/W7*W2-z1b/(W7*W7)*W2*W3-1.0/2.0* z1b/W7*a/(rb*rb*rb)*2*y3b)+y1/rb*(a/rb2+1/W6)-1.0/2.0*y1*W8/(rb*rb2) *(a/rb2+1/W6)*2*y3b+y1*W8/rb*(-a/(rb2*rb2)*2*y3b-1/(W6*W6)* (y3b/rb+1))-1/W7*(sinB*(cosB-a/rb)+z1b/rb*(1+a*y3b/rb2)-1/ rb/W7*((y2*y2)*cosB*sinB-a*z1b/rb*W1))+W8/(W7*W7)*(sinB*(cosB- a/rb)+z1b/rb*(1+a*y3b/rb2)-1/rb/W7*((y2*y2)*cosB*sinB-a*z1b/ rb*W1))*W3-W8/W7*(1.0/2.0*sinB*a/(rb*rb*rb)*2*y3b+sinB/rb*(1+a*y3b/ rb2)-1.0/2.0*z1b/(rb*rb*rb)*(1+a*y3b/rb2)*2*y3b+z1b/rb*(a/rb2-a* y3b/(rb2*rb2)*2*y3b)+1.0/2.0/(rb*rb*rb)/W7*((y2*y2)*cosB*sinB-a*z1b/rb* W1)*2*y3b+1/rb/(W7*W7)*((y2*y2)*cosB*sinB-a*z1b/rb*W1)*W3-1/ rb/W7*(-a*sinB/rb*W1+1.0/2.0*a*z1b/(rb*rb*rb)*W1*2*y3b-a*z1b/rb* (cosB*y3b/rb+1))))/M_PI/(1-nu))+ b1/2.0*(1.0/4.0*((2-2*nu)*(N1*rFib_ry2*cotB+1/W6*W5-(y2*y2)/(W6*W6)*W5/ rb-(y2*y2)/W6*a/(rb*rb*rb)-cosB/W7*W2+(y2*y2)*cosB/(W7*W7)*W2/rb+(y2*y2)* cosB/W7*a/(rb*rb*rb))+W8/rb*(2*nu/W6+a/rb2)-(y2*y2)*W8/(rb*rb*rb)*(2* nu/W6+a/rb2)+y2*W8/rb*(-2*nu/(W6*W6)/rb*y2-2*a/(rb2*rb2)*y2)+ W8*cosB/rb/W7*(1-2*nu-W1/W7*W2-a*y3b/rb2)-(y2*y2)*W8*cosB/ (rb*rb*rb)/W7*(1-2*nu-W1/W7*W2-a*y3b/rb2)-(y2*y2)*W8*cosB/rb2/(W7*W7) *(1-2*nu-W1/W7*W2-a*y3b/rb2)+y2*W8*cosB/rb/W7*(-1/rb* cosB*y2/W7*W2+W1/(W7*W7)*W2/rb*y2+W1/W7*a/(rb*rb*rb)*y2+2*a* y3b/(rb2*rb2)*y2))/M_PI/(1-nu))+ b2/2*(1.0/4.0*((-2+2*nu)*N1*cotB*(1/rb*y2/W6-cosB/rb*y2/W7)+(2- 2*nu)*y1/(W6*W6)*W5/rb*y2+(2-2*nu)*y1/W6*a/(rb*rb*rb)*y2-(2-2* nu)*z1b/(W7*W7)*W2/rb*y2-(2-2*nu)*z1b/W7*a/(rb*rb*rb)*y2-W8/(rb*rb2) *(N1*cotB-2*nu*y1/W6-a*y1/rb2)*y2+W8/rb*(2*nu*y1/(W6*W6)/ rb*y2+2*a*y1/(rb2*rb2)*y2)+W8/(W7*W7)*(cosB*sinB+W1*cotB/rb*((2- 2*nu)*cosB-W1/W7)+a/rb*(sinB-y3b*z1b/rb2-z1b*W1/rb/W7))/ rb*y2-W8/W7*(1/rb2*cosB*y2*cotB*((2-2*nu)*cosB-W1/W7)-W1* cotB/(rb*rb*rb)*((2-2*nu)*cosB-W1/W7)*y2+W1*cotB/rb*(-cosB/rb* y2/W7+W1/(W7*W7)/rb*y2)-a/(rb*rb*rb)*(sinB-y3b*z1b/rb2-z1b*W1/ rb/W7)*y2+a/rb*(2*y3b*z1b/(rb2*rb2)*y2-z1b/rb2*cosB*y2/W7+ z1b*W1/(rb*rb*rb)/W7*y2+z1b*W1/rb2/(W7*W7)*y2)))/M_PI/(1-nu))+ b3/2*(1.0/4.0*((2-2*nu)*rFib_ry2+(2-2*nu)*sinB/W7*W2-(2-2*nu)*(y2*y2)* sinB/(W7*W7)*W2/rb-(2-2*nu)*(y2*y2)*sinB/W7*a/(rb*rb*rb)+W8*sinB/rb/ W7*(1+W1/W7*W2+a*y3b/rb2)-(y2*y2)*W8*sinB/(rb*rb*rb)/W7*(1+W1/ W7*W2+a*y3b/rb2)-(y2*y2)*W8*sinB/rb2/(W7*W7)*(1+W1/W7*W2+a* y3b/rb2)+y2*W8*sinB/rb/W7*(1/rb*cosB*y2/W7*W2-W1/(W7*W7)* W2/rb*y2-W1/W7*a/(rb*rb*rb)*y2-2*a*y3b/(rb2*rb2)*y2))/M_PI/(1-nu));

    return out;
}

WITHIN_KERNEL Real3 AngSetupFSC(Real3 obs, Real3 slip, Real3 PA, Real3 PB, Real nu) {
    Real3 SideVec = sub3(PB, PA);
    Real3 eZ = make3(0.0f,0.0f,1.0f);
    Real beta = acos(-dot3(normalize3(SideVec), eZ));
    if (fabs(beta) < EPS || fabs(M_PI-beta) < EPS) {
        return make3(0.0f, 0.0f, 0.0f);
    }
    Real3 ey1 = SideVec;
    ey1.z = 0;
    ey1 = normalize3(ey1);

    Real3 ey3 = negate3(eZ);
    Real3 ey2 = cross3(ey3, ey1);

    Real3 yA = transform3(ey1, ey2, ey3, sub3(obs, PA));
    Real3 yAB = transform3(ey1, ey2, ey3, SideVec);
    Real3 yB = sub3(yA, yAB);

    Real3 slip_adcs = transform3(ey1, ey2, ey3, slip);

    Real configuration = beta;
    if (beta*yA.x >= 0) {
        configuration = -M_PI+beta;
    }

    Real3 vA = AngDisDispFSC(
        yA.x, yA.y, yA.z, configuration,
        slip_adcs.x, slip_adcs.y, slip_adcs.z,
        nu, -PA.z
    );
    Real3 vB = AngDisDispFSC(
        yB.x, yB.y, yB.z, configuration,
        slip_adcs.x, slip_adcs.y, slip_adcs.z,
        nu, -PB.z
    );

    Real3 v = sub3(vB, vA);
    return inv_transform3(ey1, ey2, ey3, v);
}

WITHIN_KERNEL Real6 AngSetupFSC_S(Real3 obs, Real3 slip, Real3 PA, Real3 PB, Real nu) {
    Real3 SideVec = sub3(PB, PA);
    Real3 eZ = make3(0.0f,0.0f,1.0f);
    Real beta = acos(-dot3(normalize3(SideVec), eZ));
    if (fabs(beta) < EPS || fabs(M_PI-beta) < EPS) {
        return make6(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    }
    Real3 ey1 = SideVec;
    ey1.z = 0;
    ey1 = normalize3(ey1);

    Real3 ey3 = negate3(eZ);
    Real3 ey2 = cross3(ey3, ey1);

    Real3 yA = transform3(ey1, ey2, ey3, sub3(obs, PA));
    Real3 yAB = transform3(ey1, ey2, ey3, SideVec);
    Real3 yB = sub3(yA, yAB);

    Real3 slip_adcs = transform3(ey1, ey2, ey3, slip);

    Real configuration = beta;
    if (beta*yA.x >= 0) {
        configuration = -M_PI+beta;
    }

    Real6 vA = AngDisStrainFSC(
        yA.x, yA.y, yA.z, configuration,
        slip_adcs.x, slip_adcs.y, slip_adcs.z,
        nu, -PA.z
    );
    Real6 vB = AngDisStrainFSC(
        yB.x, yB.y, yB.z, configuration,
        slip_adcs.x, slip_adcs.y, slip_adcs.z,
        nu, -PB.z
    );
    Real6 v = sub6(vB, vA);
    return tensor_transform3(ey1, ey2, ey3, v);
}

#endif
</%def> //END OF defs()

<%def name="disp_fs(tri_prefix, is_halfspace='false')">
    ${setup_tde(tri_prefix, is_halfspace)}

    Real3 out;
    if (mode == 1) {
        // Calculate first angular dislocation contribution
        Real3 r1Tp = TDSetupD(transformed_obs,A,slip,nu,transformed_tri0, negate3(e13));
        // Calculate second angular dislocation contribution
        Real3 r2Tp = TDSetupD(transformed_obs,B,slip,nu,transformed_tri1, e12);
        // Calculate third angular dislocation contribution
        Real3 r3Tp = TDSetupD(transformed_obs,C,slip,nu,transformed_tri2, e23);
        out = add3(add3(r1Tp, r2Tp), r3Tp);
    } else if (mode == -1) {
        // Calculate first angular dislocation contribution
        Real3 r1Tn = TDSetupD(transformed_obs,A,slip,nu,transformed_tri0,e13);
        // Calculate second angular dislocation contribution
        Real3 r2Tn = TDSetupD(transformed_obs,B,slip,nu,transformed_tri1,negate3(e12));
        // Calculate third angular dislocation contribution
        Real3 r3Tn = TDSetupD(transformed_obs,C,slip,nu,transformed_tri2,negate3(e23));
        out = add3(add3(r1Tn, r2Tn), r3Tn);
    } else {
        out = make3(NAN, NAN, NAN);
    }

    Real3 a = make3(
        -transformed_obs.x,
        transformed_tri0.y - transformed_obs.y,
        transformed_tri0.z - transformed_obs.z
    );
    Real3 b = negate3(transformed_obs);
    Real3 c = make3(
        -transformed_obs.x,
        transformed_tri2.y - transformed_obs.y,
        transformed_tri2.z - transformed_obs.z
    );
    Real na = length3(a);
    Real nb = length3(b);
    Real nc = length3(c);

    Real FiN = (a.x*(b.y*c.z-b.z*c.y)- \
           a.y*(b.x*c.z-b.z*c.x)+ \
           a.z*(b.x*c.y-b.y*c.x));
    Real FiD = (na*nb*nc+dot3(a,b)*nc+dot3(a,c)*nb+dot3(b,c)*na);
    Real Fi = -2*atan2(FiN,FiD)/4/M_PI;

    // Calculate the complete displacement vector components in TDCS
    out = add3(out, mul_scalar3(slip,Fi));

    // Transform the complete displacement vector components from TDCS into EFCS
    Real3 full_out = inv_transform3(Vnorm, Vstrike, Vdip, out);
</%def>

<%def name="strain_fs(tri_prefix, is_halfspace='false')">
    ${setup_tde(tri_prefix, is_halfspace)}

    Real6 out;
    if (mode == 1) {
        // Calculate first angular dislocation contribution
        Real6 comp1 = TDSetupS(transformed_obs,A,slip,nu,transformed_tri0, negate3(e13));
        // Calculate second angular dislocation contribution
        Real6 comp2 = TDSetupS(transformed_obs,B,slip,nu,transformed_tri1, e12);
        // Calculate third angular dislocation contribution
        Real6 comp3 = TDSetupS(transformed_obs,C,slip,nu,transformed_tri2, e23);
        out = add6(add6(comp1, comp2), comp3);
    } else if (mode == -1) {
        // Calculate first angular dislocation contribution
        Real6 comp1 = TDSetupS(transformed_obs,A,slip,nu,transformed_tri0,e13);
        // Calculate second angular dislocation contribution
        Real6 comp2 = TDSetupS(transformed_obs,B,slip,nu,transformed_tri1,negate3(e12));
        // Calculate third angular dislocation contribution
        Real6 comp3 = TDSetupS(transformed_obs,C,slip,nu,transformed_tri2,negate3(e23));
        out = add6(add6(comp1, comp2), comp3);
    } else {
        out = make6(NAN, NAN, NAN, NAN, NAN, NAN);
    }


    Real6 full_out = tensor_transform3(Vnorm, Vstrike, Vdip, out);
</%def>

<%def name="setup_tde(tri_prefix, is_halfspace)">

    // printf("is_halfspace: %s\n", ${is_halfspace} ? "true":"false");
    Real3 Vnorm = normalize3(cross3(
        sub3(${tri_prefix}1, ${tri_prefix}0),
        sub3(${tri_prefix}2, ${tri_prefix}0)
    ));
    Real3 eY = make3(0.0f, 1.0f, 0.0f);
    Real3 eZ = make3(0.0f,0.0f,1.0f);
    Real3 Vstrike = cross3(eZ, Vnorm);
    if (length3(Vstrike) == 0) {
        Vstrike = mul_scalar3(eY, Vnorm.z);
        if (${is_halfspace}) {
            // For horizontal elements in case of half-space calculation!!!
            // Correct the strike vector of image dislocation only
            if (${tri_prefix}0.z > 0) {
                Vstrike = negate3(Vstrike);
            }
        }
    }
    Vstrike = normalize3(Vstrike);
    Real3 Vdip = cross3(Vnorm, Vstrike);

    Real3 transformed_obs = transform3(
        Vnorm, Vstrike, Vdip, sub3(obs, ${tri_prefix}1)
    );
    Real3 transformed_tri0 = transform3(
        Vnorm, Vstrike, Vdip, sub3(${tri_prefix}0, ${tri_prefix}1)
    );
    Real3 transformed_tri1 = make3(0.0f, 0.0f, 0.0f);
    Real3 transformed_tri2 = transform3(
        Vnorm, Vstrike, Vdip, sub3(${tri_prefix}2, ${tri_prefix}1)
    );

    Real3 e12 = normalize3(sub3(transformed_tri1, transformed_tri0));
    Real3 e13 = normalize3(sub3(transformed_tri2, transformed_tri0));
    Real3 e23 = normalize3(sub3(transformed_tri2, transformed_tri1));

    Real A = acos(dot3(e12, e13));
    Real B = acos(dot3(negate3(e12), e23));
    Real C = acos(dot3(e23, e13));

    int mode = trimodefinder(
        transformed_obs,
        transformed_tri0, transformed_tri1, transformed_tri2
    );
</%def>

<%def name="disp_hs(tri_prefix)">
    Real3 summed_terms;
    {
        // Main dislocation
        ${disp_fs(tri_prefix)}

        summed_terms = full_out;

        // Harmonic free surface correction
        Real3 efcs_slip = inv_transform3(Vnorm, Vstrike, Vdip, slip);
        Real3 uvw0 = AngSetupFSC(obs, efcs_slip, ${tri_prefix}0, ${tri_prefix}1, nu);
        Real3 uvw1 = AngSetupFSC(obs, efcs_slip, ${tri_prefix}1, ${tri_prefix}2, nu);
        Real3 uvw2 = AngSetupFSC(obs, efcs_slip, ${tri_prefix}2, ${tri_prefix}0, nu);
        Real3 fsc_term = add3(add3(uvw0, uvw1), uvw2);

        summed_terms = add3(fsc_term, summed_terms);
    }
    {
        Real3 image_tri0 = tri0;
        Real3 image_tri1 = tri1;
        Real3 image_tri2 = tri2;

        image_tri0.z *= -1;
        image_tri1.z *= -1;
        image_tri2.z *= -1;

        // Image dislocation
        ${disp_fs("image_tri", "true")}

        summed_terms = add3(summed_terms, full_out);
    }
    Real3 full_out = summed_terms;
</%def>

<%def name="strain_hs(tri_prefix)">
    Real6 summed_terms;
    {
        // Main dislocation
        ${strain_fs(tri_prefix)}

        summed_terms = full_out;

        // Harmonic free surface correction
        Real3 efcs_slip = inv_transform3(Vnorm, Vstrike, Vdip, slip);
        Real6 uvw0 = AngSetupFSC_S(obs, efcs_slip, ${tri_prefix}0, ${tri_prefix}1, nu);
        Real6 uvw1 = AngSetupFSC_S(obs, efcs_slip, ${tri_prefix}1, ${tri_prefix}2, nu);
        Real6 uvw2 = AngSetupFSC_S(obs, efcs_slip, ${tri_prefix}2, ${tri_prefix}0, nu);
        Real6 fsc_term = add6(add6(uvw0, uvw1), uvw2);

        summed_terms = add6(fsc_term, summed_terms);
    }
    {
        Real3 image_tri0 = tri0;
        Real3 image_tri1 = tri1;
        Real3 image_tri2 = tri2;

        image_tri0.z *= -1;
        image_tri1.z *= -1;
        image_tri2.z *= -1;

        // Image dislocation
        ${strain_fs("image_tri", "true")}

        summed_terms = add6(summed_terms, full_out);
    }
    Real6 full_out = summed_terms;
</%def>
