<%!
def comp(d):
    return ["x", "y", "z", "a", "b", "c"][d]
%>
<%namespace name="common" file="common.cu"/>

${common.defs(preamble, float_type)}

<%def name="pairs(name, evaluator, vec_dim)">
KERNEL
void pairs_${name}(GLOBAL_MEM Real* results, int n_pairs,
    GLOBAL_MEM Real* obs_pts, GLOBAL_MEM Real* tris,
    GLOBAL_MEM Real* slips, Real nu)
{
    int i = get_global_id(0);
    if (i >= n_pairs) {
        return;
    }
    Real3 obs;
    % for d1 in range(3):
        Real3 tri${d1};
        obs.${comp(d1)} = obs_pts[i * 3 + ${d1}];
        % for d2 in range(3):
            tri${d1}.${comp(d2)} = tris[i * 9 + ${d1} * 3 + ${d2}];
        % endfor
    % endfor

    Real3 slip = make3(
        slips[i * 3 + 2],
        slips[i * 3 + 0],
        slips[i * 3 + 1]
    );

    ${evaluator("tri")}

    %for d in range(vec_dim):
        results[i * ${vec_dim} + ${d}] = full_out.${comp(d)};
    %endfor
}
</%def>

${pairs("disp_fs", common.disp_fs, 3)}
${pairs("disp_hs", common.disp_hs, 3)}
${pairs("strain_fs", common.strain_fs, 6)}
${pairs("strain_hs", common.strain_hs, 6)}
