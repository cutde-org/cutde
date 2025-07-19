<%!
def comp(d):
    return ["x", "y", "z", "a", "b", "c"][d]
%>
<%namespace name="common" file="common.cu"/>

${common.defs(preamble, float_type)}

<%def name="tde_blocks(name, evaluator, vec_dim)">
KERNEL
void blocks_${name}(GLOBAL_MEM Real* results,
    GLOBAL_MEM Real* obs_pts, GLOBAL_MEM Real* tris,
    GLOBAL_MEM int* obs_start, GLOBAL_MEM int* obs_end,
    GLOBAL_MEM int* src_start, GLOBAL_MEM int* src_end,
    GLOBAL_MEM int* block_start,
    Real nu)
{
    int block_idx = get_group_id(0);
    int team_id = get_local_id(0);
    int team_size = get_local_size(0);

    int os = obs_start[block_idx];
    int oe = obs_end[block_idx];

    int ss = src_start[block_idx];
    int se = src_end[block_idx];
    int n_src = se - ss;

    int bs = block_start[block_idx];

    for (int i = os + team_id; i < oe; i += team_size) {
        int obs_idx = i - os;

        Real3 obs;
        % for d1 in range(3):
            obs.${comp(d1)} = obs_pts[i * 3 + ${d1}];
        % endfor

        for (int j = ss; j < se; j++) {
            int src_idx = j - ss;

            % for d1 in range(3):
                Real3 tri${d1};
                % for d2 in range(3):
                    tri${d1}.${comp(d2)} = tris[j * 9 + ${d1} * 3 + ${d2}];
                % endfor
            % endfor
            % for d_src in range(3):
            {
                Real3 slip = make3(0.0, 0.0, 0.0);
                slip.${comp([1,2,0][d_src])} = 1.0;

                ${evaluator("tri")}

                %for d_obs in range(vec_dim):
                {
                    int idx = bs + (
                        (obs_idx * ${vec_dim} + ${d_obs}) * n_src + src_idx
                    ) * 3 + ${d_src};
                    results[idx] = full_out.${comp(d_obs)};
                }
                %endfor
            }
            % endfor
        }
    }
}
</%def>

${tde_blocks("disp_fs", common.disp_fs, 3)}
${tde_blocks("disp_hs", common.disp_hs, 3)}
${tde_blocks("strain_fs", common.strain_fs, 6)}
${tde_blocks("strain_hs", common.strain_hs, 6)}
