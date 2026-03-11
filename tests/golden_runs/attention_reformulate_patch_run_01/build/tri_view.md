# Source / PTX / SASS Tri-View

- backend: `triton_nvidia`
- correlation_method: `ptx_loc_source_map_v1`
- source_path: `/shared_folders/querylake_server/side_projects/gpu_rl/tests/tmp_transition_goldens/workloads/reference/triton_attention_score_kernel.py`
- source_ref: `build/source.txt`
- ttir_ref: `build/ttir.mlir`
- ttgir_ref: `build/ttgir.mlir`
- llir_ref: `build/llir.ll`
- ptx_ref: `build/ptx.txt`
- sass_ref: `build/sass.txt`

## Preview
- line 11: source=`def attention_score_kernel(` ptx=`$L__func_begin0:` sass=`--:-:-:-:2	MOV R1, c[0x0][0x28];`
- line 11: source=`def attention_score_kernel(` ptx=`ld.param.u64 	%rd5, [attention_score_kernel_param_0];` sass=`--:-:0:-:1	S2R R17, SR_TID.X;`
- line 30: source=`row = tl.program_id(0)` ptx=`mov.u32 	%r30, %ctaid.x;` sass=`--:-:-:-:1	MOV R16, 0x4;`
- line 31: source=`col_block = tl.program_id(1)` ptx=`mov.u32 	%r31, %ctaid.y;` sass=`--:-:-:-:1	CS2R R8, SRZ;`
- line 32: source=`cols = col_block * BLOCK_K + tl.arange(0, BLOCK_K)` ptx=`shl.b32 	%r32, %r31, 5;` sass=`--:-:-:-:1	CS2R R10, SRZ;`
- line 32: source=`cols = col_block * BLOCK_K + tl.arange(0, BLOCK_K)` ptx=`mov.u32 	%r35, %tid.x;` sass=`--:-:1:-:1	S2R R23, SR_CTAID.Y;`
- line 32: source=`cols = col_block * BLOCK_K + tl.arange(0, BLOCK_K)` ptx=`or.b32  	%r45, %r44, 16;` sass=`--:-:-:-:1	CS2R R12, SRZ;`
- line 35: source=`q_ptrs = q_ptr + row * q_row_stride + dims * q_col_stride` ptx=`mul.lo.s32 	%r47, %r33, %r30;` sass=`--:-:-:-:1	CS2R R14, SRZ;`
- line 35: source=`q_ptrs = q_ptr + row * q_row_stride + dims * q_col_stride` ptx=`mul.wide.s32 	%rd8, %r47, 4;` sass=`--:-:-:-:1	ULDC.64 UR4, c[0x0][0x118];`
- line 35: source=`q_ptrs = q_ptr + row * q_row_stride + dims * q_col_stride` ptx=`mul.wide.u32 	%rd10, %r42, 4;` sass=`--:-:2:-:1	S2R R0, SR_CTAID.X;`

## Warnings
- none
