# Source / PTX / SASS Tri-View

- backend: `triton_nvidia`
- correlation_method: `ptx_loc_source_map_v1`
- source_path: `/shared_folders/querylake_server/side_projects/gpu_rl/tests/tmp_transition_goldens/workloads/reference/triton_row_sum_broken_kernel.py`
- source_ref: `build/source.txt`
- ttir_ref: `build/ttir.mlir`
- ttgir_ref: `build/ttgir.mlir`
- llir_ref: `build/llir.ll`
- ptx_ref: `build/ptx.txt`
- sass_ref: `build/sass.txt`

## Preview
- line 9: source=`def row_sum_broken_kernel(x_ptr, out_ptr, stride_row, n_cols` ptx=`$L__func_begin0:` sass=`--:-:-:-:2	MOV R1, c[0x0][0x28];`
- line 9: source=`def row_sum_broken_kernel(x_ptr, out_ptr, stride_row, n_cols` ptx=`ld.param.u64 	%rd3, [row_sum_broken_kernel_param_0];` sass=`--:-:0:-:1	S2R R9, SR_TID.X;`
- line 10: source=`row = tl.program_id(0)` ptx=`mov.u32 	%r10, %ctaid.x;` sass=`--:-:-:-:1	ULDC UR4, c[0x0][0x174];`
- line 11: source=`cols = tl.arange(0, BLOCK_SIZE)` ptx=`mov.u32 	%r12, %tid.x;` sass=`--:-:-:-:1	MOV R11, 0x4;`
- line 12: source=`effective_cols = tl.maximum(n_cols - 1, 0)` ptx=`add.s32 	%r16, %r14, -1;` sass=`--:-:-:-:1	UIADD3 UR4, UR4, -0x1, URZ;`
- line 13: source=`mask = cols < effective_cols` ptx=`setp.gt.s32 	%p1, %r16, %r15;` sass=`--:-:1:-:1	S2R R8, SR_CTAID.X;`
- line 14: source=`row_ptr = x_ptr + row * stride_row + cols` ptx=`mul.lo.s32 	%r17, %r11, %r10;` sass=`--:-:-:-:1	BSSY B0, 0x110;`
- line 14: source=`row_ptr = x_ptr + row * stride_row + cols` ptx=`mul.wide.s32 	%rd5, %r17, 4;` sass=`--:-:-:-:2	MOV R0, RZ;`
- line 14: source=`row_ptr = x_ptr + row * stride_row + cols` ptx=`mul.wide.u32 	%rd7, %r15, 4;` sass=`01:-:-:Y:4	LOP3.LUT R5, R9, 0x7f, RZ, 0xc0, !PT;`
- line 15: source=`values = tl.load(row_ptr, mask=mask, other=0.0)` ptx=`mov.u32 %r1, %r2;` sass=`--:-:-:-:1	ISETP.LT.AND P0, PT, R5, UR4, PT;`

## Warnings
- none
