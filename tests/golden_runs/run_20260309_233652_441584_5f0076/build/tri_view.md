# Source / PTX / SASS Tri-View

- backend: `triton_nvidia`
- correlation_method: `ptx_loc_source_map_v1`
- source_path: `/shared_folders/querylake_server/side_projects/gpu_rl/tests/tmp_rewrite_check/workloads/reference/triton_row_sum_repaired_kernel.py`
- source_ref: `build/source.txt`
- ttir_ref: `build/ttir.mlir`
- ttgir_ref: `build/ttgir.mlir`
- llir_ref: `build/llir.ll`
- ptx_ref: `build/ptx.txt`
- sass_ref: `build/sass.txt`

## Preview
- line 9: source=`def row_sum_repaired_kernel(x_ptr, out_ptr, stride_row, n_co` ptx=`$L__func_begin0:` sass=`--:-:-:-:2	MOV R1, c[0x0][0x28];`
- line 9: source=`def row_sum_repaired_kernel(x_ptr, out_ptr, stride_row, n_co` ptx=`ld.param.u64 	%rd3, [row_sum_repaired_kernel_param_0];` sass=`--:-:0:-:1	S2R R9, SR_TID.X;`
- line 10: source=`row = tl.program_id(0)` ptx=`mov.u32 	%r10, %ctaid.x;` sass=`--:-:-:-:1	MOV R11, 0x4;`
- line 11: source=`cols = tl.arange(0, BLOCK_SIZE)` ptx=`mov.u32 	%r12, %tid.x;` sass=`--:-:-:-:1	ULDC.64 UR4, c[0x0][0x118];`
- line 12: source=`mask = cols < n_cols` ptx=`setp.lt.s32 	%p1, %r15, %r14;` sass=`--:-:-:-:1	BSSY B0, 0xf0;`
- line 13: source=`row_ptr = x_ptr + row * stride_row + cols` ptx=`mul.lo.s32 	%r16, %r11, %r10;` sass=`--:-:1:-:1	S2R R8, SR_CTAID.X;`
- line 13: source=`row_ptr = x_ptr + row * stride_row + cols` ptx=`mul.wide.s32 	%rd5, %r16, 4;` sass=`--:-:-:-:2	MOV R0, RZ;`
- line 13: source=`row_ptr = x_ptr + row * stride_row + cols` ptx=`mul.wide.u32 	%rd7, %r15, 4;` sass=`01:-:-:Y:4	LOP3.LUT R5, R9, 0x7f, RZ, 0xc0, !PT;`
- line 14: source=`values = tl.load(row_ptr, mask=mask, other=0.0)` ptx=`mov.u32 %r1, %r2;` sass=`--:-:-:-:1	ISETP.GE.AND P0, PT, R5, c[0x0][0x174], PT;`
- line 286: source=`` ptx=`shfl.sync.bfly.b32	%r17, %r1, 16, 31, -1;` sass=`02:-:-:Y:4	IMAD R2, R8, c[0x0][0x170], RZ;`

## Warnings
- none
