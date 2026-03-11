; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64"

@global_smem = external addrspace(3) global [0 x i8], align 16

define ptx_kernel void @row_sum_broken_kernel(ptr addrspace(1) %0, ptr addrspace(1) %1, i32 %2, i32 %3, ptr addrspace(1) readnone captures(none) %4) local_unnamed_addr !dbg !6 {
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !dbg !9
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !dbg !10
  %8 = and i32 %7, 31, !dbg !10
  %9 = lshr i32 %7, 5, !dbg !10
  %10 = and i32 %7, 127, !dbg !10
  %11 = add i32 %3, -1, !dbg !11
  %12 = icmp sgt i32 %11, %10, !dbg !12
  %13 = mul i32 %2, %6, !dbg !13
  %14 = sext i32 %13 to i64, !dbg !14
  %15 = getelementptr float, ptr addrspace(1) %0, i64 %14, !dbg !14
  %16 = zext nneg i32 %10 to i64, !dbg !15
  %17 = getelementptr float, ptr addrspace(1) %15, i64 %16, !dbg !15
  %18 = tail call i32 asm sideeffect "mov.u32 $0, $1;\0A\09@$3 ld.global.b32 { $0 }, [ $2 + 0 ];", "=r,r,l,b"(i32 0, ptr addrspace(1) %17, i1 %12) #3, !dbg !16
  %19 = bitcast i32 %18 to float, !dbg !16
  %20 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %18, i32 16, i32 31), !dbg !17
  %21 = bitcast i32 %20 to float, !dbg !17
  %22 = fadd float %19, %21, !dbg !21
  %23 = bitcast float %22 to i32, !dbg !17
  %24 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %23, i32 8, i32 31), !dbg !17
  %25 = bitcast i32 %24 to float, !dbg !17
  %26 = fadd float %22, %25, !dbg !21
  %27 = bitcast float %26 to i32, !dbg !17
  %28 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %27, i32 4, i32 31), !dbg !17
  %29 = bitcast i32 %28 to float, !dbg !17
  %30 = fadd float %26, %29, !dbg !21
  %31 = bitcast float %30 to i32, !dbg !17
  %32 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %31, i32 2, i32 31), !dbg !17
  %33 = bitcast i32 %32 to float, !dbg !17
  %34 = fadd float %30, %33, !dbg !21
  %35 = bitcast float %34 to i32, !dbg !17
  %36 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %35, i32 1, i32 31), !dbg !17
  %37 = bitcast i32 %36 to float, !dbg !17
  %38 = fadd float %34, %37, !dbg !21
  %39 = and i32 %9, 3, !dbg !17
  %40 = icmp eq i32 %8, 0, !dbg !17
  %41 = zext nneg i32 %39 to i64, !dbg !17
  %42 = getelementptr float, ptr addrspace(3) @global_smem, i64 %41, !dbg !17
  %43 = bitcast float %38 to <1 x i32>, !dbg !17
  tail call void asm sideeffect "@$2 st.shared.b32 [ $0 + 0 ], $1;", "r,r,b"(ptr addrspace(3) %42, <1 x i32> %43, i1 %40) #3, !dbg !17
  tail call void @llvm.nvvm.barrier0(), !dbg !17
  %44 = icmp slt i32 %7, 4, !dbg !17
  %45 = sext i32 %7 to i64, !dbg !17
  %46 = getelementptr float, ptr addrspace(3) @global_smem, i64 %45, !dbg !17
  %47 = tail call i32 asm sideeffect "@$2 ld.shared.b32 $0, [ $1 + 0 ];", "=r,r,b"(ptr addrspace(3) %46, i1 %44) #3, !dbg !17
  %48 = bitcast i32 %47 to float, !dbg !17
  %49 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %47, i32 2, i32 31), !dbg !17
  %50 = bitcast i32 %49 to float, !dbg !17
  %51 = fadd float %48, %50, !dbg !21
  %52 = bitcast float %51 to i32, !dbg !17
  %53 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %52, i32 1, i32 31), !dbg !17
  %54 = bitcast i32 %53 to float, !dbg !17
  %55 = fadd float %51, %54, !dbg !21
  %56 = and i32 %7, 3, !dbg !17
  %57 = icmp eq i32 %56, 0, !dbg !17
  %58 = and i1 %44, %57, !dbg !17
  %59 = bitcast float %55 to <1 x i32>, !dbg !17
  tail call void asm sideeffect "@$2 st.shared.b32 [ $0 + 0 ], $1;", "r,r,b"(ptr addrspace(3) %46, <1 x i32> %59, i1 %58) #3, !dbg !17
  tail call void @llvm.nvvm.barrier0(), !dbg !17
  %60 = load i32, ptr addrspace(3) @global_smem, align 16, !dbg !17
  %61 = sext i32 %6 to i64, !dbg !23
  %62 = getelementptr float, ptr addrspace(1) %1, i64 %61, !dbg !23
  %63 = icmp eq i32 %7, 0, !dbg !24
  tail call void asm sideeffect "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b"(i32 %60, ptr addrspace(1) %62, i1 %63) #3, !dbg !24
  ret void, !dbg !25
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.bfly.i32(i32, i32, i32, i32) #1

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier0() #2

attributes #0 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
attributes #2 = { convergent nocallback nounwind }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}
!nvvm.annotations = !{!4}
!llvm.ident = !{!5}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 1}
!2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "triton", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!3 = !DIFile(filename: "triton_row_sum_broken_kernel.py", directory: "/shared_folders/querylake_server/side_projects/gpu_rl/tests/tmp_sft/workloads/reference")
!4 = !{ptr @row_sum_broken_kernel, !"reqntidx", i32 128}
!5 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!6 = distinct !DISubprogram(name: "row_sum_broken_kernel", linkageName: "row_sum_broken_kernel", scope: !3, file: !3, line: 9, type: !7, scopeLine: 9, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!7 = !DISubroutineType(cc: DW_CC_normal, types: !8)
!8 = !{}
!9 = !DILocation(line: 10, column: 24, scope: !6)
!10 = !DILocation(line: 11, column: 24, scope: !6)
!11 = !DILocation(line: 12, column: 41, scope: !6)
!12 = !DILocation(line: 13, column: 18, scope: !6)
!13 = !DILocation(line: 14, column: 28, scope: !6)
!14 = !DILocation(line: 14, column: 22, scope: !6)
!15 = !DILocation(line: 14, column: 41, scope: !6)
!16 = !DILocation(line: 15, column: 21, scope: !6)
!17 = !DILocation(line: 286, column: 36, scope: !18, inlinedAt: !20)
!18 = distinct !DILexicalBlockFile(scope: !6, file: !19, discriminator: 0)
!19 = !DIFile(filename: "standard.py", directory: "/home/querylake_manager/miniconda3/lib/python3.12/site-packages/triton/language")
!20 = !DILocation(line: 16, column: 19, scope: !6)
!21 = !DILocation(line: 256, column: 15, scope: !22, inlinedAt: !20)
!22 = distinct !DILexicalBlockFile(scope: !18, file: !19, discriminator: 0)
!23 = !DILocation(line: 17, column: 23, scope: !6)
!24 = !DILocation(line: 17, column: 28, scope: !6)
!25 = !DILocation(line: 17, column: 4, scope: !6)
