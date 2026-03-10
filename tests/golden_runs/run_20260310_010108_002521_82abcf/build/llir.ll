; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64"

@global_smem = external addrspace(3) global [0 x i8], align 16

define ptx_kernel void @row_sum_repaired_kernel(ptr addrspace(1) %0, ptr addrspace(1) %1, i32 %2, i32 %3, ptr addrspace(1) readnone captures(none) %4) local_unnamed_addr !dbg !6 {
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !dbg !9
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !dbg !10
  %8 = and i32 %7, 31, !dbg !10
  %9 = lshr i32 %7, 5, !dbg !10
  %10 = and i32 %7, 127, !dbg !10
  %11 = icmp slt i32 %10, %3, !dbg !11
  %12 = mul i32 %2, %6, !dbg !12
  %13 = sext i32 %12 to i64, !dbg !13
  %14 = getelementptr float, ptr addrspace(1) %0, i64 %13, !dbg !13
  %15 = zext nneg i32 %10 to i64, !dbg !14
  %16 = getelementptr float, ptr addrspace(1) %14, i64 %15, !dbg !14
  %17 = tail call i32 asm sideeffect "mov.u32 $0, $1;\0A\09@$3 ld.global.b32 { $0 }, [ $2 + 0 ];", "=r,r,l,b"(i32 0, ptr addrspace(1) %16, i1 %11) #3, !dbg !15
  %18 = bitcast i32 %17 to float, !dbg !15
  %19 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %17, i32 16, i32 31), !dbg !16
  %20 = bitcast i32 %19 to float, !dbg !16
  %21 = fadd float %18, %20, !dbg !20
  %22 = bitcast float %21 to i32, !dbg !16
  %23 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %22, i32 8, i32 31), !dbg !16
  %24 = bitcast i32 %23 to float, !dbg !16
  %25 = fadd float %21, %24, !dbg !20
  %26 = bitcast float %25 to i32, !dbg !16
  %27 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %26, i32 4, i32 31), !dbg !16
  %28 = bitcast i32 %27 to float, !dbg !16
  %29 = fadd float %25, %28, !dbg !20
  %30 = bitcast float %29 to i32, !dbg !16
  %31 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %30, i32 2, i32 31), !dbg !16
  %32 = bitcast i32 %31 to float, !dbg !16
  %33 = fadd float %29, %32, !dbg !20
  %34 = bitcast float %33 to i32, !dbg !16
  %35 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %34, i32 1, i32 31), !dbg !16
  %36 = bitcast i32 %35 to float, !dbg !16
  %37 = fadd float %33, %36, !dbg !20
  %38 = and i32 %9, 3, !dbg !16
  %39 = icmp eq i32 %8, 0, !dbg !16
  %40 = zext nneg i32 %38 to i64, !dbg !16
  %41 = getelementptr float, ptr addrspace(3) @global_smem, i64 %40, !dbg !16
  %42 = bitcast float %37 to <1 x i32>, !dbg !16
  tail call void asm sideeffect "@$2 st.shared.b32 [ $0 + 0 ], $1;", "r,r,b"(ptr addrspace(3) %41, <1 x i32> %42, i1 %39) #3, !dbg !16
  tail call void @llvm.nvvm.barrier0(), !dbg !16
  %43 = icmp slt i32 %7, 4, !dbg !16
  %44 = sext i32 %7 to i64, !dbg !16
  %45 = getelementptr float, ptr addrspace(3) @global_smem, i64 %44, !dbg !16
  %46 = tail call i32 asm sideeffect "@$2 ld.shared.b32 $0, [ $1 + 0 ];", "=r,r,b"(ptr addrspace(3) %45, i1 %43) #3, !dbg !16
  %47 = bitcast i32 %46 to float, !dbg !16
  %48 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %46, i32 2, i32 31), !dbg !16
  %49 = bitcast i32 %48 to float, !dbg !16
  %50 = fadd float %47, %49, !dbg !20
  %51 = bitcast float %50 to i32, !dbg !16
  %52 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %51, i32 1, i32 31), !dbg !16
  %53 = bitcast i32 %52 to float, !dbg !16
  %54 = fadd float %50, %53, !dbg !20
  %55 = and i32 %7, 3, !dbg !16
  %56 = icmp eq i32 %55, 0, !dbg !16
  %57 = and i1 %43, %56, !dbg !16
  %58 = bitcast float %54 to <1 x i32>, !dbg !16
  tail call void asm sideeffect "@$2 st.shared.b32 [ $0 + 0 ], $1;", "r,r,b"(ptr addrspace(3) %45, <1 x i32> %58, i1 %57) #3, !dbg !16
  tail call void @llvm.nvvm.barrier0(), !dbg !16
  %59 = load i32, ptr addrspace(3) @global_smem, align 16, !dbg !16
  %60 = sext i32 %6 to i64, !dbg !22
  %61 = getelementptr float, ptr addrspace(1) %1, i64 %60, !dbg !22
  %62 = icmp eq i32 %7, 0, !dbg !23
  tail call void asm sideeffect "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b"(i32 %59, ptr addrspace(1) %61, i1 %62) #3, !dbg !23
  ret void, !dbg !24
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
!3 = !DIFile(filename: "triton_row_sum_repaired_kernel.py", directory: "/shared_folders/querylake_server/side_projects/gpu_rl/workloads/reference")
!4 = !{ptr @row_sum_repaired_kernel, !"reqntidx", i32 128}
!5 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!6 = distinct !DISubprogram(name: "row_sum_repaired_kernel", linkageName: "row_sum_repaired_kernel", scope: !3, file: !3, line: 9, type: !7, scopeLine: 9, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!7 = !DISubroutineType(cc: DW_CC_normal, types: !8)
!8 = !{}
!9 = !DILocation(line: 10, column: 24, scope: !6)
!10 = !DILocation(line: 11, column: 24, scope: !6)
!11 = !DILocation(line: 12, column: 18, scope: !6)
!12 = !DILocation(line: 13, column: 28, scope: !6)
!13 = !DILocation(line: 13, column: 22, scope: !6)
!14 = !DILocation(line: 13, column: 41, scope: !6)
!15 = !DILocation(line: 14, column: 21, scope: !6)
!16 = !DILocation(line: 286, column: 36, scope: !17, inlinedAt: !19)
!17 = distinct !DILexicalBlockFile(scope: !6, file: !18, discriminator: 0)
!18 = !DIFile(filename: "standard.py", directory: "/home/querylake_manager/miniconda3/lib/python3.12/site-packages/triton/language")
!19 = !DILocation(line: 15, column: 19, scope: !6)
!20 = !DILocation(line: 256, column: 15, scope: !21, inlinedAt: !19)
!21 = distinct !DILexicalBlockFile(scope: !17, file: !18, discriminator: 0)
!22 = !DILocation(line: 16, column: 23, scope: !6)
!23 = !DILocation(line: 16, column: 28, scope: !6)
!24 = !DILocation(line: 16, column: 4, scope: !6)
