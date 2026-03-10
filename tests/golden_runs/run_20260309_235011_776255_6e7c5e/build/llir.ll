; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64"

@global_smem = external addrspace(3) global [0 x i8], align 16

define ptx_kernel void @attention_score_kernel(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8, float %9, float %10, ptr addrspace(1) readnone captures(none) %11) local_unnamed_addr !dbg !6 {
  %13 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !dbg !9
  %14 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y(), !dbg !10
  %15 = shl i32 %14, 5, !dbg !11
  %16 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !dbg !12
  %.lobit1 = lshr i32 %16, 3, !dbg !12
  %17 = and i32 %.lobit1, 15, !dbg !12
  %18 = or disjoint i32 %17, 16, !dbg !12
  %19 = shl i32 %16, 2, !dbg !12
  %20 = and i32 %19, 28, !dbg !12
  %21 = and i32 %16, 31, !dbg !12
  %22 = or disjoint i32 %17, %15, !dbg !13
  %23 = or disjoint i32 %18, %15, !dbg !13
  %24 = or disjoint i32 %15, %21, !dbg !13
  %25 = mul i32 %3, %13, !dbg !14
  %26 = sext i32 %25 to i64, !dbg !15
  %27 = getelementptr float, ptr addrspace(1) %0, i64 %26, !dbg !15
  %28 = zext nneg i32 %20 to i64, !dbg !16
  %29 = getelementptr float, ptr addrspace(1) %27, i64 %28, !dbg !16
  %30 = icmp slt i32 %20, %8, !dbg !17
  %31 = tail call { i32, i32, i32, i32 } asm sideeffect "mov.u32 $0, $4;\0A\09mov.u32 $1, $5;\0A\09mov.u32 $2, $6;\0A\09mov.u32 $3, $7;\0A\09@$9 ld.global.v4.b32 { $0, $1, $2, $3 }, [ $8 + 0 ];", "=r,=r,=r,=r,r,r,r,r,l,b"(i32 0, i32 0, i32 0, i32 0, ptr addrspace(1) %29, i1 %30) #3, !dbg !18
  %32 = extractvalue { i32, i32, i32, i32 } %31, 0, !dbg !18
  %33 = extractvalue { i32, i32, i32, i32 } %31, 1, !dbg !18
  %34 = extractvalue { i32, i32, i32, i32 } %31, 2, !dbg !18
  %35 = extractvalue { i32, i32, i32, i32 } %31, 3, !dbg !18
  %36 = bitcast i32 %32 to float, !dbg !18
  %37 = bitcast i32 %33 to float, !dbg !18
  %38 = bitcast i32 %34 to float, !dbg !18
  %39 = bitcast i32 %35 to float, !dbg !18
  %40 = mul i32 %4, %22, !dbg !19
  %41 = mul i32 %4, %23, !dbg !19
  %42 = sext i32 %40 to i64, !dbg !20
  %43 = getelementptr float, ptr addrspace(1) %1, i64 %42, !dbg !20
  %44 = sext i32 %41 to i64, !dbg !20
  %45 = getelementptr float, ptr addrspace(1) %1, i64 %44, !dbg !20
  %46 = getelementptr float, ptr addrspace(1) %43, i64 %28, !dbg !21
  %47 = getelementptr float, ptr addrspace(1) %45, i64 %28, !dbg !21
  %48 = icmp slt i32 %22, %7, !dbg !22
  %49 = icmp slt i32 %23, %7, !dbg !22
  %50 = and i1 %48, %30, !dbg !23
  %51 = and i1 %49, %30, !dbg !23
  %52 = tail call { i32, i32, i32, i32 } asm sideeffect "mov.u32 $0, $4;\0A\09mov.u32 $1, $5;\0A\09mov.u32 $2, $6;\0A\09mov.u32 $3, $7;\0A\09@$9 ld.global.v4.b32 { $0, $1, $2, $3 }, [ $8 + 0 ];", "=r,=r,=r,=r,r,r,r,r,l,b"(i32 0, i32 0, i32 0, i32 0, ptr addrspace(1) %46, i1 %50) #3, !dbg !24
  %53 = extractvalue { i32, i32, i32, i32 } %52, 0, !dbg !24
  %54 = extractvalue { i32, i32, i32, i32 } %52, 1, !dbg !24
  %55 = extractvalue { i32, i32, i32, i32 } %52, 2, !dbg !24
  %56 = extractvalue { i32, i32, i32, i32 } %52, 3, !dbg !24
  %57 = bitcast i32 %53 to float, !dbg !24
  %58 = bitcast i32 %54 to float, !dbg !24
  %59 = bitcast i32 %55 to float, !dbg !24
  %60 = bitcast i32 %56 to float, !dbg !24
  %61 = tail call { i32, i32, i32, i32 } asm sideeffect "mov.u32 $0, $4;\0A\09mov.u32 $1, $5;\0A\09mov.u32 $2, $6;\0A\09mov.u32 $3, $7;\0A\09@$9 ld.global.v4.b32 { $0, $1, $2, $3 }, [ $8 + 0 ];", "=r,=r,=r,=r,r,r,r,r,l,b"(i32 0, i32 0, i32 0, i32 0, ptr addrspace(1) %47, i1 %51) #3, !dbg !24
  %62 = extractvalue { i32, i32, i32, i32 } %61, 0, !dbg !24
  %63 = extractvalue { i32, i32, i32, i32 } %61, 1, !dbg !24
  %64 = extractvalue { i32, i32, i32, i32 } %61, 2, !dbg !24
  %65 = extractvalue { i32, i32, i32, i32 } %61, 3, !dbg !24
  %66 = bitcast i32 %62 to float, !dbg !24
  %67 = bitcast i32 %63 to float, !dbg !24
  %68 = bitcast i32 %64 to float, !dbg !24
  %69 = bitcast i32 %65 to float, !dbg !24
  %70 = fmul float %36, %57, !dbg !25
  %71 = fmul float %37, %58, !dbg !25
  %72 = fmul float %38, %59, !dbg !25
  %73 = fmul float %39, %60, !dbg !25
  %74 = fmul float %36, %66, !dbg !25
  %75 = fmul float %37, %67, !dbg !25
  %76 = fmul float %38, %68, !dbg !25
  %77 = fmul float %39, %69, !dbg !25
  %78 = fadd float %70, %71, !dbg !26
  %79 = fadd float %72, %78, !dbg !26
  %80 = fadd float %73, %79, !dbg !26
  %81 = fadd float %74, %75, !dbg !26
  %82 = fadd float %76, %81, !dbg !26
  %83 = fadd float %77, %82, !dbg !26
  %84 = bitcast float %80 to i32, !dbg !31
  %85 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %84, i32 4, i32 31), !dbg !31
  %86 = bitcast i32 %85 to float, !dbg !31
  %87 = fadd float %80, %86, !dbg !26
  %88 = bitcast float %87 to i32, !dbg !31
  %89 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %88, i32 2, i32 31), !dbg !31
  %90 = bitcast i32 %89 to float, !dbg !31
  %91 = fadd float %87, %90, !dbg !26
  %92 = bitcast float %91 to i32, !dbg !31
  %93 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %92, i32 1, i32 31), !dbg !31
  %94 = bitcast i32 %93 to float, !dbg !31
  %95 = fadd float %91, %94, !dbg !26
  %96 = bitcast float %83 to i32, !dbg !31
  %97 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %96, i32 4, i32 31), !dbg !31
  %98 = bitcast i32 %97 to float, !dbg !31
  %99 = fadd float %83, %98, !dbg !26
  %100 = bitcast float %99 to i32, !dbg !31
  %101 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %100, i32 2, i32 31), !dbg !31
  %102 = bitcast i32 %101 to float, !dbg !31
  %103 = fadd float %99, %102, !dbg !26
  %104 = bitcast float %103 to i32, !dbg !31
  %105 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %104, i32 1, i32 31), !dbg !31
  %106 = bitcast i32 %105 to float, !dbg !31
  %107 = fadd float %103, %106, !dbg !26
  %108 = fmul float %9, %95, !dbg !32
  %109 = fmul float %9, %107, !dbg !32
  %.not = icmp sgt i32 %22, %13, !dbg !33
  %.not2 = icmp sgt i32 %23, %13, !dbg !33
  %110 = select i1 %.not, float %10, float %108, !dbg !34
  %111 = select i1 %.not2, float %10, float %109, !dbg !34
  %112 = mul i32 %5, %13, !dbg !35
  %113 = sext i32 %112 to i64, !dbg !36
  %114 = getelementptr float, ptr addrspace(1) %2, i64 %113, !dbg !36
  %115 = sext i32 %24 to i64, !dbg !37
  %116 = getelementptr float, ptr addrspace(1) %114, i64 %115, !dbg !37
  %117 = icmp slt i32 %13, %6, !dbg !38
  %118 = icmp slt i32 %24, %7, !dbg !39
  %119 = and i1 %117, %118, !dbg !40
  %120 = zext nneg i32 %17 to i64, !dbg !41
  %121 = getelementptr inbounds nuw float, ptr addrspace(3) @global_smem, i64 %120, !dbg !41
  %122 = bitcast float %110 to <1 x i32>, !dbg !41
  tail call void asm sideeffect "@$2 st.shared.b32 [ $0 + 0 ], $1;", "r,r,b"(ptr addrspace(3) %121, <1 x i32> %122, i1 true) #3, !dbg !41
  %123 = zext nneg i32 %18 to i64, !dbg !41
  %124 = getelementptr inbounds nuw float, ptr addrspace(3) @global_smem, i64 %123, !dbg !41
  %125 = bitcast float %111 to <1 x i32>, !dbg !41
  tail call void asm sideeffect "@$2 st.shared.b32 [ $0 + 0 ], $1;", "r,r,b"(ptr addrspace(3) nonnull %124, <1 x i32> %125, i1 true) #3, !dbg !41
  tail call void @llvm.nvvm.barrier0(), !dbg !41
  %126 = zext nneg i32 %21 to i64, !dbg !41
  %127 = getelementptr inbounds nuw float, ptr addrspace(3) @global_smem, i64 %126, !dbg !41
  %128 = load i32, ptr addrspace(3) %127, align 4, !dbg !41
  %129 = and i32 %16, 96, !dbg !41
  %130 = icmp eq i32 %129, 0, !dbg !41
  %131 = and i1 %130, %119, !dbg !41
  tail call void asm sideeffect "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b"(i32 %128, ptr addrspace(1) %116, i1 %131) #3, !dbg !41
  ret void, !dbg !42
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #0

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
!3 = !DIFile(filename: "triton_attention_score_kernel.py", directory: "/shared_folders/querylake_server/side_projects/gpu_rl/workloads/reference")
!4 = !{ptr @attention_score_kernel, !"reqntidx", i32 128}
!5 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!6 = distinct !DISubprogram(name: "attention_score_kernel", linkageName: "attention_score_kernel", scope: !3, file: !3, line: 11, type: !7, scopeLine: 11, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!7 = !DISubroutineType(cc: DW_CC_normal, types: !8)
!8 = !{}
!9 = !DILocation(line: 30, column: 24, scope: !6)
!10 = !DILocation(line: 31, column: 30, scope: !6)
!11 = !DILocation(line: 32, column: 23, scope: !6)
!12 = !DILocation(line: 32, column: 46, scope: !6)
!13 = !DILocation(line: 32, column: 33, scope: !6)
!14 = !DILocation(line: 35, column: 27, scope: !6)
!15 = !DILocation(line: 35, column: 21, scope: !6)
!16 = !DILocation(line: 35, column: 42, scope: !6)
!17 = !DILocation(line: 36, column: 43, scope: !6)
!18 = !DILocation(line: 36, column: 23, scope: !6)
!19 = !DILocation(line: 38, column: 37, scope: !6)
!20 = !DILocation(line: 38, column: 21, scope: !6)
!21 = !DILocation(line: 38, column: 52, scope: !6)
!22 = !DILocation(line: 39, column: 53, scope: !6)
!23 = !DILocation(line: 39, column: 61, scope: !6)
!24 = !DILocation(line: 39, column: 23, scope: !6)
!25 = !DILocation(line: 40, column: 31, scope: !6)
!26 = !DILocation(line: 256, column: 15, scope: !27, inlinedAt: !30)
!27 = distinct !DILexicalBlockFile(scope: !29, file: !28, discriminator: 0)
!28 = !DIFile(filename: "standard.py", directory: "/home/querylake_manager/miniconda3/lib/python3.12/site-packages/triton/language")
!29 = distinct !DILexicalBlockFile(scope: !6, file: !28, discriminator: 0)
!30 = !DILocation(line: 40, column: 20, scope: !6)
!31 = !DILocation(line: 286, column: 36, scope: !29, inlinedAt: !30)
!32 = !DILocation(line: 40, column: 60, scope: !6)
!33 = !DILocation(line: 42, column: 34, scope: !6)
!34 = !DILocation(line: 42, column: 47, scope: !6)
!35 = !DILocation(line: 44, column: 31, scope: !6)
!36 = !DILocation(line: 44, column: 25, scope: !6)
!37 = !DILocation(line: 44, column: 48, scope: !6)
!38 = !DILocation(line: 45, column: 43, scope: !6)
!39 = !DILocation(line: 45, column: 58, scope: !6)
!40 = !DILocation(line: 45, column: 51, scope: !6)
!41 = !DILocation(line: 45, column: 23, scope: !6)
!42 = !DILocation(line: 45, column: 4, scope: !6)
