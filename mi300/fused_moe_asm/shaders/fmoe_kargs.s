.text
.global kernel_func
.p2align 8
.type kernel_func,@function

kernel_func:

  s_and_b32     s1, s1, 0x0000ffff                      // 000000000000: 8601FF01 0000FFFF
  s_load_dwordx2  s[8:9], s[0:1], 0x00                  // 000000000008: C0060200 00000000
  s_load_dwordx2  s[16:17], s[0:1], 0x10                // 000000000010: C0060400 00000010
  s_load_dwordx2  s[20:21], s[0:1], 0x20                // 000000000018: C0060500 00000020
  s_load_dwordx2  s[12:13], s[0:1], 0x40                // 000000000020: C0060300 00000040
  s_load_dwordx2  s[28:29], s[0:1], 0x90                // 000000000028: C0060700 00000090
  s_load_dwordx2  s[24:25], s[0:1], 0xa0                // 000000000030: C0060600 000000A0
  s_load_dwordx2  s[30:31], s[0:1], 0xb0                // 000000000038: C0060780 000000B0
  s_load_dword  s64, s[0:1], 0xc0                       // 000000000040: C0021000 000000C0
  s_load_dword  s65, s[0:1], 0xd0                       // 000000000048: C0021040 000000D0
  s_load_dword  s66, s[0:1], 0xe0                       // 000000000050: C0021080 000000E0
  s_load_dword  s67, s[0:1], 0xf0                       // 000000000058: C00210C0 000000F0
  s_load_dword  s68, s[0:1], 0x100                      // 000000000060: C0021100 00000100
  s_load_dword  s69, s[0:1], 0x110                      // 000000000068: C0021140 00000110
  s_load_dword  s70, s[0:1], 0x120                      // 000000000070: C0021180 00000120
  s_load_dword  s71, s[0:1], 0x130                      // 000000000078: C00211C0 00000130
  s_load_dword  s72, s[0:1], 0x140                      // 000000000080: C0021200 00000140
  s_load_dword  s73, s[0:1], 0x150                      // 000000000088: C0021240 00000150
  v_lshrrev_b32  v1, 10, v0                             // 000000000090: 2002008A
  v_lshrrev_b32  v2, 10, v1                             // 000000000094: 2004028A
  v_and_b32     v2, 0x000003ff, v2                      // 000000000098: 260404FF 000003FF
  v_and_b32     v1, 0x000003ff, v1                      // 0000000000A0: 260202FF 000003FF
  v_and_b32     v0, 0x000003ff, v0                      // 0000000000A8: 260000FF 000003FF
  v_lshrrev_b32  v3, 6, v0                              // 0000000000B0: 20060086
  v_and_b32     v0, 63, v0                              // 0000000000B4: 260000BF
  s_mov_b32     s2, s2                                  // 0000000000B8: BE820002
  s_mov_b32     s3, s3                                  // 0000000000BC: BE830003
  s_mov_b32     s4, s4                                  // 0000000000C0: BE840004
  v_readfirstlane_b32  s7, v3                           // 0000000000C4: 7E0E0503
  s_waitcnt     lgkmcnt(0)                              // 0000000000C8: BF8CC07F
  s_and_b32     s29, s29, 0x0000ffff                    // 0000000000CC: 861DFF1D 0000FFFF
  s_and_b32     s31, s31, 0x0000ffff                    // 0000000000D4: 861FFF1F 0000FFFF
  s_and_b32     s9, s9, 0x0000ffff                      // 0000000000DC: 8609FF09 0000FFFF
  s_mul_i32     s60, s66, s68                           // 0000000000E4: 923C4442
  s_mul_i32     s62, s66, s71                           // 0000000000E8: 923E4742
  s_mov_b32     s18, s60                                // 0000000000EC: BE92003C
  s_mov_b32     s22, 0x80000000                         // 0000000000F0: BE9600FF 80000000
  s_mov_b32     s14, 0x80000000                         // 0000000000F8: BE8E00FF 80000000
  s_mov_b32     s26, 0x80000000                         // 000000000100: BE9A00FF 80000000
  s_mov_b32     s10, 0x80000000                         // 000000000108: BE8A00FF 80000000
  s_mov_b32     s11, 0x00020000                         // 000000000110: BE8B00FF 00020000
  s_mov_b32     s19, 0x00020000                         // 000000000118: BE9300FF 00020000
  s_mov_b32     s23, 0x00020000                         // 000000000120: BE9700FF 00020000
  s_mov_b32     s15, 0x00020000                         // 000000000128: BE8F00FF 00020000
  s_mov_b32     s27, 0x00020000                         // 000000000130: BE9B00FF 00020000
  s_and_b32     s9, s9, 0x0000ffff                      // 000000000138: 8609FF09 0000FFFF
  s_and_b32     s17, s17, 0x0000ffff                    // 000000000140: 8611FF11 0000FFFF
  s_and_b32     s21, s21, 0x0000ffff                    // 000000000148: 8615FF15 0000FFFF
  s_and_b32     s13, s13, 0x0000ffff                    // 000000000150: 860DFF0D 0000FFFF
  s_and_b32     s25, s25, 0x0000ffff                    // 000000000158: 8619FF19 0000FFFF
  s_or_b32      s9, s9, 0x00040000                      // 000000000160: 8709FF09 00040000
  s_or_b32      s17, s17, 0x00040000                    // 000000000168: 8711FF11 00040000
  s_or_b32      s21, s21, 0x00040000                    // 000000000170: 8715FF15 00040000
  s_or_b32      s13, s13, 0x00040000                    // 000000000178: 870DFF0D 00040000
  s_or_b32      s25, s25, 0x00040000                    // 000000000180: 8719FF19 00040000
  v_accvgpr_write  acc255, 0                            // 000000000188: D3D940FF 18000080
  v_mov_b32     v255, 0                                 // 000000000190: 7FFE0280
  s_cmp_lt_i32  s7, 1                                   // 000000000194: BF048107
  s_cbranch_scc0  label_00BE                            // 000000000198: BF840057
  v_mov_b32     v54, s8                                 // 00000000019C: 7E6C0208
  v_mov_b32     v55, s9                                 // 0000000001A0: 7E6E0209
  v_mov_b32     v56, s16                                // 0000000001A4: 7E700210
  v_mov_b32     v57, s17                                // 0000000001A8: 7E720211
  v_mov_b32     v58, s20                                // 0000000001AC: 7E740214
  v_mov_b32     v59, s21                                // 0000000001B0: 7E760215
  v_mov_b32     v60, 0xdead0001                         // 0000000001B4: 7E7802FF DEAD0001
  v_mov_b32     v61, 0xdead0001                         // 0000000001BC: 7E7A02FF DEAD0001
  v_mov_b32     v62, s12                                // 0000000001C4: 7E7C020C
  v_mov_b32     v63, s13                                // 0000000001C8: 7E7E020D
  v_mov_b32     v64, 0xdead0002                         // 0000000001CC: 7E8002FF DEAD0002
  v_mov_b32     v65, 0xdead0002                         // 0000000001D4: 7E8202FF DEAD0002
  v_mov_b32     v66, 0xdead0003                         // 0000000001DC: 7E8402FF DEAD0003
  v_mov_b32     v67, 0xdead0003                         // 0000000001E4: 7E8602FF DEAD0003
  v_mov_b32     v68, 0xdead0004                         // 0000000001EC: 7E8802FF DEAD0004
  v_mov_b32     v69, 0xdead0004                         // 0000000001F4: 7E8A02FF DEAD0004
  v_mov_b32     v70, 0xdead0005                         // 0000000001FC: 7E8C02FF DEAD0005
  v_mov_b32     v71, 0xdead0005                         // 000000000204: 7E8E02FF DEAD0005
  v_mov_b32     v72, s28                                // 00000000020C: 7E90021C
  v_mov_b32     v73, s29                                // 000000000210: 7E92021D
  v_mov_b32     v74, s24                                // 000000000214: 7E940218
  v_mov_b32     v75, s25                                // 000000000218: 7E960219
  v_mov_b32     v76, s30                                // 00000000021C: 7E98021E
  v_mov_b32     v77, s31                                // 000000000220: 7E9A021F
  v_mov_b32     v78, s64                                // 000000000224: 7E9C0240
  v_mov_b32     v79, s65                                // 000000000228: 7E9E0241
  v_mov_b32     v80, s66                                // 00000000022C: 7EA00242
  v_mov_b32     v81, s67                                // 000000000230: 7EA20243
  v_mov_b32     v82, s68                                // 000000000234: 7EA40244
  v_mov_b32     v83, s69                                // 000000000238: 7EA60245
  v_mov_b32     v84, s70                                // 00000000023C: 7EA80246
  v_mov_b32     v85, s71                                // 000000000240: 7EAA0247
  v_mov_b32     v86, s72                                // 000000000244: 7EAC0248
  v_mov_b32     v87, s73                                // 000000000248: 7EAE0249
  v_mov_b32     v88, 0xdead0006                         // 00000000024C: 7EB002FF DEAD0006
  v_mov_b32     v89, 0xdead0007                         // 000000000254: 7EB202FF DEAD0007
  v_mov_b32     v90, 0xdead0008                         // 00000000025C: 7EB402FF DEAD0008
  v_mov_b32     v91, 0xdeadbeef                         // 000000000264: 7EB602FF DEADBEEF
  v_mov_b32     v92, 0xdeadbeef                         // 00000000026C: 7EB802FF DEADBEEF
  v_mov_b32     v93, 0xdeadbeef                         // 000000000274: 7EBA02FF DEADBEEF
  v_mov_b32     v255, 0                                 // 00000000027C: 7FFE0280
  buffer_store_dwordx4  v[54:57], v255, s[8:11], 0 idxen // 000000000280: E07C2000 800236FF
  v_add_u32     v255, 4, v255                           // 000000000288: 69FFFE84
  buffer_store_dwordx4  v[58:61], v255, s[8:11], 0 idxen // 00000000028C: E07C2000 80023AFF
  v_add_u32     v255, 4, v255                           // 000000000294: 69FFFE84
  buffer_store_dwordx4  v[62:65], v255, s[8:11], 0 idxen // 000000000298: E07C2000 80023EFF
  v_add_u32     v255, 4, v255                           // 0000000002A0: 69FFFE84
  buffer_store_dwordx4  v[66:69], v255, s[8:11], 0 idxen // 0000000002A4: E07C2000 800242FF
  v_add_u32     v255, 4, v255                           // 0000000002AC: 69FFFE84
  buffer_store_dwordx4  v[70:73], v255, s[8:11], 0 idxen // 0000000002B0: E07C2000 800246FF
  v_add_u32     v255, 4, v255                           // 0000000002B8: 69FFFE84
  buffer_store_dwordx4  v[74:77], v255, s[8:11], 0 idxen // 0000000002BC: E07C2000 80024AFF
  v_add_u32     v255, 4, v255                           // 0000000002C4: 69FFFE84
  buffer_store_dwordx4  v[78:81], v255, s[8:11], 0 idxen // 0000000002C8: E07C2000 80024EFF
  v_add_u32     v255, 4, v255                           // 0000000002D0: 69FFFE84
  buffer_store_dwordx4  v[82:85], v255, s[8:11], 0 idxen // 0000000002D4: E07C2000 800252FF
  v_add_u32     v255, 4, v255                           // 0000000002DC: 69FFFE84
  buffer_store_dwordx4  v[86:89], v255, s[8:11], 0 idxen // 0000000002E0: E07C2000 800256FF
  v_add_u32     v255, 4, v255                           // 0000000002E8: 69FFFE84
  buffer_store_dwordx4  v[90:93], v255, s[8:11], 0 idxen // 0000000002EC: E07C2000 80025AFF
  v_add_u32     v255, 4, v255                           // 0000000002F4: 69FFFE84
label_00BE:
  s_waitcnt     0x0000                                  // 0000000002F8: BF8C0000
  s_endpgm                                              // 0000000002FC: BF810000

.rodata
.p2align 6
.amdhsa_kernel kernel_func
    .amdhsa_group_segment_fixed_size 65536
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_sgpr_workgroup_id_z 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 512 
    .amdhsa_next_free_sgpr 96
    .amdhsa_accum_offset 256
    .amdhsa_ieee_mode 0
    .amdhsa_dx10_clamp 0
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [ 1, 0 ]
amdhsa.kernels:
  - .name: kernel_func
    .symbol: kernel_func.kd
    .sgpr_count: 96
    .vgpr_count: 512
    .kernarg_segment_align: 4
    .kernarg_segment_size: 400
    .group_segment_fixed_size: 65536
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .reqd_workgroup_size : [256, 1, 1]
    .max_flat_workgroup_size: 256
    .args:
    - {.name: O, .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global, .actual_access: read_write}
    - {.name: pad, .size: 8, .offset: 8, .value_kind: by_value, .value_type: i32}
    - {.name: X, .size: 8, .offset: 16, .value_kind: global_buffer, .address_space: global, .actual_access: read_only}
    - {.name: pad, .size: 8, .offset: 24, .value_kind: by_value, .value_type: i32}
    - {.name: G, .size: 8, .offset: 32, .value_kind: global_buffer, .address_space: global, .actual_access: read_only}
    - {.name: pad, .size: 8, .offset: 40, .value_kind: by_value, .value_type: i32}
    - {.name: U, .size: 8, .offset: 48, .value_kind: global_buffer, .address_space: global, .actual_access: read_only}
    - {.name: pad, .size: 8, .offset: 56, .value_kind: by_value, .value_type: i32}
    - {.name: D, .size: 8, .offset: 64, .value_kind: global_buffer, .address_space: global, .actual_access: read_only}
    - {.name: pad, .size: 8, .offset: 72, .value_kind: by_value, .value_type: i32}
    - {.name: XQ, .size: 8, .offset: 80, .value_kind: global_buffer, .address_space: global, .actual_access: read_only}
    - {.name: pad, .size: 8, .offset: 88, .value_kind: by_value, .value_type: i32}
    - {.name: GQ, .size: 8, .offset: 96, .value_kind: global_buffer, .address_space: global, .actual_access: read_only}
    - {.name: pad, .size: 8, .offset: 104, .value_kind: by_value, .value_type: i32}
    - {.name: DQ, .size: 8, .offset: 112, .value_kind: global_buffer, .address_space: global, .actual_access: read_only}
    - {.name: pad, .size: 8, .offset: 120, .value_kind: by_value, .value_type: i32}
    - {.name: SMQ, .size: 8, .offset: 128, .value_kind: global_buffer, .address_space: global, .actual_access: read_only}
    - {.name: pad, .size: 8, .offset: 136, .value_kind: by_value, .value_type: i32}
    - {.name: STP, .size: 8, .offset: 144, .value_kind: global_buffer, .address_space: global, .actual_access: read_only}
    - {.name: pad, .size: 8, .offset: 152, .value_kind: by_value, .value_type: i32}
    - {.name: SW, .size: 8, .offset: 160, .value_kind: global_buffer, .address_space: global, .actual_access: read_only}
    - {.name: pad, .size: 8, .offset: 168, .value_kind: by_value, .value_type: i32}
    - {.name: SEP, .size: 8, .offset: 176, .value_kind: global_buffer, .address_space: global, .actual_access: read_only}
    - {.name: pad, .size: 8, .offset: 184, .value_kind: by_value, .value_type: i32}
    - {.name: dim, .size: 4, .offset: 192, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 196, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 200, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 204, .value_kind: by_value, .value_type: i32}
    - {.name: hdim, .size: 4, .offset: 208, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 212, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 216, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 220, .value_kind: by_value, .value_type: i32}
    - {.name: batch, .size: 4, .offset: 224, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 228, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 232, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 236, .value_kind: by_value, .value_type: i32}
    - {.name: eprt, .size: 4, .offset: 240, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 244, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 248, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 252, .value_kind: by_value, .value_type: i32}
    - {.name: Xs, .size: 4, .offset: 256, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 260, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 264, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 268, .value_kind: by_value, .value_type: i32}
    - {.name: GUs, .size: 4, .offset: 272, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 276, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 280, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 284, .value_kind: by_value, .value_type: i32}
    - {.name: Ds, .size: 4, .offset: 288, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 292, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 296, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 300, .value_kind: by_value, .value_type: i32}
    - {.name: Os, .size: 4, .offset: 304, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 308, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 312, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 316, .value_kind: by_value, .value_type: i32}
    - {.name: eGUs, .size: 4, .offset: 320, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 324, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 328, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 332, .value_kind: by_value, .value_type: i32}
    - {.name: eDs, .size: 4, .offset: 336, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 340, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 344, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 348, .value_kind: by_value, .value_type: i32}
    - {.name: eGUQs, .size: 4, .offset: 352, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 356, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 360, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 364, .value_kind: by_value, .value_type: i32}
    - {.name: eDQs, .size: 4, .offset: 368, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 372, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 376, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 380, .value_kind: by_value, .value_type: i32}
    - {.name: eSMQs, .size: 4, .offset: 384, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 388, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 392, .value_kind: by_value, .value_type: i32}
    - {.name: pad, .size: 4, .offset: 396, .value_kind: by_value, .value_type: i32}
...
.end_amdgpu_metadata
