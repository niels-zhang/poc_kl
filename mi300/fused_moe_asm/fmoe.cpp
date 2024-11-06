#include <stdio.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <random>
#include <map>
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#ifdef USE_MKL
#include <mkl.h>
#endif
#define HALF
#ifdef HALF
#include "half.hpp"
#endif
#include "fmoe.hpp"
// #define PER_PIXEL_CHECK
#define ASSERT_ON_FAIL
#define MFMA
// #define ASM_PRINT

#ifndef ABS
#define ABS(x) ((x) > 0 ? (x) : -1 * (x))
#endif

#define HIP_CALL(call)                                                 \
    do                                                                 \
    {                                                                  \
        hipError_t err = call;                                         \
        if (err != hipSuccess)                                         \
        {                                                              \
            printf("[hiperror](%d) fail to call %s", (int)err, #call); \
            exit(0);                                                   \
        }                                                              \
    } while (0)

static inline int get_int(const char *env_name, int def_value)
{
    char *v = getenv(env_name);
    if (v)
        return atoi(v);
    return def_value;
}

#define HSACO "kernel.co"
#define HSA_KERNEL "kernel_func"

std::map<std::string, unsigned int> parse_options(const std::vector<std::string> &optionList)
{
    std::map<std::string, unsigned int> options;
    for (const std::string &option : optionList)
    {
        size_t equalPos = option.find('=');
        if (equalPos != std::string::npos)
        {
            std::string key = option.substr(0, equalPos);
            std::string value = option.substr(equalPos + 1);
            options[key] = std::stoull(value);
        }
        else
        {
            // Handle error or ignore options with no equal sign
            std::cerr << "Error: Invalid option format: " << option << std::endl;
        }
    }
    return options;
}

void get_param(std::map<std::string, unsigned int> parsedOptions, std::string key, unsigned int &value)
{
    auto it = parsedOptions.find(key);
    if (it != parsedOptions.end())
    {
        value = it->second;
    }
}

int main(int argc, char **argv)
{
    
    hipModule_t module;
    hipFunction_t kernel_func;
    hipEvent_t evt_00, evt_11;
    HIP_CALL(hipSetDevice(0));

    HIP_CALL(hipModuleLoad(&module, HSACO));
    HIP_CALL(hipModuleGetFunction(&kernel_func, module, HSA_KERNEL));

    int validate = get_int("VALIDATE", 0);
    
    unsigned int batch = 20;
    unsigned int dim = 512;
    unsigned int hidden_dim = 512;
    unsigned int eprt = 1;
    unsigned int topk = 1;
    unsigned int dump_result = 1;
    unsigned int even_dist = 1;
    unsigned int seed = 0;
    unsigned int sub_X = 32;
    unsigned int sub_GU = 512;
    unsigned int total_loop = 1;
    unsigned int init_pattern = 0;
    unsigned int layout = LAYOUT_16X16;
    
    std::vector<std::string> options;
    for (int i = 1; i < argc; i++)
    {
        options.push_back(argv[i]);
    }
    
    std::map<std::string, unsigned int> parsedOptions = parse_options(options);
    get_param(parsedOptions, "batch",       batch);
    get_param(parsedOptions, "dim",         dim);
    get_param(parsedOptions, "hidden_dim",  hidden_dim);
    get_param(parsedOptions, "eprt",        eprt);
    get_param(parsedOptions, "topk",        topk);
    get_param(parsedOptions, "dump_result", dump_result);
    get_param(parsedOptions, "even_dist",   even_dist);
    get_param(parsedOptions, "seed",        seed);
    get_param(parsedOptions, "sub_X",       sub_X);
    get_param(parsedOptions, "sub_GU",      sub_GU);
    get_param(parsedOptions, "total_loop",  total_loop);
    get_param(parsedOptions, "init_pattern",init_pattern);
    get_param(parsedOptions, "layout",      layout);

    std::cout << "batch:"       << batch        << std::endl;
    std::cout << "dim:"         << dim          << std::endl;
    std::cout << "hidden_dim:"  << hidden_dim   << std::endl;
    std::cout << "eprt:"        << eprt         << std::endl;
    std::cout << "topk:"        << topk         << std::endl;
    std::cout << "dump_result:" << dump_result  << std::endl;
    std::cout << "even_dist:"   << even_dist    << std::endl;
    std::cout << "seed:"        << seed         << std::endl;
    std::cout << "sub_X:"       << sub_X        << std::endl;
    std::cout << "sub_GU:"      << sub_GU       << std::endl;
    std::cout << "total_loop:"  << total_loop   << std::endl;
    std::cout << "init_pattern:"<< init_pattern << std::endl;
    std::cout << "layout:"      << layout       << std::endl;

    //-----host buffer init-------//
    int sz_X, sz_G, sz_U, sz_D, sz_O, sz_W;
    int X_dqn_size;
    int G_dqn_size, D_dqn_size, Smooth_qnt_size;
    
    sz_X        = batch*dim;
    sz_G        = sz_U = sz_D     = eprt*dim*hidden_dim;
    sz_O        = batch*dim;

    sz_W        = batch*topk;
    X_dqn_size  = batch;
    G_dqn_size  = Smooth_qnt_size = eprt*hidden_dim;
    D_dqn_size  = eprt*dim;

    //input buffer init
    T *X_buf                 = (T*)      malloc(sz_X            * sizeof(float)/2);
    T *G_buf                 = (T*)      malloc(sz_G            * sizeof(float)/2);
    T *U_buf                 = (T*)      malloc(sz_U            * sizeof(float)/2);
    T *D_buf                 = (T*)      malloc(sz_D            * sizeof(float)/2);

    uint32   *TKI_buf        = (uint32*) malloc(sz_W            * sizeof(uint32));      //topk_idx
    cl_float *W_buf          = (float*)  malloc(sz_W            * sizeof(float));       //weight
    cl_float *X_dqn_buf      = (float*)  malloc(X_dqn_size      * sizeof(float));       //dequant values for X
    cl_float *G_dqn_buf      = (float*)  malloc(G_dqn_size      * sizeof(float));       //dequant values for Gate
    cl_float *D_dqn_buf      = (float*)  malloc(D_dqn_size      * sizeof(float));       //dequant values for Down
    cl_float *Smooth_qnt_buf = (float*)  malloc(Smooth_qnt_size * sizeof(float));       //smooth quant values for GEMM0

    uint32 O_elemSize        = 2;
    DATA_TYPE type           = BF16;
    DATA_TYPE O_elemType     = BF16;
    //cpu_O_buf and gpu_O_buf use cl_float ensures large enough room
    cl_float *gpu_O_buf      = (float*)  malloc(sz_O * sizeof(float));
    cl_float *cpu_O_buf      = (float*)  malloc(sz_O * sizeof(float));
    memset(gpu_O_buf, 0, sz_O*sizeof(cl_float));
    memset(cpu_O_buf, 0, sz_O*sizeof(cl_float));

    srand(++seed);
    moe_init(X_buf,         1,      batch,      dim,        type, init_pattern);
    srand(++seed);
    moe_init(G_buf,         eprt,   hidden_dim, dim,        type, init_pattern);
    srand(++seed);
    moe_init(U_buf,         eprt,   hidden_dim, dim,        type, init_pattern);
    srand(++seed);
    moe_init(D_buf,         eprt,   dim,        hidden_dim, type, init_pattern);
    srand(++seed);
    moe_init(W_buf,         1,      batch,      topk,       FP32, init_pattern);
    if (topk != 1) { //if only one expert, no need softmax
        moe_weight_softmax(W_buf, batch,topk);
    }

    srand(++seed);
    //if even_dist==0, W_buf may be rewritten inside, depend on internal value useSimpleRandom
    moe_topk_init(TKI_buf, W_buf, eprt, batch, topk, even_dist);
    srand(++seed);
    moe_init(X_dqn_buf,     1,      batch,      1,          FP32, init_pattern);
    srand(++seed);
    moe_init(G_dqn_buf,     eprt,   1,          hidden_dim, FP32, init_pattern);
    srand(++seed);
    moe_init(D_dqn_buf,     eprt,   1,          dim,        FP32, init_pattern);
    srand(++seed);
    moe_init(Smooth_qnt_buf,eprt,   1,          hidden_dim, FP32, init_pattern);
    // clang-format off
    // [indexing implementation-1]
    // using M_a as constexpr block_size to partition all tokens into different slices
    // each slice map to one expert, and one expert can have multiple slices
    // e.g. num_experts = 6, top_k=3, M_a = 4, input_tokens = 5
    // before sort, topk_ids is : [[0, 3, 5], [2, 3, 5], [1, 3, 5], [1, 2, 3], [1, 3, 5]]
    //                            tok-0      tok-1      tok-2      tok-3      tok-4
    //           topk_weight is : [[a, b, c], [d, e, f], [g, h, i], [j, k, l], [m, n, o]] (some float number)
    //
    // token_id_per_expert is : [[0], [2, 3, 4], [1, 3], [0, 1, 2, 3, 4], [], [0, 1, 5, 5]]
    //  (only for reference)    exp-0  exp-1     exp-2   exp-3          exp-4  exp-5
    // weight_id_per_expert is: [[a], [g, j, m], [d, k], [b, e, h, l, n], [], [c, f, i, o]]
    //
    // max_tokens_post_padded : top_k * input_tokens + num_experts * (M_a - 1)
    // * this could be larger than actual, since actual tokens are on GPU
    //
    // sorted_token_ids_ptr   : [0, 6, 6, 6, 2, 3, 4, 6, 1, 3, 6, 6, 0, 1, 2, 3, 4, 6, 6, 6, 6, 6, 6, 6, 0, 1, 2, 5]
    //                          |-  exp-0  -|-  exp-1  -|-  exp-2  -|-      exp-3          -|-  exp-4 -|-  exp-5  -|
    // sorted_weight_ptr      : [a, *, *, *, g, j, m, *, d, k, *, *, b, e, h, l, n, *, *, *, *, *, *, *, c, f, i, o]
    //
    // * length is max_tokens_post_padded, actual size is num_tokens_post_padded_ptr
    //
    // sorted_expert_ids_ptr  : [0, 1, 2, 3, 3, 4, 5]
    // * length is (max_tokens_post_padded + block_size - 1) / block_size
    // sub_X_cnt = size of available sorted_expert_ids_ptr = 7

    uint32 sz_stp, sz_sw, sz_sep, sub_X_cnt = 0;//initialize sub_X_cnt to 0, important
    sz_stp = sz_sw = topk*batch + eprt*sub_X-1; //max_length
    sz_sep = (sz_stp + sub_X - 1)/sub_X;        //max_length

    uint32*   sorted_token_ids_ptr  = (uint32*)   malloc(sz_stp * sizeof(uint32));
    cl_float* sorted_weight_buf     = (cl_float*) malloc(sz_sw  * sizeof(cl_float));
    uint32*   sorted_expert_ids_ptr = (uint32*)   malloc(sz_sep * sizeof(uint32));
    moe_twe_ptr_gen(sorted_token_ids_ptr, sorted_weight_buf, sorted_expert_ids_ptr, sub_X_cnt, W_buf, TKI_buf, batch, eprt, topk, sub_X);
    if(dump_result)   
    {
        moe_dump_inHex(X_buf,           "X.hex" ,       1,     batch,       dim,        type,   0); //batch*dim      row major
        moe_dump_inHex(G_buf,           "G.hex" ,       eprt,  hidden_dim,  dim,        type,   1); //dim*hidden_dim col major
        moe_dump_inHex(U_buf,           "U.hex" ,       eprt,  hidden_dim,  dim,        type,   1); //dim*hidden_dim col major
        moe_dump_inHex(D_buf,           "D.hex" ,       eprt,  dim,         hidden_dim, type,   1); //hidden_dim*dim col major
        moe_dump_inHex(W_buf,           "W.hex" ,       1,     batch,       topk,       FP32,   0); //batch*topk     row major
        moe_dump_inHex(X_dqn_buf,       "X_DQN.hex" ,   1,     batch,       1,          FP32,   0); //batch*1        row major
        moe_dump_inHex(G_dqn_buf,       "G_DQN.hex" ,   eprt,  1,           hidden_dim, FP32,   0); //1*hidden_dim   row major
        moe_dump_inHex(D_dqn_buf,       "D_DQN.hex" ,   eprt,  1,           dim,        FP32,   0); //1*dim          row major
        moe_dump_inHex(Smooth_qnt_buf,  "SMT_QNT.hex" , eprt,  1,           hidden_dim, FP32,   0); //1*hidden_dim   row major

        moe_dump_topk_inHex  (TKI_buf,  "Topk.hex",     batch, topk);
        moe_dump_topk_inValue(TKI_buf,  "Topk.txt",     batch, topk);

        moe_dump_weight_inHex  (W_buf,  "Weight.hex",   batch, topk);
        moe_dump_weight_inValue(W_buf,  "Weight.txt",   batch, topk);

        uint32 *eprt_slices = (uint32*) malloc(eprt * sizeof(uint32));
        memset(eprt_slices, 0, eprt*sizeof(uint32)); //init to 0, important
        for (uint32 i = 0; i < sub_X_cnt; i++) {
            eprt_slices[sorted_expert_ids_ptr[i]]++;
        }
        moe_dump_topk_inHex  (sorted_expert_ids_ptr, "Exp_IDs.hex", 1, sub_X_cnt);
        moe_dump_topk_inValue(sorted_expert_ids_ptr, "Exp_IDs.txt", 1, sub_X_cnt);

        moe_twe_ptr_dump_inHex  ("SortedTokenWeights.hex", sorted_token_ids_ptr, sorted_weight_buf, eprt_slices, eprt, sub_X);
        moe_twe_ptr_dump_inValue("SortedTokenWeights.txt", sorted_token_ids_ptr, sorted_weight_buf, eprt_slices, eprt, sub_X);
        free(eprt_slices);
    }
    //-----host buffer data init end-------//

    printf("Calculating CPU result ~~~~~\n");
    //generate reference data. standard golden generation should be done before shuffle
    /*moe_ref_std(host_X, host_G, U_buf, host_D, host_W, host_TKI, 
                X_dqn_buf, G_dqn_buf, D_dqn_buf, Smooth_qnt_buf,
                cpu_O_buf, atm_f32,
                batch, hidden_dim, dim, eprt, topk, sub_X, sub_GU,
                type, dbg_trace, layout);*/

    moe_shuffle(G_buf, eprt, hidden_dim, dim,  true, type, (DATA_LAYOUT)layout);
    moe_shuffle(D_buf, eprt, dim,  hidden_dim, true, type, (DATA_LAYOUT)layout);

    float16 *dev_X, *dev_G, *dev_D, *dev_O;
    unsigned int*  dev_STP;
    float* dev_SW;
    unsigned int*  dev_SEP;

    HIP_CALL(hipSetDevice(0));
    HIP_CALL(hipMalloc(&dev_X, sz_X * sizeof(float) / 2));
    HIP_CALL(hipMalloc(&dev_G, sz_G * sizeof(float) / 2));
    HIP_CALL(hipMalloc(&dev_D, sz_D * sizeof(float) / 2));
    HIP_CALL(hipMalloc(&dev_O, sz_O * sizeof(float) / 2));
    HIP_CALL(hipMalloc(&dev_STP, sz_stp * sizeof(unsigned int)));
    HIP_CALL(hipMalloc(&dev_SW,   sz_sw * sizeof(float)));
    HIP_CALL(hipMalloc(&dev_SEP, sz_sep * sizeof(unsigned int)));

    printf("dev_X:   start-%p, 0x%lxKB, end-%p\n", dev_X,  sz_X * sizeof(float) / 2/1024, dev_X+sz_X);
    printf("dev_G:   start-%p, 0x%lxKB, end-%p\n", dev_G,  sz_G * sizeof(float) / 2/1024, dev_G+sz_G);
    printf("dev_D:   start-%p, 0x%lxKB, end-%p\n", dev_D,  sz_D * sizeof(float) / 2/1024, dev_D+sz_D);
    printf("dev_O:   start-%p, 0x%lxKB, end-%p\n", dev_O,  sz_O * sizeof(float) / 2/1024, dev_O+sz_O);
    printf("dev_STP: start-%p, 0x%lxKB, end-%p\n", dev_STP,sz_stp * sizeof(unsigned int)/1024, dev_STP +sz_stp);
    printf("dev_SW : start-%p, 0x%lxKB, end-%p\n", dev_SW, sz_sw * sizeof(float)/1024, dev_SW+sz_sw);
    printf("dev_SEP: start-%p, 0x%lxKB, end-%p\n", dev_SEP,sz_sep * sizeof(unsigned int)/1024, dev_SEP+sz_sep);

    HIP_CALL(hipMemcpy(dev_X, X_buf,     sz_X * sizeof(float) / 2, hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(dev_G, G_buf,     sz_G * sizeof(float) / 2, hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(dev_D, D_buf,     sz_D * sizeof(float) / 2, hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(dev_O, gpu_O_buf, sz_O * sizeof(float) / 2, hipMemcpyHostToDevice));
    
    HIP_CALL(hipMemcpy(dev_STP, sorted_token_ids_ptr,  sz_stp * sizeof(unsigned int), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(dev_SW,  sorted_weight_buf,     sz_sw  * sizeof(float),        hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(dev_SEP, sorted_expert_ids_ptr, sz_sep * sizeof(unsigned int), hipMemcpyHostToDevice));

#ifdef ASM_PRINT
    // debug pointer
    float *host_print, *print;
    host_print = (float *)malloc(bdx * 8);
    HIP_CALL(hipMalloc(&print, bdx * 8));
#endif
    struct p3
    {
        unsigned int _p0;
        unsigned int _p1;
        unsigned int _p2;
    };
    struct p2
    {
        unsigned int _p0;

        unsigned int _p1;
    };
    struct __attribute__((packed))
    {
        void *ptr_O;
        p2 _p0;
        void *ptr_X;
        p2 _p1;
        void *ptr_G;
        p2 _p2;
        void *ptr_U;
        p2 _p3;
        void *ptr_D;
        p2 _p4;
        void *ptr_XQ;
        p2 _p5;
        void *ptr_GQ;
        p2 _p6;
        void *ptr_DQ;
        p2 _p7;
        void *ptr_SMQ;
        p2 _p8;
        void *ptr_STP;
        p2 _p9;
        void *ptr_SW;
        p2 _p10;
        void *ptr_SEP;
        p2 _p11;
        unsigned int dim;
        p3 _p12;
        unsigned int hidden_dim;
        p3 _p13;
        unsigned int token_cnt;
        p3 _p14;
        unsigned int eprt_cnt;
        p3 _p15;
        unsigned int Xs;
        p3 _p16;
        unsigned int GUs;
        p3 _p17;
        unsigned int Ds;
        p3 _p18;
        unsigned int Os;
        p3 _p19;
        unsigned int eGUs;
        p3 _p20;
        unsigned int eDs;
        p3 _p21;
        unsigned int eGUQs;
        p3 _p22;
        unsigned int eDQs;
        p3 _p23;
        unsigned int eSMQs;
        p3 _p24;
#ifdef ASM_PRINT
        void *print;
#endif
    } args;
    size_t  arg_size         = sizeof(args);
    int stride_X             = dim        * sizeof(float)/2;
    int stride_GU            = dim        * sizeof(float)/2;
    int stride_D             = hidden_dim * sizeof(float)/2;
    int stride_expert_GU     = stride_GU  * hidden_dim;
    int stride_expert_D      = stride_D   * dim;
    int stride_expert_GUDQN  = hidden_dim * sizeof(float);
    int stride_expert_DDQN   = dim        * sizeof(float);
    int stride_expert_SMTDQN = hidden_dim * sizeof(float);
    int stride_O             = dim        * sizeof(float)/2;

    args.ptr_O      = (void *)dev_O;
    args.ptr_X      = (void *)dev_X;
    args.ptr_G      = (void *)dev_G;
    args.ptr_U      = (void *)NULL;
    args.ptr_D      = (void *)dev_D;
    args.ptr_XQ     = (void *)NULL;
    args.ptr_GQ     = (void *)NULL;
    args.ptr_DQ     = (void *)NULL;
    args.ptr_SMQ    = (void *)NULL;
    args.ptr_STP    = (void *)dev_STP;
    args.ptr_SW     = (void *)dev_SW;
    args.ptr_SEP    = (void *)dev_SEP;
    args.dim        = dim;
    args.hidden_dim = hidden_dim;
    args.token_cnt  = batch;
    args.eprt_cnt   = eprt;
    args.Xs         = stride_X;
    args.GUs        = stride_GU;
    args.Ds         = stride_D;
    args.Os         = stride_O;
    args.eGUs       = stride_expert_GU;
    args.eDs        = stride_expert_D;
    args.eGUQs      = stride_expert_GUDQN;
    args.eDQs       = stride_expert_DDQN;
    args.eSMQs      = stride_expert_SMTDQN;   
#ifdef ASM_PRINT
    args.print = (void *)print;
#endif

    printf("argsize: %zu\n", arg_size);

    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &arg_size, HIP_LAUNCH_PARAM_END};

    int warm_ups = 0;
    int i;
    int bdx = 256;
    int gdx = ((hidden_dim+sub_GU-1)/sub_GU);
    int gdy = sub_X_cnt;
    int gdz = 1;

    std::cout << "sub_X_cnt: " << sub_X_cnt << std::endl;
    for (i = 0; i < warm_ups; i++)
    {
        HIP_CALL(hipModuleLaunchKernel(kernel_func, gdx, gdy, gdz, bdx, 1, 1, 0, 0, NULL, (void **)&config));
        if (i == warm_ups-1)
            std::cout << "warm up done" << std::endl;
    }

#ifdef ASM_PRINT
    int max_i = 256;
    HIP_CALL(hipMemcpy(host_print, print, 8 * max_i, hipMemcpyDeviceToHost));
    for (int i = 0; i < max_i; i++)
    {
        if (((uint32_t *)host_print)[2 * i + 1] != 0x5c005c00)
            printf("Thread%d, PrintVal:0x%x\n", ((int *)host_print)[2 * i], ((uint32_t *)host_print)[2 * i + 1]);
        // std::cout<<"Thread"<<((int*) host_print)[2*i]<<", PrintVal1:"<<(((float16*)host_print)[4*i+2])<<
        //", PrintVal2:"<<( ( (float16*)host_print )[4*i+3] )<<std::endl;
    }
#endif

    HIP_CALL(hipEventCreate(&evt_00));
    HIP_CALL(hipEventCreate(&evt_11));
    HIP_CALL(hipDeviceSynchronize());
    HIP_CALL(hipEventRecord(evt_00, NULL));
    for (i = 0; i < total_loop; i++)
        HIP_CALL(hipModuleLaunchKernel(kernel_func, gdx, gdy, gdz, bdx, 1, 1, 0, 0, NULL, (void **)&config));

    std::cout << "we are done, looped:" << total_loop << " times" << std::endl;
    float elapsed_ms;
    HIP_CALL(hipEventRecord(evt_11, NULL));
    HIP_CALL(hipEventSynchronize(evt_11));
    HIP_CALL(hipDeviceSynchronize());
    HIP_CALL(hipEventElapsedTime(&elapsed_ms, evt_00, evt_11));
    HIP_CALL(hipEventDestroy(evt_00));
    HIP_CALL(hipEventDestroy(evt_11));

    float time_per_loop = elapsed_ms / total_loop;
    float gflops = (float)2 * batch * topk * dim * hidden_dim *2 / time_per_loop / (1e6);

    printf("b:%d,d:%d,hd:%d,e:%d,tpk:%d,evn:%d, time: %.3f, gflops:%.3f\n", batch, dim, hidden_dim, eprt, topk,even_dist, time_per_loop, gflops);
    // if(validate){
    //     hgemm_cr_kpack2(host_c, host_a, host_b, alpha, m,n,k,lda/sizeof(float),ldb/sizeof(float),ldc/sizeof(float));
    //     HIP_CALL(hipMemcpy(fp16_c, dev_c, ldc*(n>>1), hipMemcpyDeviceToHost));
    //     bool res = valid_vector( host_c, fp16_c, m*n );
    //     printf(",%s",res?"valid":"fail");
    // }
    // printf("\n");
    
    HIP_CALL(hipMemcpy(gpu_O_buf, dev_O, sz_O * sizeof(float) / 2, hipMemcpyDeviceToHost));
     
    if (dump_result)
    { 
        moe_dump_inHex(gpu_O_buf,  "gpu_O.hex", 1, batch, dim, type);
    }

    free(X_buf);
    free(G_buf);
    free(U_buf);
    free(D_buf);
    free(gpu_O_buf);
    free(cpu_O_buf);
    free(W_buf);
    free(TKI_buf);
    free(X_dqn_buf);
    free(G_dqn_buf);
    free(D_dqn_buf);
    free(Smooth_qnt_buf);
    free(sorted_token_ids_ptr);
    free(sorted_weight_buf);
    free(sorted_expert_ids_ptr);

    HIP_CALL(hipFree(dev_X));
    HIP_CALL(hipFree(dev_G));
    HIP_CALL(hipFree(dev_D));
    HIP_CALL(hipFree(dev_O));
    HIP_CALL(hipFree(dev_STP));
    HIP_CALL(hipFree(dev_SW));
    HIP_CALL(hipFree(dev_SEP));

#ifdef ASM_PRINT
    free(host_print);
    hipFree(print);
#endif
    // printf("CU:%d, TIPS:%.3f(2x:%.3f, 4x:%.3f), cost:%fms per loop\n", num_cu, tips, 2*tips, 4*tips, time_per_loop);
}
