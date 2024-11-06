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

using float16 = half_float::half;
static inline bool valid_vector(const float *ref, const float16 *pred, int n, double nrms = 1e-3)
{
    double s0 = 0.0;
    double s1 = 0.0;
#ifdef PER_PIXEL_CHECK
    int pp_err = 0;
#endif
    int i_start = 0, i_end = n;
    int i_num = i_end - i_start;
    for (int i = i_start; i < i_end; ++i)
    {
        double ri = (double)ref[i];
        double pi = (double)pred[i];
        double d = ri - pi;
        double dd = d * d;
        double rr = 2.0 * ri * ri;
        s0 += dd;
        s1 += rr;

#ifdef PER_PIXEL_CHECK
        double delta = ABS(ri - pi) / ri;
        if (delta > 1e-3)
        {
#ifdef ASSERT_ON_FAIL
            if (pp_err < 100)
                printf("diff at %4d, ref:%lf, pred:%lf(0x%04x), d:%lf\n", i, ri, pi, ((uint16_t *)pred)[i], delta);
#endif
            pp_err++;
        }
#endif
    }
    //    printf("pp_crr:%d, pp_err:%d, crr_ratio:%.3f, nrms:%lf, s0:%lf, s1:%lf\n",i_num-pp_err, pp_err, (float)(i_num-pp_err)/(float)i_num, sqrt(s0/s1),s0,s1);

    return (sqrt(s0 / s1) < nrms)
#ifdef PER_PIXEL_CHECK
           && (pp_err == 0)
#endif
        ;
}

void hgemm_cr_kpack2(
    float *ptr_c,
    const float *__restrict__ ptr_a,
    const float *__restrict__ ptr_b,
    float alpha,
    unsigned int m,
    unsigned int n,
    unsigned int k,
    unsigned int lda,
    unsigned int ldb,
    unsigned int ldc)
{
#ifdef USE_MKL
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                m, n, k, alpha, ptr_a, lda, ptr_b, ldb, 0, ptr_c, ldc);
#else
    // change the layout
    unsigned int im, in, ik;
    for (in = 0; in < n; in++)
    {
        for (im = 0; im < m; im++)
        {
#ifndef MFMA
            float c = .0;
            for (ik = 0; ik < (k >> 1); ik++)
            {
                c += ptr_a[ik * lda * 2 + im * 2] * ptr_b[ik * ldb * 2 + in * 2];
                c += ptr_a[ik * lda * 2 + im * 2 + 1] * ptr_b[ik * ldb * 2 + in * 2 + 1];
            }
            ptr_c[in * ldc + im] = alpha * c;
#endif

#ifdef MFMA
            float c = .0;
            for (ik = 0; ik < (k >> 2); ik++)
            {
                c += ptr_a[ik * 4 * lda + im * 4 + 0] * ptr_b[ik * 4 * ldb + in * 4 + 0] + ptr_a[ik * 4 * lda + im * 4 + 1] * ptr_b[ik * 4 * ldb + in * 4 + 1] + ptr_a[ik * 4 * lda + im * 4 + 2] * ptr_b[ik * 4 * ldb + in * 4 + 2] + ptr_a[ik * 4 * lda + im * 4 + 3] * ptr_b[ik * 4 * ldb + in * 4 + 3];
            }
            ptr_c[in * ldc + im] = alpha * c;
#endif
        }
    }
#endif
}

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

void rand_vector_2d(float *v, int row, int col, int ld)
{
    int r, c;
    static int flag = 0;
    if (!flag)
    {
        srand(time(NULL));
        flag = 1;
    }

    for (r = 0; r < row; r++)
    {
        for (c = 0; c < col; c++)
        {
            v[r * ld + c] = ((float)(rand() % 100)) / 100.0f;
            // v[r*ld+c] = ((float)(r % 100)+1) / 100.0f + ((float)(c % 100)+1) / 1000.0f;
            // v[r*ld+c] = 1.0;
        }
    }
}

// #define HSACO "hgemm128x128.hsaco"
#define HSACO "kernel.co"
// #define HSACO "hgemm_128x128_kpack2"
#define HSA_KERNEL "kernel_func"

// #define Batch    1
// #define Head_num 1
// #define seq_len  128
// #define head_dim  128
// #define seq_len  128

std::map<std::string, int> parse_options(const std::vector<std::string> &optionList)
{
    std::map<std::string, int> options;
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

void get_param(std::map<std::string, int> parsedOptions, std::string key, int &value)
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
    // int m = get_int("M", HGEMM_M);
    // int n = get_int("N", HGEMM_N);
    // int k = get_int("K", HGEMM_K);

    // int lda = m*sizeof(float);
    // int ldb = n*sizeof(float);
    // int ldc = m*sizeof(float);
    int b = 20;     // get_int("M", HGEMM_M);
    int d = 512;    // get_int("N", HGEMM_N);
    int hdim = 512; // get_int("K", HGEMM_K);
    int eprt = 1;
    int topk = 1;
    int even_dist = 1;
    int seed = 0;

    int sub_X = 32;
    int sub_GU = 512;
    int dump_result = 0;
    int type = BF16;
    std::vector<std::string> options;
    for (int i = 1; i < argc; i++)
    {
        options.push_back(argv[i]);
    }
    std::map<std::string, int> parsedOptions = parse_options(options);
    get_param(parsedOptions, "b", b);
    get_param(parsedOptions, "d", d);
    get_param(parsedOptions, "hdim", hdim);
    get_param(parsedOptions, "eprt", eprt);
    get_param(parsedOptions, "dump_result", dump_result);
    get_param(parsedOptions, "topk", topk);
    get_param(parsedOptions, "even_dist", even_dist);
    get_param(parsedOptions, "seed", seed);
    get_param(parsedOptions, "sub_X", sub_X);
    get_param(parsedOptions, "sub_GU", sub_GU);

    std::cout << "b:" << b << std::endl;
    std::cout << "d:" << d << std::endl;
    std::cout << "hdim:" << hdim << std::endl;
    std::cout << "eprt:" << eprt << std::endl;
    std::cout << "dump_result:" << dump_result << std::endl;
    std::cout << "topk:" << topk << std::endl;
    std::cout << "even_dist:" << even_dist << std::endl;
    std::cout << "seed:" << seed << std::endl;
    std::cout << "sub_X:" << sub_X << std::endl;
    std::cout << "sub_GU:" << sub_GU << std::endl;

    float16 *host_X, *host_G, *host_D, *host_O;
    float *W_buf;
    unsigned int *TKI_buf;

    int sz_X, sz_G, sz_U, sz_D, sz_O, sz_W;
    sz_X        = b*d;
    sz_G        = sz_D     = eprt*d*hdim;
    sz_O        = b*d;
    sz_W        = b*topk;

    host_X = (float16 *)malloc(sz_X * sizeof(float) / 2);
    host_G = (float16 *)malloc(sz_G * sizeof(float) / 2);
    host_D = (float16 *)malloc(sz_D * sizeof(float) / 2);
    host_O = (float16 *)malloc(sz_O * sizeof(float) / 2);

    W_buf = (float *)malloc(sz_W * sizeof(float));
    TKI_buf = (unsigned int *)malloc(sz_W * sizeof(int));

    int init_pattern = 0;
    get_param(parsedOptions, "init_pattern", init_pattern);

            srand(++seed);
            moe_init(host_X,         1,      b,      d,        type, init_pattern);
            srand(++seed);
            moe_init(host_G,         eprt,   hdim,     d,        type, init_pattern);
            srand(++seed);
            moe_init(host_D,         eprt,   d,      hdim,       type, init_pattern);
            srand(++seed);
            moe_init(W_buf,         1,      b,      topk,     FP32, init_pattern);
            if (topk != 1) { //if only one expert, no need softmax
                moe_weight_softmax(W_buf, b,topk);
            }
            srand(++seed);
            //if even_dist==0, W_buf may be rewritten inside, depend on internal value useSimpleRandom
            moe_topk_init(TKI_buf, W_buf, eprt, b, topk, even_dist, host_X, d, type, init_pattern);

    memset(host_O, 0, sz_O * sizeof(float) / 2);

        unsigned int sz_stp, sz_sw, sz_sep, sub_X_cnt = 0;//initialize sub_X_cnt to 0, important
        sz_stp = sz_sw = topk*b + eprt*sub_X-1; //max_length
        sz_sep = (sz_stp + sub_X - 1)/sub_X;        //max_length

        unsigned int*   sorted_token_ids_ptr  = malloc(sz_stp * sizeof(unsigned int));
        float* sorted_weight_buf     = malloc(sz_sw * sizeof(float));
        unsigned int*   sorted_expert_ids_ptr = malloc(sz_sep * sizeof(unsigned int));

        moe_twe_ptr_gen(sorted_token_ids_ptr, sorted_weight_buf, sorted_expert_ids_ptr, sub_X_cnt, W_buf, TKI_buf, b, eprt, topk, sub_X);

        if(dump_result)
        {
            moe_dump_inHex(host_X,           "X.hex" ,       1,     b,       d,        type,   0); //batch*dim      row major
            moe_dump_inHex(host_G,           "G.hex" ,       eprt,  hdim,      d,        type,   1); //dim*hidden_dim col major
            moe_dump_inHex(host_D,           "D.hex" ,       eprt,  d,       hdim,       type,   1); //hidden_dim*dim col major
            moe_dump_inHex(host_W,           "W.hex" ,       1,     b,       topk,     FP32,   0); //batch*topk     row major

            moe_dump_topk_inHex  (host_TKI,  "Topk.hex",     b, topk);
            moe_dump_topk_inValue(host_TKI,  "Topk.txt",     b, topk);

            moe_dump_weight_inHex  (host_W,  "Weight.hex",   b, topk);
            moe_dump_weight_inValue(host_W,  "Weight.txt",   b, topk);

            unsigned int *eprt_slices = malloc(eprt * sizeof(unsigned int));
            memset(eprt_slices, 0, eprt*sizeof(unsigned int)); //init to 0, important
            for (unsigned int i = 0; i < sub_X_cnt; i++) {
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
        moe_ref_std(host_X, host_G, U_buf, host_D, host_W, host_TKI, 
                    X_dqn_buf, G_dqn_buf, D_dqn_buf, Smooth_qnt_buf,
                    cpu_O_buf, atm_f32,
                    batch, hidden_dim, dim, eprt, topk, sub_X, sub_GU,
                    type, dbg_trace, layout);

        moe_shuffle(host_G, eprt, hdim, d,  true, type, layout);
        moe_shuffle(host_D, eprt, d,  hdim, true, type, layout);

   
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
    HIP_CALL(hipMalloc(&dev_SW, sz_sw * sizeof(float)));
    HIP_CALL(hipMalloc(&dev_SEP, sz_sep * sizeof(unsigned int)));

    printf("dev_X:   start-%p, 0x%lxKB, end-%p\n", dev_X,  sz_X * sizeof(float) / 2/1024, dev_X+sz_X);
    printf("dev_G:   start-%p, 0x%lxKB, end-%p\n", dev_G,  sz_G * sizeof(float) / 2/1024, dev_G+sz_G);
    printf("dev_D:   start-%p, 0x%lxKB, end-%p\n", dev_D,  sz_D * sizeof(float) / 2/1024, dev_D+sz_D);
    printf("dev_O:   start-%p, 0x%lxKB, end-%p\n", dev_O,  sz_O * sizeof(float) / 2/1024, dev_O+sz_O);
    printf("dev_STP: start-%p, 0x%lxKB, end-%p\n", dev_STP,sz_stp * sizeof(unsigned int)/1024, dev_STP +sz_stp);
    printf("dev_SW : start-%p, 0x%lxKB, end-%p\n", dev_SW, sz_sw * sizeof(float)/1024, dev_SW+sz_sw);
    printf("dev_SEP: start-%p, 0x%lxKB, end-%p\n", dev_SEP,sz_sep * sizeof(unsigned int)/1024, dev_SEP+sz_sep);

    HIP_CALL(hipMemcpy(dev_X, host_X, sz_X * sizeof(float) / 2, hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(dev_G, host_G, sz_G * sizeof(float) / 2, hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(dev_D, host_D, sz_D * sizeof(float) / 2, hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(dev_O, host_O, sz_O * sizeof(float) / 2, hipMemcpyHostToDevice));
    
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
    size_t arg_size = sizeof(args);
    args.ptr_dq = (void *)dev_dq;
    args.ptr_dk = (void *)dev_dk;
    args.ptr_dv = (void *)dev_dv;
    args.ptr_q = (void *)dev_q;
    args.ptr_k = (void *)dev_k;
    args.ptr_v = (void *)dev_v;
    args.ptr_do = (void *)dev_do;
    args.ptr_lse = (void *)dev_lse;
    args.ptr_odo = (void *)dev_odo;
    args.scalar = k_scalar;
    args.log2e = k_log2e;
    args.seq_len = s;
    args.Ts = stride_tg;
    args.Hs = stride_head;
    args.BAs = stride_batch;
    args.Seqs = stride_seqlen;
    args.ratio = qa_rt;
    args.Hs_kv = stride_head_kv;
    args.BAs_kv = stride_batch_kv;
    args.Seqs_kv = stride_seqlen_kv;
    args.Seqs_dkv = stride_seqlen_dkv;
#ifdef ASM_PRINT
    args.print = (void *)print;
#endif

    printf("argsize: %zu\n", arg_size);

    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &arg_size, HIP_LAUNCH_PARAM_END};

    int total_loop = 8;
    int warm_ups = 0;
    int i;

    int bdx = 256;
    int gdx = (s+ts_kv-1) / ts_kv;
    int gdy = h;
    int gdz = b;

    if (mask && mask_kb)
    {
        int num_tg = (s+ts_kv-1) / ts_kv;
        gdx = (num_tg%2) ? (num_tg/2+1) : (num_tg/2);
    }

    for (i = 0; i < warm_ups; i++)
    {
        HIP_CALL(hipModuleLaunchKernel(kernel_func, gdx, gdy, gdz, bdx, 1, 1, 0, 0, NULL, (void **)&config));
        std::cout << "safe here" << std::endl;
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

    std::cout << "we are done" << std::endl;
    float elapsed_ms;
    HIP_CALL(hipEventRecord(evt_11, NULL));
    HIP_CALL(hipEventSynchronize(evt_11));
    HIP_CALL(hipDeviceSynchronize());
    HIP_CALL(hipEventElapsedTime(&elapsed_ms, evt_00, evt_11));
    HIP_CALL(hipEventDestroy(evt_00));
    HIP_CALL(hipEventDestroy(evt_11));

    float time_per_loop = elapsed_ms / total_loop;
    float gflops = (float)2 * b * topk * d * hdim *2 / time_per_loop / (1e6);

    printf("b:%d,d:%d,hd:%d,e:%d,tpk:%d,evn:%d, time: %.3f, gflops:%.3f\n", b, d, hdim, d, eprt,topk,even_dist, time_per_loop, gflops);
    // if(validate){
    //     hgemm_cr_kpack2(host_c, host_a, host_b, alpha, m,n,k,lda/sizeof(float),ldb/sizeof(float),ldc/sizeof(float));
    //     HIP_CALL(hipMemcpy(fp16_c, dev_c, ldc*(n>>1), hipMemcpyDeviceToHost));
    //     bool res = valid_vector( host_c, fp16_c, m*n );
    //     printf(",%s",res?"valid":"fail");
    // }
    // printf("\n");
    
    HIP_CALL(hipMemcpy(host_O, dev_O, sz_O * sizeof(float) / 2, hipMemcpyDeviceToHost));
     
    if (dump_result)
    { 
        moe_dump_inHex(host_O,  "gpu_O.hex", 1, b, d, type);
    }

    free(host_X);
    free(host_G);
    free(host_D);
    free(host_O);
    free(W_buf);
    free(TKI_buf);
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
}
