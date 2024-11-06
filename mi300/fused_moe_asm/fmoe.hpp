#ifndef FUSED_MOE_HPP
#define FUSED_MOE_HPP

#define BEXP16(x) ((x & 0x7c00) >> 10)
#define MAN16(x)  (x & 0x3ff)
#define SIGN16(x) ((x >> 15) & 1)
#define BEXP32(x) ((x & 0x7f800000) >> 23)
#define MAN32(x)  (x & 0x7fffff)
#define SIGN32(x) (x >> 31)
#define BEXP64(x) ((x & 0x7ff0000000000000ULL) >> 52)
#define MAN64(x)  (x & 0xfffffffffffffULL)
#define SIGN64(x) (x >> 63)
#define FP16(s,e,m) ((s<<15) | (e<<10) | (m&0x3ff))
#define FP32(s,e,m) ((s<<31) | (e<<23) | (m&0x7fffff))
#define FP64(s,e,m) ((s<<63) | (e<<52) | (m&0xfffffffffffffULL))

using cl_float          = float;
using uint32            = unsigned int;
using int32             = int;
using uint16            = unsigned short;
using float16           = unsigned short;
using T                 = float16;

const float f_log2E     = log2f(expf(1));

enum //F8 formats
{
    BF8_FMT,
    FP8_FMT
};

enum DATA_TYPE
{
    FP16,
    BF16,
    FP8,
    FP32,
    TYPE_NUM
};

enum DATA_LAYOUT 
{
    LAYOUT_NONE=0,
    LAYOUT_32X32=1,
    LAYOUT_16X16=2,
    LAYOUT_32x8= 3
};

typedef struct TileSize {
    uint32 tileSizeMajor;
    uint32 tileSizeMinor;
    uint32 tileNumOfDWX4;
} TileSize;

uint32 FloatMapToInt(float in)
{
	return *((uint32 *)&in);
}

uint32 round_bf16_significand_rne(bool &is_significand_ovf, uint32 trail_sig_bf16)
{
    is_significand_ovf = false;
    // trail_sig_bf16 is of the form 1.31
    uint32 trail_significand = (trail_sig_bf16 >>  24) & 0x7f;
    uint32 ulp_half_ulp      = (trail_sig_bf16 >>  23) & 0x3;    // 1.31 >> 23 = 1.8
    uint32 or_remain         = (trail_sig_bf16 >>  0) & 0xffffff;
    switch(ulp_half_ulp) {
        case 0:
        case 2:
            break;
        case 1:
            if(or_remain) {
                trail_significand += 1;
            }
            break;
        case 3:
            trail_significand += 1;
            break;
        default:
            break;
    }
    is_significand_ovf = (((trail_significand >> 7) & 0x1) == 0x1);
    return (trail_significand & 0x7f);   // trail_significand is of the form .7
}

uint32 FP32toBFP16(uint32 in, bool clamp=0)
{
    uint32  sign_f32                 = SIGN32(in);
    uint32  trailing_significand_f32 = MAN32(in);
    int32   exp_f32                  = BEXP32(in);
    int32   unbiased_exp_f32         = exp_f32 - 127;
    bool    is_f32_pre_scale_inf     = (exp_f32 == 0xff)
                                        && (trailing_significand_f32 == 0);
    bool    is_f32_pre_scale_nan     = (exp_f32 == 0xff)
                                        && (trailing_significand_f32 != 0);
    bool    is_f32_pre_scale_zero    = ((in & 0x7fffffff) == 0);
    bool    is_f32_pre_scale_denorm  = (exp_f32 == 0x00)
                                        && (trailing_significand_f32 != 0);
    // normalize subnormal number
    if (is_f32_pre_scale_denorm) {
        unbiased_exp_f32 = -126;
        for (uint32 mB = 22; mB >= 0; mB--) {
            if ((trailing_significand_f32 >> mB) != 0) {
                trailing_significand_f32 
                                 = (trailing_significand_f32 << (23 - mB)) & 0x7fffff;
                unbiased_exp_f32 = unbiased_exp_f32 - (23 - mB);
                break;
            }
        }
    }
    // at this point, leading significand bit is always 1 for non-zero input

    // at this point the exponent is the output exponent range

    uint16_t bf16 = 0;

    if (is_f32_pre_scale_inf) {
        bf16 = (sign_f32 << 15) | ((clamp == 0) ? 0x7f80 : 0x7f7f);
    } else if (is_f32_pre_scale_nan) {
        bf16 = (sign_f32 << 15) | 0x7fff;
    } else if (is_f32_pre_scale_zero) {
        bf16 = (sign_f32 << 15) | 0x0;
    } else {
        if (unbiased_exp_f32 < -149) {
            // scaled number is less than bf16 min subnorm; output 0
            bf16 = (sign_f32 << 15) | 0x0;
        } else if (unbiased_exp_f32 < -126) {
            int32 exp_shift     = -126 - unbiased_exp_f32;
            int32 unbiased_exp_bf16
                                = unbiased_exp_f32 + exp_shift;
            assert(unbiased_exp_bf16 == -126);
            uint32 trail_sig_bf16 = (1 << 31) | (trailing_significand_f32 << 8);
            trail_sig_bf16      >>= exp_shift;
            bool   is_sig_ovf   = false;
            trail_sig_bf16        = round_bf16_significand_rne(is_sig_ovf, trail_sig_bf16);
            bf16 = (sign_f32 << 15)
                    | ((is_sig_ovf ? 0x01 : 0x00) << 7)
                    | (trail_sig_bf16 & 0x7f);
        } else if (unbiased_exp_f32 < +128) {
            // scaled number is in bf16 normal range
            //  apply rne
            uint32 biased_exp_bf16
                                = unbiased_exp_f32 + 127;
            uint32 trail_sig_bf16 = (1 << 31) | (trailing_significand_f32 << 8);
            bool   is_sig_ovf   = false;
            trail_sig_bf16        = round_bf16_significand_rne(is_sig_ovf, trail_sig_bf16);
            biased_exp_bf16      += (is_sig_ovf ? 1 : 0);
            if (biased_exp_bf16 == +255) {
                bf16 = (sign_f32 << 15) | ((clamp == 0) ? 0x7f80 : 0x7f7f);
            } else {
                bf16 = (sign_f32 << 15)
                        | ((biased_exp_bf16 & 0xff) << 7)
                        | (trail_sig_bf16 & 0x7f);
            }
        } else {
            // scaled number is greater than bf16 max normal output
            //  clamp to bf16 flt_max/inf based on clamp control
            bf16 = (sign_f32 << 15) | ((clamp == 0) ? 0x7f80 : 0x7f7f);
        }
    }

    return (uint32)bf16;
}
double gaussrand()
{
    static double V1, V2, S;
    static int phase = 0;
    double X;

    //srand((unsigned)time(NULL));
    if ( phase == 0 ) 
    {
        do 
        {
            double U1 = (double)rand() / RAND_MAX;
            double U2 = (double)rand() / RAND_MAX;
            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } 
        while(S >= 1 || S == 0);
        X = V1 * sqrt(-2 * log(S) / S);
    } 
    else
       X = V2 * sqrt(-2 * log(S) / S);

    phase = 1 - phase;
    return X;
}
template<typename TT>
static void fmha_dumpMatrixInHex(TT *buffer,  FILE *file, int m, int n, DATA_TYPE in_type, int transpose = 0)
{
    if(transpose == 0)
    {
        for(int i = 0; i < m ; i++)
        {
            fprintf(file, "R[%04d]: ", i);
            for(int j = 0; j < n ; j++)
            {
                TT value;
                value = buffer[i * n + j];
                
                if((in_type == FP16) || (in_type == BF16))
                   fprintf(file, "0x%04x ", (*((uint32*)&value) & 0xffff));
                else if (in_type == FP8) //FP8
                   fprintf(file, "0x%02x ", (*((uint32*)&value) & 0xff));
                else if (in_type == FP32) //FP32
                   fprintf(file, "0x%08x ", (*((uint32*)&value) & 0xffffffff));
            }
            fprintf(file, "\n");
        }
    }
    else
    {
        for(int j = 0; j < n ; j++)
        {
            fprintf(file, "R[%04d]: ", j);
            for(int i = 0; i < m ; i++)
            {
                TT value;
                value = buffer[i * n + j];
                
                if((in_type == FP16) || (in_type == BF16))
                   fprintf(file, "0x%04x ", (*((uint32*)&value) & 0xffff));
                else if (in_type == FP8) //FP8
                   fprintf(file, "0x%02x ", (*((uint32*)&value) & 0xff));
                else if (in_type == FP32) //FP32
                   fprintf(file, "0x%08x ", (*((uint32*)&value) & 0xffffffff));
            }
            fprintf(file, "\n");
        }
    }
}

template<typename TT>
static void fmha_dump_batch_inHex(TT *buffer, const char *fileName, int batch, int head_num, int m, int n, DATA_TYPE in_type, int transpose = 0)
{
    FILE *file = fopen(fileName, "w+t");

    for(int b = 0; b < batch; b++)
    {
        for(int h = 0; h < head_num; h++)
        {
            fprintf(file, "++++Batch[%04d]---head[%04d]++++: \n", b, h);
            fmha_dumpMatrixInHex(buffer+b*head_num*m*n+h*m*n, file, m, n, in_type, transpose);
        }
    }

    fclose(file);
}

template<typename TT>
static void fmha_batch_init(TT *buffer, int batch, int head_num, int seq_len, int head_dim, DATA_TYPE in_type, int init_pattern = 0, int fp_format = FP8_FMT, bool f8_bias = false)
{
    for(int b = 0; b < batch; b++)
    {
        for(int h = 0; h < head_num; h++)
        {
            for(int s = 0; s < seq_len; s++)
            {
                for(int d = 0; d < head_dim; d++)
                {
                    int offset = b * head_num * seq_len * head_dim  + h * seq_len * head_dim + s * head_dim + d;
 
                    float temp_var;
                    switch (init_pattern)
                    {
                        case 0:
                            temp_var = (float)gaussrand();
                            break;
                        //case 1:
                        //    temp_var = cos(offset);
                        //    break;
                        //case 2:
                        //    temp_var = sin(offset);
                        //    break;
                        //case 3:
                        //    temp_var = cos(offset) + sin(offset);
                        //    break;
                        case 10:
                            temp_var = 0.25;
                            break;
                        case 11:
                            temp_var = 0.01*d;
                            break;
                        case 12:
                            temp_var = 0.01*s;
                            break;
                        default:
                            temp_var = 0;
                            break;
                    }

                    switch(in_type)
                    {
                        case FP16:
                             //buffer[offset] = (uint16)FP32toFP16(FloatMapToInt(temp_var));
                             buffer[offset] = __float2half_rn(temp_var);
                             break;
                        case BF16:
                             buffer[offset] = (uint16)FP32toBFP16(FloatMapToInt(temp_var));
                             //buffer[offset] = __float2half_rn(temp_var);
                             break;
                        //case FP8:
                        //     buffer[offset] = f32_to_fp8(FloatMapToInt(temp_var), 127, fp_format, f8_bias, true, false, 0);
                        //     break;
                        case FP32:
                             buffer[offset] = temp_var;
                             break;
                        default:
                             break;
                    }
                }//head_dim
            }//seq_len
        }//head_num
    }//batch
}

//softmax functions//////
void fmha_softmax_dev(cl_float *a, cl_float s, int m, int n, int start, cl_float *p_max = NULL, cl_float *p_max_delta = NULL, cl_float *p_sum = NULL, DATA_LAYOUT klc = LAYOUT_NONE, DATA_TYPE type = FP16)
{
    float s2E = s * f_log2E;
    int   sum_step = n;

    for(int i = 0; i < m; i++)
    {
        float em  = 0.0;
        float sum = 0.0;
        float klc_tmp_sum0 = 0.0;
        float klc_tmp_sum1 = 0.0;
        float klc_tmp_sum2 = 0.0;
        float klc_tmp_sum3 = 0.0;
        float klc_tmp_sum01 = 0.0;
        float klc_tmp_sum11 = 0.0;
        float klc_tmp_sum21 = 0.0;
        float klc_tmp_sum31 = 0.0;

        if(start)
            em  = a[i * n + 0];
        else
            em  = p_max[i];

        for(int j = 0; j < n; j++)
        {
            em = (em > a[i * n + j]) ? em : a[i * n + j];
        }

        if(!start)
        {
            //save max_delta
            p_max_delta[i] = p_max[i] - em;
            p_max_delta[i] = p_max_delta[i] * s2E;

            //e^(old_max-new_max) * old_sum
            sum = p_sum[i] * (float)pow(2.0, p_max_delta[i]);
            //save max
            p_max[i] = em;
        }
        else if(p_max != NULL)
           p_max[i] = em;

        for(int j = 0; j < n; j++)
        {
            float max_slog2E = em * s2E;

            a[i * n + j] = (float)pow(2.0, (a[i * n + j] *s2E - max_slog2E));

            if (klc == LAYOUT_32X32)
            {
                if (type == FP8)
                {
                    if(j%8 < 4)
                    {
                        if(j%2 == 0)
                           klc_tmp_sum0+=a[i * n + j];
                        else
                           klc_tmp_sum1+=a[i * n + j];
                    }
                    else
                    {
                        if(j%2 == 0)
                           klc_tmp_sum2+=a[i * n + j];
                        else
                           klc_tmp_sum3+=a[i * n + j];
                    }
                    
                    if (j%sum_step == (sum_step-1))
                    {
                        klc_tmp_sum0 += klc_tmp_sum1;
                        klc_tmp_sum2 += klc_tmp_sum3;
                        klc_tmp_sum0 += klc_tmp_sum2;
                    
                        sum += klc_tmp_sum0;
                        klc_tmp_sum0 = 0.0;
                        klc_tmp_sum1 = 0.0;
                        klc_tmp_sum2 = 0.0;
                        klc_tmp_sum3 = 0.0;
                    }
                }
                else
                {
                    if(j%8 < 4)
                        klc_tmp_sum0+=a[i * n + j];
                    else
                        klc_tmp_sum1+=a[i * n + j];
                    
                    if (j%sum_step == (sum_step-1))
                    {
                        klc_tmp_sum0 += klc_tmp_sum1;
                        sum += klc_tmp_sum0;
                        klc_tmp_sum0 = 0.0;
                        klc_tmp_sum1 = 0.0;
                    }
                }
            } else if (klc == LAYOUT_16X16) {
                if (type == FP8) {
                    int step = j%16;
                    if(step < 4) {
                        if (step%2 == 0)
                            klc_tmp_sum0+=a[i * n + j];
                        else
                            klc_tmp_sum01+=a[i * n + j];
                    }
                    else if (step >= 4 && step < 8) {
                        if (step%2 == 0)
                            klc_tmp_sum1+=a[i * n + j];
                        else
                            klc_tmp_sum11+=a[i * n + j];
                    }
                    else if (step >= 8 && step < 12) {
                        if (step%2 == 0)
                            klc_tmp_sum2+=a[i * n + j];
                        else
                            klc_tmp_sum21+=a[i * n + j];
                    }
                    else if (step >= 12 && step < 16) {
                        if (step%2 == 0)
                            klc_tmp_sum3+=a[i * n + j];
                        else
                            klc_tmp_sum31+=a[i * n + j];
                    }

                    if (j%sum_step == (sum_step-1))
                    {
                        klc_tmp_sum0 += klc_tmp_sum01;
                        klc_tmp_sum1 += klc_tmp_sum11;
                        klc_tmp_sum2 += klc_tmp_sum21;
                        klc_tmp_sum3 += klc_tmp_sum31;
                        
                        klc_tmp_sum0 += klc_tmp_sum2;
                        klc_tmp_sum1 += klc_tmp_sum3;
                        klc_tmp_sum0 += klc_tmp_sum1;
                        sum += klc_tmp_sum0;
                        klc_tmp_sum0 = 0.0;
                        klc_tmp_sum1 = 0.0;
                        klc_tmp_sum2 = 0.0;
                        klc_tmp_sum3 = 0.0;
                        klc_tmp_sum01 = 0.0;
                        klc_tmp_sum11 = 0.0;
                        klc_tmp_sum21 = 0.0;
                        klc_tmp_sum31 = 0.0;
                    }
                } else {
                    int step = j%16;
                    if(step < 4)
                        klc_tmp_sum0+=a[i * n + j];
                    else if (step >= 4 && step < 8)
                        klc_tmp_sum1+=a[i * n + j];
                    else if (step >= 8 && step < 12)
                        klc_tmp_sum2+=a[i * n + j];
                    else if (step >= 12 && step < 16)
                        klc_tmp_sum3+=a[i * n + j];

                    if (j%sum_step == (sum_step-1))
                    {
                        klc_tmp_sum0 += klc_tmp_sum2;
                        klc_tmp_sum1 += klc_tmp_sum3;
                        klc_tmp_sum0 += klc_tmp_sum1;
                        sum += klc_tmp_sum0;
                        klc_tmp_sum0 = 0.0;
                        klc_tmp_sum1 = 0.0;
                        klc_tmp_sum2 = 0.0;
                        klc_tmp_sum3 = 0.0;
                    }
                }
            }
            else
                sum = sum + a[i * n + j];
        }

        p_sum[i] = sum;
    }
}

//M*N row-major, N is fast-changing
//transpose==0, dump out M*N row-major
//transpose==1, dump out N*M col-major
template<typename TT>
static void moe_dump_inHex(TT *buffer,  std::string fileName, uint32 exprt, uint32 M, uint32 N, DATA_TYPE in_type, int transpose = 0) {
    FILE *file = fopen(fileName.c_str(), "w+t");
    if (file == NULL) {
        printf("Failed to open file %s\n", fileName.c_str());
        return;
    }
    for(uint32 exprtIdx = 0; exprtIdx < exprt; exprtIdx++) {
        if (exprt > 1) {
            fprintf(file, "++++Expert[%04d]: \n", exprtIdx);
        }
        fmha_dumpMatrixInHex(buffer+exprtIdx*M*N, file, M, N, in_type, transpose);
    }

    fclose(file);
}

static void dump_uint32_in_value(uint32* buffer, FILE *file, uint32 m, uint32 n) {
    for (int b = 0; b < m; b++) {
        for (int k = 0; k < n; k++) {
            fprintf(file, "%10d ", *(buffer + b*n+k));
        }
        fprintf(file, "\n");
    }
}

static void dump_float_in_value(cl_float* buffer, FILE *file, uint32 m, uint32 n) {
    for (int b = 0; b < m; b++) {
        for (int k = 0; k < n; k++) {
            fprintf(file, "%10.8f ", *(buffer + b*n+k));
        }
        fprintf(file, "\n");
    }
}

static void moe_dump_topk_inHex(uint32* TKI_buf,  std::string fileName, uint32 batch, uint32 topk) {
    return moe_dump_inHex(TKI_buf, fileName, 1, batch, topk, FP32);
}

static void moe_dump_topk_inValue(uint32* TKI_buf,  std::string fileName, uint32 batch, uint32 topk) {
    FILE *file = fopen(fileName.c_str(), "w+t");
    if (file == NULL) {
        printf("Failed to open file %s\n", fileName.c_str());
        return;
    }

    dump_uint32_in_value(TKI_buf, file, batch, topk);
    fclose(file);
}

static void moe_weight_softmax(cl_float *weight, uint32 batch, uint32 topk) {
    cl_float scaler = 1;
    cl_float *p_sum = (cl_float*)malloc(batch*sizeof(cl_float));
    int      start  = 1;

    fmha_softmax_dev(weight, scaler, batch, topk, start, NULL, NULL, p_sum);
    //divide by sum
    for (uint32 b = 0; b < batch; b++) {
        for (uint32 t = 0; t < topk; t++) {
            *(weight+b*topk+t) = (*(weight+b*topk+t))/(*(p_sum+b));
        }
    }

    free(p_sum);
}

static void moe_dump_weight_inHex(cl_float* weight,  std::string fileName, uint32 batch, uint32 topk) {
    return moe_dump_inHex(weight, fileName, 1, batch, topk, FP32);
}

static void moe_dump_weight_inValue(cl_float* weight,  std::string fileName, uint32 batch, uint32 topk) {
    FILE *file = fopen(fileName.c_str(), "w+t");
    if (file == NULL) {
        printf("Failed to open file %s\n", fileName.c_str());
        return;
    }

    dump_float_in_value(weight, file, batch, topk);
    fclose(file);
}


//Input M*N row-major, N is fast-changing
//majorInN means for the little tile block, its data fast-changing dimension; it's also fast-changing dimension between tiles
//i think tileSizeMajor is how many points to form 4 words, for FP16, it should be 8
//tileSizeMinor is related to XDL instructions, for 16x16 it should be 16, for 32x8 it should be 32
template<typename TT>
static void moe_shuffle_one(TT *buffer, uint32 M, uint32 N, bool majorInN=true, uint32 tileSizeMajor=8, uint32 tileSizeMinor=16) {
    uint32 tilesInM;
    uint32 tilesInN;
    uint32 tileStride = tileSizeMajor*tileSizeMinor;

    if (majorInN) {
        if (M % tileSizeMinor != 0 || N % tileSizeMajor != 0) {
            printf("Not divisible M:%d,tileSizeMinor:%d N:%d,tileSizeMajor:%d\n", M, tileSizeMinor, N, tileSizeMajor);
            assert(0);
        }
        tilesInM = (M+tileSizeMinor-1)/tileSizeMinor;
        tilesInN = (N+tileSizeMajor-1)/tileSizeMajor;
    } else {
        if (M % tileSizeMajor != 0 || N % tileSizeMinor != 0) {
            printf("Not divisible M:%d,tileSizeMajor:%d N:%d,tileSizeMinor:%d\n", M, tileSizeMajor, N, tileSizeMinor);
            assert(0);
        }
        tilesInM = (M+tileSizeMajor-1)/tileSizeMajor;
        tilesInN = (N+tileSizeMinor-1)/tileSizeMinor;
    }

    TT *shuffle = (TT*) malloc(M*N*sizeof(TT));
    TT *tilePtr;
    uint32 originM;
    uint32 originN;
    for (uint32 tileM = 0; tileM < tilesInM; tileM++) {
        for (uint32 tileN = 0; tileN < tilesInN; tileN++) {
            //first find tile's position
            if (majorInN) {
                tilePtr = shuffle+(tileM*tilesInN+tileN)*tileStride;
            } else {
                tilePtr = shuffle+(tileN*tilesInM+tileM)*tileStride;
            }
            //scan each point inside the tile, remap the position back to original buffer

            for(uint32 m=0; m < tileSizeMinor; m++) {
                for(uint32 k=0; k < tileSizeMajor; k++) {
                    if (majorInN) {
                        originM = tileM * tileSizeMinor + m;
                        originN = tileN * tileSizeMajor + k;
                    } else {
                        originM = tileM * tileSizeMajor + k;
                        originN = tileN * tileSizeMinor + m;
                    }
                     //for each point inside tile, no matter majorInN or not, its fast changing is tileSizeMajor
                     //origin buffer is M*N row-major
                    *(tilePtr + m*tileSizeMajor+k) = *(buffer + originM*N+originN);
                }
            }
        }
    }

    memcpy(buffer, shuffle, sizeof(TT)*M*N);
    free(shuffle);
    return;
}

static void getTileSize(DATA_TYPE type, DATA_LAYOUT layout, TileSize *tileSize) {
    uint32 tileSizeMajor = 8;
    uint32 tileSizeMinor = 16;
    uint32 DWX4 = 16;
    uint32 elemSize = 2;
    uint32 totalSizeDWX4 = 64*4*4;

    if (type == FP16 || type == BF16) {
        elemSize = 2;
    } /*else if (type == FP8 || type == INT8) {
        elemSize = 1;
    } */else if (type == FP32) {
        elemSize = 4;
    }
    tileSizeMajor = DWX4/elemSize;

    if (layout == LAYOUT_16X16) {
        tileSizeMinor = 16; 
    } else if (layout == LAYOUT_32x8 || layout == LAYOUT_32X32) {
        tileSizeMinor = 32;
    }
    tileSize->tileSizeMajor = tileSizeMajor;
    tileSize->tileSizeMinor = tileSizeMinor;
    tileSize->tileNumOfDWX4 = totalSizeDWX4/(tileSizeMajor*tileSizeMinor*elemSize);
}

//Input M*N row-major, N is fast-changing
template<typename TT>
static void moe_shuffle(TT *buffer, uint32 eprt, uint32 M, uint32 N, bool majorInN, DATA_TYPE type, DATA_LAYOUT layout) {
    TileSize tileSize;
    getTileSize(type, layout, &tileSize);

    for (uint32 i = 0; i < eprt; i++) {
        moe_shuffle_one(buffer+i*M*N, M, N, majorInN, tileSize.tileSizeMajor, tileSize.tileSizeMinor);
    }
    return ;
}

//M*N row-major, N is fast-changing
template<typename TT>
static void moe_init(TT *buffer, uint32 eprt, uint32 M, uint32 N, DATA_TYPE in_type, int init_pattern = 0, int fp_format = FP8_FMT, bool f8_bias = false)
{
    return fmha_batch_init(buffer, 1, eprt, M, N, in_type, init_pattern, fp_format, f8_bias);
}

static void moe_topk_init(uint32* TKI_buf, cl_float* weight, uint32 eprt, uint32 batch, uint32 topk, uint32 even_dist = 1) {
    memset(TKI_buf, 0, batch*topk*sizeof(uint32));

    if (even_dist) {
        uint32 *idxLeft     = (uint32*) malloc(eprt*sizeof(uint32));
        uint32 *idxLeftTmp  = (uint32*) malloc(eprt*sizeof(uint32));
        uint32 idxLeftRange = eprt;
        for (uint32 c = 0; c < eprt; c++) {
            idxLeft[c] = c;
        }

        uint32 e;
        uint32 j;
        for (uint32 b = 0; b < batch; b++) {
            for (uint32 t = 0; t < topk; t++) {
                if (idxLeftRange == 0) {
                    //reset for another round
                    for (uint32 c = 0; c < eprt; c++) {
                        idxLeft[c] = c;
                    }
                    idxLeftRange = eprt;
                }

                //actually after reset for another round, it is possible to have duplication
                while(1) {
                    bool duplicate = false;
                    e = rand()%idxLeftRange;
                    for (uint32 i = 0; i < t; i++) {
                        if (*(TKI_buf+b*topk+i) == idxLeft[e]) {
                            duplicate = true;
                        }
                    }

                    if (!duplicate) {
                        *(TKI_buf+b*topk+t) = idxLeft[e];
                        break;
                    }
                }

                //remove idx
                j = 0;
                for (uint32 i = 0; i < idxLeftRange; i++) {
                    if (i != e) {
                        idxLeftTmp[j++] = idxLeft[i];
                    }
                }

                idxLeftRange--;
                if (idxLeftRange != 0) {
                    memcpy(idxLeft, idxLeftTmp, idxLeftRange*sizeof(uint32));
                }
            }
        }

        free(idxLeft);
        free(idxLeftTmp);
    } else {
        bool useSimpleRandom = true; //i don't want to delete matrix random code
        if (useSimpleRandom) {
            for (uint32 token = 0; token < batch; token++) {
                uint32 offset = token*topk;
                *(TKI_buf+offset) = rand()%eprt; //set first without check duplicate

                for (uint32 idx = 1; idx < topk; idx++) {
                    uint32 topkValue;
                    while(1) {
                        topkValue = rand()%eprt;
                        bool duplicate = false;
                        for(uint32 i = 0; i < idx; i++) {
                            if (topkValue == *(TKI_buf+offset+i)) {
                                duplicate = true;
                                break;
                            }
                        }
                        if (!duplicate) {
                            *(TKI_buf+offset+idx) = topkValue;
                            break; //break out of while(1)
                        }
                    }
                }
            }
        }
    }
}

static void moe_twe_ptr_dump_inValue(std::string fileName, uint32 *sorted_token_ids_ptr, cl_float *sorted_weight_buf, uint32 *eprt_slices, uint32 eprt, uint32 sub_X) {
    FILE *file = fopen(fileName.c_str(), "w+t");
    if (file == NULL) {
        printf("Failed to open file %s\n", fileName.c_str());
        return;
    }

    for (uint32 e = 0; e < eprt; e++) {
        uint32 m = eprt_slices[e];
        uint32 n = sub_X;
        fprintf(file, "++++Expert[%04d]: \n", e);
        dump_uint32_in_value(sorted_token_ids_ptr, file, m, n);
        dump_float_in_value(sorted_weight_buf, file, m, n);
        sorted_token_ids_ptr += m*n;
        sorted_weight_buf    += m*n;
    }

    fclose(file);
}

static void moe_twe_ptr_dump_inHex(std::string fileName, uint32 *sorted_token_ids_ptr, cl_float *sorted_weight_buf, uint32 *eprt_slices, uint32 eprt, uint32 sub_X) {
    FILE *file = fopen(fileName.c_str(), "w+t");
    if (file == NULL) {
        printf("Failed to open file %s\n", fileName.c_str());
        return;
    }

    for (uint32 e = 0; e < eprt; e++) {
        uint32 m = eprt_slices[e];
        uint32 n = sub_X;
        fprintf(file, "++++Expert[%04d]: \n", e);
        fmha_dumpMatrixInHex(sorted_token_ids_ptr, file, m, n, FP32, 0);
        fmha_dumpMatrixInHex(sorted_weight_buf, file, m, n, FP32, 0);
        sorted_token_ids_ptr += m*n;
        sorted_weight_buf    += m*n;
    }

    fclose(file);
}

static void moe_twe_ptr_gen(uint32 *sorted_token_ids_ptr, cl_float *sorted_weight_buf, uint32 *sorted_expert_ids_ptr, uint32 &sub_X_cnt, cl_float *W_buf, uint32 *TKI_buf,
                            uint32 batch, uint32 eprt, uint32 topk, uint32 sub_X) {
    uint32 **eprt_tokens            = new uint32*[eprt];
    cl_float **eprt_token_weights   = new cl_float*[eprt];
    uint32 *eprt_slices             = new uint32[eprt]();
    uint32 *eprt_slice_idxs         = new uint32[eprt]();

    //init
    for (uint32 e = 0; e < eprt; e++) {
        eprt_slices[e]          = 1;
        eprt_slice_idxs[e]      = 0;
        eprt_tokens[e]          = new uint32[sub_X]();
        eprt_token_weights[e]   = new cl_float[sub_X];
        for(uint32 x = 0; x < sub_X; x++) {
            eprt_tokens[e][x]           = batch;
            eprt_token_weights[e][x]    = 0;
        }
    }

    //sort
    for (uint32 t = 0; t < batch; t++) {
        for(uint32 k = 0; k < topk; k++) {
            uint32   e  = *(TKI_buf + t*topk+k);
            cl_float w  = *(W_buf + t*topk+k);
            uint32 idx  = eprt_slice_idxs[e];

            //realloc if needed
            if (idx > eprt_slices[e]*sub_X-1) {
                eprt_slices[e]++;
                uint32 newSize          = eprt_slices[e]*sub_X;
                uint32 *eprt_new        = new uint32[newSize]();
                cl_float *eprt_w_new    = new cl_float[newSize];
                for (uint32 idx = (eprt_slices[e]-1)*sub_X; idx < newSize; idx++) {
                    eprt_new[idx]       = batch;
                    eprt_w_new[idx]     = 0;
                }
                memcpy(eprt_new, eprt_tokens[e], (eprt_slices[e]-1)*sub_X*sizeof(uint32));
                memcpy(eprt_w_new, eprt_token_weights[e], (eprt_slices[e]-1)*sub_X*sizeof(cl_float));
                delete []eprt_tokens[e];
                delete []eprt_token_weights[e];
                eprt_tokens[e]          = eprt_new;
                eprt_token_weights[e]   = eprt_w_new;
            }

            eprt_tokens[e][idx] = t;
            eprt_token_weights[e][idx] = w;
            eprt_slice_idxs[e]++;
        }
    }

    //put back
    uint32 *tokens    = sorted_token_ids_ptr;
    cl_float *weights = sorted_weight_buf;
    uint32 *erp_ids   = sorted_expert_ids_ptr;
    for (uint32 e = 0; e < eprt; e++) {
        memcpy(tokens, eprt_tokens[e], sizeof(uint32)*eprt_slices[e]*sub_X);
        tokens += eprt_slices[e]*sub_X;
        memcpy(weights, eprt_token_weights[e], sizeof(cl_float)*eprt_slices[e]*sub_X);
        weights += eprt_slices[e]*sub_X;

        for (uint32 s = 0; s < eprt_slices[e]; s++) {
            erp_ids[s]= e;
            sub_X_cnt++;
        }
        erp_ids += eprt_slices[e];
    }

    //clean up
    for (uint32 e = 0; e < eprt; e++) {
        delete []eprt_tokens[e];
        delete []eprt_token_weights[e];
    }
    delete []eprt_slices;
    delete []eprt_slice_idxs;
    return;
}

#endif