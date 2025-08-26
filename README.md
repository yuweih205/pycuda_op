## 运行本项目有两种办法
一：
1、python ops_loader.py会自动生成so包

2、由于运行环境不同，请自行配置ops_loader.py的编译参数 
    -- a800为例 compute_90改为compute_80，其余亦同

3、可通过环境变量export PYTHONPATH=/path/to/compiled_ops/*.so:$PYTHONPATH
    例如:export PYTHONPATH=/mnt/fast-disks/nfs/hyw/pycuda/compiled_ops:$PYTHONPATH 
    -- 以前者为例，使用方法为：
        import pycuda_ops
        ****省略****
        pycuda_ops.sum_pooling_fw(*,*,*,*,*) //sum_pooling_fw可替换包中任意方法

二：编译成wheel包调用
1、根目录下 
    pip install -v --no-build-isolation . 优先推荐不隔离安装 
    或 pip install . 

2、已开启自动检测架构 如未识别架构
    export TORCH_CUDA_ARCH_LIST="8.0"   8.0为a800 9.0H100

3、使用方法为 
    from pycuda_op import sum_pooling_fw 
    ****省略****     
    sum_pooling_fw(*,*,*,*,*)


## sumpooling解析
1、前向
    void sum_pooling_fw_launcher(const Tensor& values_bf16,  //写入数据
                             const Tensor& offsets_i32,        
                             const Tensor& dims_i32,           //有多少维度
                             Tensor&       output_f32,         //output张量
                             const Tensor& output_offsets_i32) //每个outout写的位置

2、反向
    static void launch_impl(const at::Tensor& go,    //写入数据
                            at::Tensor& gv,          //写回数据
                            const at::Tensor& offs, //相当于offset
                            const at::Tensor& dims, 
                            const at::Tensor& ooffs) //每个前向out写的位置 相当于output_offset

                            a800 前向 0.64  反向 0.947ms  4096x128x[16,128]x[1,10] bf16->fp32
                            H100 前向 0.319 反向 0.353ms 1.7/1.6TB  fp32->bf16
                                

前向支持bf16/fp32 -> fp32  反向支持fp32 -> fp32/bf16
 
H100
全fp32 前向 0.905ms 2.5T  反向 0.715ms 2.45T  4096x128x168x[1,10]
bf16->fp32   0.492ms 2.4T 反向 0.362  2.5T   4096x128x168x[1,10] 
4096x128x[4,128]x[1,10]  bf16->fp32  前向0.64ms  fp32->bf16 反向0.947ms  a800
4096x128x168x[1,10]   全fp32 前向 0.905ms 2.5T  反向 0.715ms 2.45T  H100上带宽利用率70-80 
4096x128x168x[1,10]   全fp16 前向 0.643ms 1.8T  反向 1ms 1.15T 

之前测的数据和时间对不上 重测
H100
bf16->bf16   0.306ms 1.88T     0.363    1.58T     4096x128x[16,128]x[1,10]
bf16->fp32   0.324ms 1.78T     0.354ms  1.6T      4096x128x[16,128]x[1,10]
fp32->fp32   0.414ms 2.39T     0.386ms  2.56T     4096x128x[16,128]x[1,10]