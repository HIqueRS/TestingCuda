
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <ctime>
#include <iostream>

#define BLOCK_SIZE 16

float* CreateMatrix(float tam,int type)
{
    float* m = new float[tam*tam];
    

    if (type == 0)
    {
        for (int i = 0; i < tam*tam; i++)
        {
            m[i] = 2;
        }
    }
    else if( type == 1)
    {
        int n = 1;
        for (int i = 0; i < tam*tam; i++)
        {
            m[i] = n;
            n++;
        }
    }
    else
    {
        for (int i = 0; i < tam * tam; i++)
        {
            m[i] = 0;
        }
    }

    return m;
}

float* MultiplyMatrix(float* a, float* b, float*c,int tam)
{
    float aux;
    for (int i = 0; i < tam; i++)
    {        
        for (int j = 0; j < tam; j++)
        {
            aux = 0;
            for (int k = 0; k < tam; k++)
            {
                aux += a[i * tam + k ] * b[k * tam + j ];
            }
            c[i * tam + j ] = aux;
        }
    }

    return c;
}

void PrintMatrix(float* m,int tam )
{
    for (int i = 0; i < tam; i++)
    {
        std::cout << "{";
        for (int j = 0; j < tam; j++)
        {
            std::cout <<" " << m[ i *tam + j] << " ";
        }
   
        std::cout << "}\n";
    }
}

__device__ float* GetSubMatrix(float * a, int tam, int row, int col)
{
    float* aSub;
    aSub = &a[tam * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return aSub;
}

__global__ void MultiplyGPUMult(float * a, float *b, float *c,int t)
{
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    float* Csub = GetSubMatrix(c, t, blockRow, blockCol);

    float Cvalue = 0;

    int row = threadIdx.y;
    int col = threadIdx.x;

    for (int m = 0; m < t / BLOCK_SIZE; m++)
    {
        float* Asub = GetSubMatrix(a, t, blockRow, m);
        float* Bsub = GetSubMatrix(b, t, m, blockCol);

        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
       
        As[row][col] = Asub[row * t + col];
        Bs[row][col] = Bsub[row * t + col];

        __syncthreads();

        for (int e = 0; e < BLOCK_SIZE; e++)
        {
            Cvalue += As[row][e] * Bs[e][col];
        }
        
        __syncthreads();

    }

    Csub[row * t + col] = Cvalue;
}

__global__ void MultiplyGPU(float* a, float* b, float* c,int t)
{
    
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    float aux =0;

    if (i < t)
    {
        if (j < t)
        {
            for (int k = 0; k < t; k++)
            {
                aux += a[i * t + k] * b[k * t + j];
            }

            c[i * t + j] = aux;
        }
       
    }
   
}

float* CudaSetUp(float* ma, float* mb , float* mc,int tam)
{
    float* dev_ma;
    float* dev_mb;
    float* dev_mc;

    int t;

    int size = tam * tam;   

    
    t = tam;
   
    

    cudaSetDevice(0);

    cudaMalloc((void**)&dev_ma, size * sizeof(float));
    cudaMalloc((void**)&dev_mb, size * sizeof(float));
    cudaMalloc((void**)&dev_mc, size * sizeof(float));
    
      

    cudaMemcpy(dev_ma, ma, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_mb, mb, size * sizeof(float), cudaMemcpyHostToDevice);
   
    

    dim3 dimGrid((tam + BLOCK_SIZE - 1) / BLOCK_SIZE, (tam + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    MultiplyGPU<<<dimGrid,dimBlock>>> (dev_ma, dev_mb, dev_mc,t);
 

    cudaDeviceSynchronize();

   

    cudaMemcpy(mc, dev_mc, size * sizeof(float), cudaMemcpyDeviceToHost);
   

  

    cudaFree(dev_ma);
    cudaFree(dev_mb);
    cudaFree(dev_mc);

    return mc;
}

float* CudaLocalSetUp(float* ma, float* mb, float* mc, int tam)
{
    float* dev_ma;
    float* dev_mb;
    float* dev_mc;

    int t;

    int size = tam * tam;

    t = tam;

    cudaSetDevice(0);

    cudaMalloc((void**)&dev_ma, size * sizeof(float));
    cudaMalloc((void**)&dev_mb, size * sizeof(float));
    cudaMalloc((void**)&dev_mc, size * sizeof(float));

    cudaMemcpy(dev_ma, ma, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_mb, mb, size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimGrid((tam + BLOCK_SIZE - 1) / BLOCK_SIZE, (tam + BLOCK_SIZE - 1) / BLOCK_SIZE);//maybe here is diferent
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    MultiplyGPUMult << <dimGrid, dimBlock >> > (dev_ma, dev_mb, dev_mc, t);

    cudaDeviceSynchronize();

    cudaMemcpy(mc, dev_mc, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_ma);
    cudaFree(dev_mb);
    cudaFree(dev_mc);

    return mc;
}

int main()
{
    
    float* mA;
    float* mB;
    float* mC;
    int tam;
    int in;
    clock_t timeForExecution;

    std::cout << "Escolha o tamanho: \n [1] 128 \n [2] 1024 \n [3] 2048 \n [4] 4096 \n";
    std::cin >> tam;

    switch (tam)
    {
    default:
        tam = 128;
        break;
    case 2:
        tam = 1024;
        break;
    case 3:
        tam = 2048;
        break;
    case 4:
        tam = 4096;
        break;
    }

    mA = CreateMatrix(tam,0);   
    mB = CreateMatrix(tam,1);
    mC = CreateMatrix(tam,2);


    std::cout << "CPU vs GPU vs GPU(local): \n [1] CPU \n [2] GPU \n [3] GPU(local) \n";
    std::cin >> in;

    switch (in)
    {
    default:
        timeForExecution = clock();
        mC = MultiplyMatrix(mA, mB, mC, tam);
        timeForExecution = clock() - timeForExecution;
        break;
    case 2:
        timeForExecution = clock();
        mC = CudaSetUp(mA,mB,mC,tam);
        
        timeForExecution = clock() - timeForExecution;
        break;
    case 3:
        timeForExecution = clock();
        mC = CudaLocalSetUp(mA, mB, mC, tam);
        timeForExecution = clock() - timeForExecution;
        break;
    }
   
    PrintMatrix(mC, tam);
    
    std::cout << "Tempo levado: " << (float)timeForExecution/CLOCKS_PER_SEC << "\n";

    return 0;
}



