#include "PointCloudSampler.h"
#include <algorithm>
#include <numeric>

#if defined(__GNUC__)
#define SafeCall(expr) ___SafeCall(expr, __FILE__, __LINE__, __func__)
#else
#define SafeCall(expr) ___SafeCall(expr, __FILE__, __LINE__)
#endif

static inline void error(const char *error_string, const char *file, const int line, const char *func)
{
    std::cout << "Error: " << error_string << "\t" << file << ":" << line << std::endl;
    exit(0);
}

static inline void ___SafeCall(cudaError_t err, const char *file, const int line, const char *func = "")
{
    if (cudaSuccess != err)
        error(cudaGetErrorString(err), file, line, func);
}

// compare val with the old value stored in *add and write the bigger one to *add
__device__ __forceinline__ void atomicMax(float *add, float val)
{
    int *address_as_i = (int *)add;
    int old = *address_as_i, assumed;
    do
    {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

// compare val with the old value stored in *add and write the smaller one to *add
__device__ __forceinline__ void atomicMin(float *add, float val)
{
    int *address_as_i = (int *)add;
    int old = *address_as_i, assumed;
    do
    {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

namespace internal
{
    struct GridElement
    {
        Eigen::Vector4f pt;
        int count;
    };

    __global__ void FindPointCloudBounds_kernel(cv::cuda::PtrStepSz<Eigen::Vector4f> pcd,
                                                float *bounds)
    {
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        if (x >= pcd.cols || y >= pcd.rows)
            return;

        auto &pt = pcd.ptr(y)[x];

        if (pt[3] > 0)
        {
            atomicMin(&bounds[0], pt[0]);
            atomicMin(&bounds[1], pt[1]);
            atomicMin(&bounds[2], pt[2]);
            atomicMax(&bounds[3], pt[0]);
            atomicMax(&bounds[4], pt[1]);
            atomicMax(&bounds[5], pt[2]);
        }
    }

    __global__ void ComputeVertexMap_kernel(cv::cuda::PtrStepSz<float> depth,
                                            cv::cuda::PtrStep<Eigen::Vector4f> vmap,
                                            float invfx, float invfy, float cx, float cy)
    {
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        if (x >= depth.cols || y >= depth.rows)
            return;

        auto &z = depth.ptr(y)[x];
        if (z > 0 && z < 10.0)
        {
            Eigen::Vector4f pt;
            pt[0] = (x - cx) * invfx * z;
            pt[1] = (y - cy) * invfy * z;
            pt[2] = z;
            pt[3] = 1;
            vmap.ptr(y)[x] = pt;
        }
        else
        {
            vmap.ptr(y)[x] = Eigen::Vector4f(0, 0, 0, 0);
        }
    }

    __global__ void ComputeCentroids_kernel(cv::cuda::PtrStepSz<Eigen::Vector4f> vmap,
                                            Eigen::Vector4f *grid,
                                            float voxelSize, Eigen::Vector3i gridSize,
                                            Eigen::Vector3i minElem)
    {
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        if (x >= vmap.cols || y >= vmap.rows)
            return;

        Eigen::Vector4f pt = vmap.ptr(y)[x];
        if (pt[3] > 0.5)
        {
            Eigen::Vector3i coord = (pt / voxelSize).head<3>().cast<int>() - minElem;
            Eigen::Vector4f cell = grid[gridSize[0] * gridSize[1] * coord[2] + gridSize[0] * coord[1] + coord[0]];
            grid[gridSize[0] * gridSize[1] * coord[2] + gridSize[0] * coord[1] + coord[0]] = cell + pt;
        }
    }

    __global__ void ComputeCentroidsColour_kernel(cv::cuda::PtrStepSz<Eigen::Vector4f> vmap,
                                                  cv::cuda::PtrStepSz<Eigen::Vector3f> colour,
                                                  Eigen::Vector4f *grid,
                                                  Eigen::Vector3f *gridColour,
                                                  float voxelSize, Eigen::Vector3i gridSize,
                                                  Eigen::Vector3i minElem)
    {
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        if (x >= vmap.cols || y >= vmap.rows)
            return;

        Eigen::Vector4f pt = vmap.ptr(y)[x];
        if (pt[3] > 0.5)
        {
            Eigen::Vector3f rgb = colour.ptr(y)[x];
            Eigen::Vector3i coord = (pt / voxelSize).head<3>().cast<int>() - minElem;
            int idx = gridSize[0] * gridSize[1] * coord[2] + gridSize[0] * coord[1] + coord[0];
            Eigen::Vector4f cell = grid[idx];
            Eigen::Vector3f cellColour = gridColour[idx];
            grid[idx] = cell + pt;
            gridColour[idx] = cellColour + rgb;
        }
    }

    __global__ void RemoveOutliers_kernel(cv::cuda::PtrStepSz<Eigen::Vector4f> vmap,
                                          Eigen::Vector3f centroid, float sampleRadius)
    {
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        if (x >= vmap.cols || y >= vmap.rows)
            return;

        auto &pt = vmap.ptr(y)[x];
        if (pt[3] > 0)
        {
            if ((pt.head<3>() - centroid).norm() > sampleRadius)
                pt = Eigen::Vector4f::Zero();
        }
    }

    struct PointCloudSamplerImpl
    {
        Eigen::Matrix3f K;
        float voxelSize;
        cv::Mat depthF, rgbF;
        cv::cuda::GpuMat depth_gpu;
        cv::cuda::GpuMat rgb_gpu;
        cv::cuda::GpuMat vmap;

        PointCloudSamplerImpl(Eigen::Matrix3f _K, float voxel_size)
            : K(_K), voxelSize(voxel_size)
        {
        }

        void BackProjectDepth(cv::Mat depth, float scale)
        {
            depth.convertTo(depthF, CV_32FC1, 1.0 / scale);
            depth_gpu.upload(depthF);

            if (vmap.empty())
                vmap = cv::cuda::GpuMat(depthF.size(), CV_32FC4);

            dim3 block(16, 8);
            dim3 grid(cv::divUp(depth.cols, block.x), cv::divUp(depth.rows, block.y));

            ComputeVertexMap_kernel<<<grid, block>>>(depth_gpu, vmap, 1.0 / K(0, 0), 1.0 / K(1, 1), K(0, 2), K(1, 2));

            SafeCall(cudaDeviceSynchronize());
            SafeCall(cudaGetLastError());
        }

        void ComputeCloudCenteroid(float sampleRadius)
        {
            cv::cuda::GpuMat temp;
            cv::cuda::reduce(vmap, temp, 0, cv::REDUCE_SUM);
            cv::cuda::reduce(temp, temp, 1, cv::REDUCE_SUM);
            cv::Mat temp_host(temp);

            Eigen::Vector3f centroid;
            centroid[0] = temp_host.ptr<float>(0)[0];
            centroid[1] = temp_host.ptr<float>(0)[1];
            centroid[2] = temp_host.ptr<float>(0)[2];
            int count = temp_host.ptr<float>(0)[3];
            centroid = centroid / count;

            dim3 block(16, 8);
            dim3 grid(cv::divUp(vmap.cols, block.x), cv::divUp(vmap.rows, block.y));

            RemoveOutliers_kernel<<<grid, block>>>(vmap, centroid, sampleRadius);

            SafeCall(cudaDeviceSynchronize());
            SafeCall(cudaGetLastError());
        }

        void ComputeCloudBoundary(Eigen::Vector3i &minBound, Eigen::Vector3i &maxBound)
        {
            float *pointBounds;
            cudaMalloc((void **)&pointBounds, sizeof(float) * 6);
            cudaMemset(&pointBounds[0], 100, sizeof(float) * 3);
            cudaMemset(&pointBounds[3], 0, sizeof(float) * 3);

            dim3 block(16, 8);
            dim3 grid(cv::divUp(vmap.cols, block.x), cv::divUp(vmap.rows, block.y));

            FindPointCloudBounds_kernel<<<grid, block>>>(vmap, pointBounds);

            SafeCall(cudaDeviceSynchronize());
            SafeCall(cudaGetLastError());

            float bounds_cpu[6];
            SafeCall(cudaMemcpy(bounds_cpu, pointBounds, sizeof(float) * 6, cudaMemcpyDeviceToHost));
            SafeCall(cudaFree(pointBounds));

            int gridBounds_cpu[6];
            gridBounds_cpu[0] = (int)(bounds_cpu[0] / voxelSize);
            gridBounds_cpu[1] = (int)(bounds_cpu[1] / voxelSize);
            gridBounds_cpu[2] = (int)(bounds_cpu[2] / voxelSize);
            gridBounds_cpu[3] = (int)(bounds_cpu[3] / voxelSize);
            gridBounds_cpu[4] = (int)(bounds_cpu[4] / voxelSize);
            gridBounds_cpu[5] = (int)(bounds_cpu[5] / voxelSize);

            if (gridBounds_cpu[0] < 0)
                gridBounds_cpu[0] -= 1;
            if (gridBounds_cpu[1] < 0)
                gridBounds_cpu[1] -= 1;
            if (gridBounds_cpu[2] < 0)
                gridBounds_cpu[2] -= 1;
            if (gridBounds_cpu[3] < 0)
                gridBounds_cpu[3] -= 1;
            if (gridBounds_cpu[4] < 0)
                gridBounds_cpu[4] -= 1;
            if (gridBounds_cpu[5] < 0)
                gridBounds_cpu[5] -= 1;

            minBound = Eigen::Vector3i(gridBounds_cpu[0], gridBounds_cpu[1], gridBounds_cpu[2]);
            maxBound = Eigen::Vector3i(gridBounds_cpu[3], gridBounds_cpu[4], gridBounds_cpu[5]);
        }

        std::vector<Eigen::Vector3f> SamplePoints(cv::Mat depth, float depth_scale)
        {
            int cols = depth.cols;
            int rows = depth.rows;

            BackProjectDepth(depth, depth_scale);
            ComputeCloudCenteroid(1.5);

            Eigen::Vector3i gridBoundMax, gridBoundMin;
            ComputeCloudBoundary(gridBoundMin, gridBoundMax);
            Eigen::Vector3i gridSize = gridBoundMax - gridBoundMin + Eigen::Vector3i(1, 1, 1);

            int totalCell = gridSize.prod();
            Eigen::Vector4f *voxelGrid;
            SafeCall(cudaMalloc((void **)&voxelGrid, sizeof(Eigen::Vector4f) * totalCell));
            SafeCall(cudaMemset(voxelGrid, 0, sizeof(Eigen::Vector4f) * totalCell));

            dim3 block(16, 8);
            dim3 grid(cv::divUp(cols, block.x), cv::divUp(rows, block.y));

            ComputeCentroids_kernel<<<grid, block>>>(vmap, voxelGrid, voxelSize, gridSize, gridBoundMin);

            SafeCall(cudaDeviceSynchronize());
            SafeCall(cudaGetLastError());

            Eigen::Vector4f *voxelGrid_cpu = new Eigen::Vector4f[totalCell];
            memset(voxelGrid_cpu, 0, sizeof(Eigen::Vector3f) * totalCell);
            SafeCall(cudaMemcpy(voxelGrid_cpu, voxelGrid, sizeof(Eigen::Vector4f) * totalCell, cudaMemcpyDeviceToHost));
            SafeCall(cudaFree(voxelGrid));

            std::vector<Eigen::Vector3f> out;
            for (int i = 0; i < totalCell; ++i)
            {
                Eigen::Vector4f cell = voxelGrid_cpu[i];
                if (cell[3] > 0)
                {
                    out.push_back(cell.head<3>() / cell[3]);
                }
            }

            delete voxelGrid_cpu;
            return out;
        }

        size_t SamplePoints(cv::Mat rgb, cv::Mat depth, float depth_scale,
                            std::vector<Eigen::Vector3f> &out,
                            std::vector<Eigen::Vector3f> &outColour)
        {
            rgb.convertTo(rgbF, CV_32FC3, 1.0 / 255);
            rgb_gpu.upload(rgbF);

            int cols = depth.cols;
            int rows = depth.rows;

            BackProjectDepth(depth, depth_scale);
            ComputeCloudCenteroid(1.5);

            Eigen::Vector3i gridBoundMax, gridBoundMin;
            ComputeCloudBoundary(gridBoundMin, gridBoundMax);
            Eigen::Vector3i gridSize = gridBoundMax - gridBoundMin + Eigen::Vector3i(1, 1, 1);

            int totalCell = gridSize.prod();
            Eigen::Vector4f *voxelGrid;
            Eigen::Vector3f *voxelGridColour;
            SafeCall(cudaMalloc((void **)&voxelGrid, sizeof(Eigen::Vector4f) * totalCell));
            SafeCall(cudaMalloc((void **)&voxelGridColour, sizeof(Eigen::Vector3f) * totalCell));
            SafeCall(cudaMemset(voxelGrid, 0, sizeof(Eigen::Vector4f) * totalCell));
            SafeCall(cudaMemset(voxelGridColour, 0, sizeof(Eigen::Vector3f) * totalCell));

            dim3 block(16, 8);
            dim3 grid(cv::divUp(cols, block.x), cv::divUp(rows, block.y));

            ComputeCentroidsColour_kernel<<<grid, block>>>(vmap, rgb_gpu, voxelGrid, voxelGridColour, voxelSize, gridSize, gridBoundMin);

            SafeCall(cudaDeviceSynchronize());
            SafeCall(cudaGetLastError());

            Eigen::Vector4f *voxelGrid_cpu = new Eigen::Vector4f[totalCell];
            Eigen::Vector3f *voxelGridColour_cpu = new Eigen::Vector3f[totalCell];

            memset(voxelGrid_cpu, 0, sizeof(Eigen::Vector4f) * totalCell);
            memset(voxelGridColour_cpu, 0, sizeof(Eigen::Vector3f) * totalCell);

            SafeCall(cudaMemcpy(voxelGrid_cpu, voxelGrid, sizeof(Eigen::Vector4f) * totalCell, cudaMemcpyDeviceToHost));
            SafeCall(cudaMemcpy(voxelGridColour_cpu, voxelGridColour, sizeof(Eigen::Vector3f) * totalCell, cudaMemcpyDeviceToHost));

            SafeCall(cudaFree(voxelGrid));
            SafeCall(cudaFree(voxelGridColour));

            out.clear();
            outColour.clear();

            for (int i = 0; i < totalCell; ++i)
            {
                Eigen::Vector4f cell = voxelGrid_cpu[i];
                Eigen::Vector3f cellColour = voxelGridColour_cpu[i];
                int weight = cell[3];
                if (weight > 0)
                {
                    Eigen::Vector3f pt = cell.head<3>() / weight;
                    Eigen::Vector3f colour = cellColour / weight;
                    // std::cout << colour.transpose() << std::endl;
                    out.push_back(pt);
                    outColour.push_back(colour);
                }
            }

            delete voxelGrid_cpu;
            delete voxelGridColour_cpu;

            return out.size();
        }

        std::vector<Eigen::Vector3f> ChooseK(std::vector<Eigen::Vector3f> &points, int desiredK)
        {
            int currentK = points.size();
            std::vector<Eigen::Vector3f> out;
            if (currentK > desiredK)
            {
                // randomly select a subset
                std::vector<size_t> indices(currentK);
                std::iota(indices.begin(), indices.end(), 0);
                std::random_shuffle(indices.begin(), indices.end());
                for (int i = 0; i < desiredK; ++i)
                {
                    out.push_back(points[indices[i]]);
                }
            }
            else if (currentK < desiredK)
            {
                // Pad the vector with (zero or ones)?
                out = points;
                while (out.size() < desiredK)
                {
                    out.push_back(Eigen::Vector3f::Zero());
                }
            }

            return out;
        }

        std::vector<Eigen::Vector<float, 6>> ChooseK(
            std::vector<Eigen::Vector3f> &points,
            std::vector<Eigen::Vector3f> &colour, int desiredK)
        {
            int currentK = points.size();
            std::vector<Eigen::Vector<float, 6>> out;
            if (currentK > desiredK)
            {
                // randomly select a subset
                std::vector<size_t> indices(currentK);
                std::iota(indices.begin(), indices.end(), 0);
                std::random_shuffle(indices.begin(), indices.end());
                for (int i = 0; i < desiredK; ++i)
                {
                    Eigen::Vector<float, 6> elem;
                    elem.head<3>() = points[indices[i]];
                    elem.tail<3>() = colour[indices[i]];
                    out.push_back(elem);
                }
            }
            else if (currentK < desiredK)
            {
                // Pad the vector with (zero or ones)?
                // out = points;
                // while (out.size() < desiredK)
                // {
                //     out.push_back(Eigen::Vector<float, 6>::Zero());
                // }
            }

            return out;
        }
    };
} // namespace internal

PointCloudSampler::PointCloudSampler(Eigen::Matrix3f K, float voxel_size)
    : impl(new internal::PointCloudSamplerImpl(K, voxel_size))
{
}

std::vector<Eigen::Vector3f> PointCloudSampler::SamplePoints(cv::Mat depth, float scale)
{
    return impl->SamplePoints(depth, scale);
}

size_t PointCloudSampler::SamplePoints(cv::Mat rgb, cv::Mat depth, float scale,
                                       std::vector<Eigen::Vector3f> &out,
                                       std::vector<Eigen::Vector3f> &outColour)
{
    return impl->SamplePoints(rgb, depth, scale, out, outColour);
}

std::vector<Eigen::Vector3f> PointCloudSampler::ChooseK(std::vector<Eigen::Vector3f> &points, int desiredK)
{
    return impl->ChooseK(points, desiredK);
}

std::vector<Eigen::Vector<float, 6>> PointCloudSampler::ChooseK(std::vector<Eigen::Vector3f> &points,
                                                                std::vector<Eigen::Vector3f> &colour, int desiredK)
{
    return impl->ChooseK(points, colour, desiredK);
}