#pragma once
#include <memory>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace internal
{
    struct PointCloudSamplerImpl;
}

class PointCloudSampler
{
public:
    PointCloudSampler(Eigen::Matrix3f K, float voxel_size = 0.1);
    std::vector<Eigen::Vector3f> SamplePoints(cv::Mat depth, float scale);
    size_t SamplePoints(
        cv::Mat rgb, cv::Mat depth, float depth_scale,
        std::vector<Eigen::Vector3f> &out,
        std::vector<Eigen::Vector3f> &outColour);
    std::vector<Eigen::Vector3f> ChooseK(
        std::vector<Eigen::Vector3f> &points,
        int desiredK);
    std::vector<Eigen::Vector<float, 6>> ChooseK(
        std::vector<Eigen::Vector3f> &points,
        std::vector<Eigen::Vector3f> &colour, int desiredK);

private:
    std::shared_ptr<internal::PointCloudSamplerImpl> impl;
};