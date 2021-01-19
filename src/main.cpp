#include "PointCloudSampler.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <pangolin/pangolin.h>
#include <InputFactory/InputFactory.h>

int main(int argc, char **argv)
{
    if (argc == 1)
        return -1;

    Eigen::Matrix3f K;
    K << 580, 0, 320, 0, 580, 240, 0, 0, 1;

    auto baseDir = std::string(argv[1]);
    auto camera = InputFactory::GetInputMethod("tum:" + baseDir);
    PointCloudSampler sampler(K, 0.03);

    pangolin::CreateWindowAndBind("Main", 640, 480);
    glEnable(GL_DEPTH_TEST);
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.1, 1000),
        pangolin::ModelViewLookAt(-0, 0.5, -3, 0, 0, 0, pangolin::AxisY));
    const int UI_WIDTH = 180;
    pangolin::View &d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -640.0f / 480.0f)
                                .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::CreatePanel("ui")
        .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));
    pangolin::Var<bool> a_button("ui.Read Next", false, false);

    cv::Mat rgbImg, depthImg;
    std::vector<Eigen::Vector3f> points;
    std::vector<Eigen::Vector3f> colours;
    while (!pangolin::ShouldQuit())
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (pangolin::Pushed(a_button))
        {
            if (camera->HasImages() && camera->GetNext(rgbImg, depthImg))
            {
                // points = sampler.SamplePoints(depthImg, 5000.0);
                cv::cvtColor(rgbImg, rgbImg, cv::COLOR_RGB2BGR);
                sampler.SamplePoints(rgbImg, depthImg, 5000.0, points, colours);
                // points = sampler.ChooseK(points, 8192);

                std::cout << "points " << points.size() << std::endl;
            }
        }
        d_cam.Activate(s_cam);

        // pangolin::glDrawVertices(points, GL_POINTS);
        pangolin::glDrawColoredVertices(points.size(), points.data(), colours.data(), GL_POINTS);
        glColor3f(1.0, 1.0, 1.0);

        pangolin::FinishFrame();
    }
}