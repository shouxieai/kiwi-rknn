#include <stdio.h>
#include <thread>
#include <opencv2/opencv.hpp>
#include "kiwi-logger.hpp"
#include "kiwi-app-nanodet.hpp"
#include "kiwi-app-scrfd.hpp"
#include "kiwi-infer-rknn.hpp"

using namespace cv;
using namespace std;

void test_pref(const char* model){

    auto infer = rknn::load_infer(model);
    auto tic = kiwi::timestamp_now_float();
    for(int i = 0; i < 100; ++i){
        infer->forward();
    }
    auto toc = kiwi::timestamp_now_float();
    INFO("%s avg time: %f ms", model, (toc - tic) / 100);
}

void scrfd_demo(){

    auto infer = scrfd::create_infer("scrfd_2.5g_bnkps.rknn", 0.4, 0.5);
    auto image = cv::imread("faces.jpg");
    auto box_result = infer->commit(image).get();

    auto tic = kiwi::timestamp_now_float();
    for(int i = 0; i < 100; ++i){
        infer->commit(image).get();
    }
    auto toc = kiwi::timestamp_now_float();
    INFO("scrfd time: %f ms", (toc - tic) / 100);

    for(auto& obj : box_result){
        cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(0, 255, 0), 2);

        auto pl = obj.landmark;
        for(int i = 0; i < 5; ++i, pl += 2){
            cv::circle(image, cv::Point(pl[0], pl[1]), 3, cv::Scalar(0, 0, 255), -1, 16);
        }
    }
    cv::imwrite("scrfd-result.jpg", image);
}

void nanodet_demo(){

    auto infer = nanodet::create_infer("person_m416_plus_V15.rknn", 0.4, 0.5);
    auto image = cv::imread("faces.jpg");
    auto box_result = infer->commit(image).get();

    auto tic = kiwi::timestamp_now_float();
    for(int i = 0; i < 100; ++i){
        infer->commit(image).get();
    }
    auto toc = kiwi::timestamp_now_float();
    INFO("nanodet time: %f ms", (toc - tic) / 100);

    for(auto& obj : box_result){
        cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(0, 255, 0), 2);
    }
    cv::imwrite("nanodet-result.jpg", image);
}

int main(){

    scrfd_demo();
    nanodet_demo();
    return 0;
}