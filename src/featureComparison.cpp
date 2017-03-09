//============================================================================
// Name        : pin_detection.cpp
// Author      :
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#define video
bool sift = 0; //else surf
bool enable_gpu = 1;

#include <stdio.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
//#include <highgui.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "../CudaSift/cudaImage.h"
#include "../CudaSift/cudaSift.h"
//#include "../CudaSift/geomFuncs.cpp"

using namespace cv;

int camera_device = 0;

int numOctaves = 5; /* Number of octaves in Gaussian pyramid */
float initBlur = 1.0f; /* Amount of initial Gaussian blurring in standard deviations */
float thresh = 3.5f; /* Threshold on difference of Gaussians for feature pruning */
float minScale = 0.0f; /* Minimum acceptable scale to remove fine-scale features */
bool upScale = true; /* Whether to upscale image before extraction */

//int ImproveHomography(SiftData &data, float *homography, int numLoops, float minScore, float maxAmbiguity, float thresh);

int main(int, char**)
{
  // Open the default camera
  VideoCapture cap(camera_device);
  // Check if camera works
  if (!cap.isOpened())
    return -1;

  namedWindow("Matches", CV_GUI_NORMAL);

  // Load images of objects to search for
  String folder_path = "/home/teddy/objects";
  std::vector<String> object_files;

  std::vector<Mat> object_img;

  glob(folder_path, object_files);

  int imported_objects = 0;
  for (size_t i=0; i < object_files.size(); i++)
  {
  object_img.push_back(imread(object_files.at(i), 1));
  if (object_img.at(i).data == NULL)
  {
    std::cout << "An object file is corrupt." << std::endl;
  }
  else
  {
    imported_objects++;
  }
  }

  if (imported_objects == 0)
  {
    std::cout << "No objects imported, terminating." << std::endl;
    return 0;
  }
  else
  {
    std::cout << "Imported " << imported_objects << " objects." << std::endl;
  }

  std::vector<Mat> object_bw_img;
//  object_bw_img.assign(imported_objects, object_img.at(0));
  for (size_t i=0; i < imported_objects; i++)
    {
      object_bw_img.push_back(Mat(object_img.at(0).cols, object_img.at(0).rows, CV_8UC3));
      cvtColor(object_img.at(0), object_bw_img.back(), COLOR_BGR2GRAY);
    }

  InitCuda(camera_device);

  cv::Mat object_sift_img;

  CudaImage object_cuda_sift_img;


  object_bw_img.at(0).convertTo(object_sift_img, CV_32FC1);
  object_cuda_sift_img.Allocate(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT),
                         cap.get(CV_CAP_PROP_FRAME_WIDTH), false, NULL, (float*)object_sift_img.data);

  // Get a frame to initialize cudaSift
//  cv::Mat frame_sift_img(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT), CV_32FC1);
  cv::Mat frame_sift_img;
  if (enable_gpu && sift)
  {
    Mat ini_frame, ini_frame_bw;
#ifdef video
  while (true)
  {
    // Get a frame from camera
    cap >> ini_frame;
    if (ini_frame.cols < cap.get(CV_CAP_PROP_FRAME_WIDTH))
      continue;
    else break;
  }
  std::cout << "found ini_frame" << std::endl;
#else
    ini_frame = imread("/home/teddy/env.jpg", 1);
#endif

    cvtColor(ini_frame, ini_frame_bw, COLOR_BGR2GRAY);
    ini_frame_bw.convertTo(frame_sift_img, CV_32FC1);
  }

  CudaImage frame_cuda_sift_img;
  frame_cuda_sift_img.Allocate(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT),
                         cap.get(CV_CAP_PROP_FRAME_WIDTH), false, NULL, (float*)frame_sift_img.data);

  // Reserve memory on GPU as allocating is slow
  SiftData frame_sift_data, object_sift_data;
  InitSiftData(frame_sift_data, 2500000, true, true);
  InitSiftData(object_sift_data, 2500000, true, true);




  cv::Ptr<Feature2D> feature_extractor;

  int min_hessian = 400;
  std::vector<KeyPoint> frame_keypoints;
  std::vector<std::vector<KeyPoint> > object_keypoints;
  Mat frame_descriptors;
  std::vector<std::vector<KeyPoint> >object_descriptors;

  std::vector<cuda::GpuMat> object_gpu_img;
  for (size_t i=0; i < object_bw_img.size(); i++)
  {
    object_gpu_img.push_back(cuda::GpuMat(object_bw_img.at(i)));
  }

  cuda::GpuMat frame_gpu_keypoints, frame_gpu_descriptors;
  std::vector<cuda::GpuMat> object_gpu_keypoints, object_gpu_descriptors;

  cuda::SURF_CUDA* surf;

  if (sift)
  {
    if (enable_gpu)
    {
//      object_bw_img.convertTo(object_sift_img, CV_32FC1);
      std::cout << "cols " << object_sift_img.cols << std::endl;
      object_cuda_sift_img.Download();
      std::cout << "test" << std::endl;
      ExtractSift(object_sift_data, object_cuda_sift_img, numOctaves, initBlur, thresh, minScale, upScale);
      std::cout << "Found " << object_sift_data.numPts << " sift points in object."<< std::endl;
    }
    else
    {
    // Create a Sift extractor object
    feature_extractor = xfeatures2d::SIFT::create();
    for (size_t i=0; i < object_bw_img.size(); i++)
    {
    feature_extractor->detect(object_bw_img.at(i), object_keypoints.at(i));
    feature_extractor->compute(object_bw_img.at(i), object_keypoints.at(i), object_descriptors.at(i));
    }
    }
  }
  else // SURF
  {
    if (enable_gpu)
    {
      surf = new cuda::SURF_CUDA(min_hessian);
      for (size_t i=0; i < object_bw_img.size(); i++)
      {
      surf->operator()( object_gpu_img.at(i), cuda::GpuMat(), object_gpu_keypoints.at(i), object_gpu_descriptors.at(i));
      surf->downloadKeypoints(object_gpu_keypoints.at(i), object_keypoints.at(i));
      }
    }
    else
    {
    feature_extractor = xfeatures2d::SURF::create(min_hessian);
    for (size_t i=0; i < object_bw_img.size(); i++)
    {
    feature_extractor->detect(object_bw_img.at(i), object_keypoints.at(i));
    feature_extractor->compute(object_bw_img.at(i), object_keypoints.at(i), object_descriptors.at(i));
    }
    }
  }

//  feature_extractor->compute(object_bw_img, object_keypoints, object_descriptors);

  clock_t clock_now = clock();
  clock_t clock_total, clock_extraction, clock_matching, clock_homography;
  int clocks_per_ms = CLOCKS_PER_SEC / 1000;

#ifdef video
  while (true)
  {
#endif

    Mat frame_img, frame_bw_img, output_frame;

#ifdef video
    // Get a frame from camera
    cap >> frame_img;
    if (frame_img.cols < 1)
      continue;
#else
    frame_img = imread("/home/teddy/env.jpg", 1);
#endif

    //orig_frame.copyTo(output_frame);

    cvtColor(frame_img, frame_bw_img, COLOR_BGR2GRAY);

    clock_extraction = clock();

    if (sift)
    {
      if (enable_gpu)
      {
        frame_bw_img.convertTo(frame_sift_img, CV_32FC1);
        frame_cuda_sift_img.Download();
        ExtractSift(frame_sift_data, frame_cuda_sift_img, numOctaves, initBlur, thresh, minScale, upScale);
        std::cout << "Found " << frame_sift_data.numPts << " sift points in frame."<< std::endl;
      }
      else
      {
      feature_extractor->detect(frame_bw_img, frame_keypoints);
      feature_extractor->compute(frame_bw_img, frame_keypoints, frame_descriptors);
      }
    }
    else // SURF
    {
      if (enable_gpu)
      {
        cuda::GpuMat frame_gpu_img(frame_bw_img);
        surf->operator()( frame_gpu_img, cuda::GpuMat(), frame_gpu_keypoints, frame_gpu_descriptors );
      }
      else
      {
      feature_extractor->detect(frame_bw_img, frame_keypoints);
      feature_extractor->compute(frame_bw_img, frame_keypoints, frame_descriptors);
      }
    }
//    if (enable_gpu)
//    {
//      if (sift)
//      {
//        frame_bw_img.convertTo(sift_img, CV_32FC1);
//        cuda_sift_img.Download();
//        ExtractSift(frame_sift_data, cuda_sift_img, numOctaves, initBlur, thresh, minScale, upScale);
//      }
//      else // SURF
//      {
////        cuda::GpuMat object_gpu_img(object_bw);
//        cuda::GpuMat img_scene_Gpu(scene_bw);
//        cuda::GpuMat keypoints_scene_Gpu, keypoints_object_Gpu;
//        cuda::GpuMat descriptors_scene_Gpu, descriptors_object_Gpu;
//
//        cuda::SURF_CUDA surf(min_hessian);
//        //surf( object_gpu_img, cuda::GpuMat(), object_gpu_keypoints, object_gpu_descriptors );
//        surf(img_scene_Gpu, cuda::GpuMat(), keypoints_scene_Gpu, descriptors_scene_Gpu);
//        std::cout << "Surf done." << std::endl;
//      }
//    }
//    else
//    {
//      if (sift)
//      {
//        // Create vector to store the frame_keypoints
//        feature_extractor->detect(frame_bw_img, keypoints);
//      }
//      else // SURF
//      {
//        feature_extractor->detect(frame_bw_img, keypoints);
//      }
//
//      feature_extractor->compute(frame_bw_img, keypoints, frame_descriptors);
//    }
    std::cout << "Feature extraction: " << (int)(double(clock() - clock_extraction) / clocks_per_ms) << " ms" << std::endl;

//  imshow("Output", output_frame);

//    BFMatcher matcher(2, true);
////    matcher.create(4, true);
//    std::vector<DMatch> matches;
//    matches.clear();
//    matcher.match(object_descriptors, descriptors, matches);

    clock_matching = clock();

    std::vector<std::vector<DMatch> > good_matches;
//    std::vector<DMatch> matches;
    std::vector<std::vector<std::vector< DMatch> > > matches;

    if (enable_gpu)
    {
      if (sift)
      {
        MatchSiftData(frame_sift_data, object_sift_data);
      }
      else
      {
        Ptr< cuda::DescriptorMatcher > matcher = cuda::DescriptorMatcher::createBFMatcher();
//        std::vector< std::vector< DMatch> > matches;
        for (size_t i=0; i < object_bw_img.size(); i++)
        {
          matches.push_back(std::vector<std::vector<DMatch> >());
          matcher->knnMatch(object_gpu_descriptors.at(i), frame_gpu_descriptors, matches.back(), 2);
          surf->downloadKeypoints(frame_gpu_keypoints, frame_keypoints);
        }
      }
    }
    else
    {
      FlannBasedMatcher matcher;

      for (size_t i=0; i < object_bw_img.size(); i++)
      {
        matcher.knnMatch(object_descriptors.at(i), frame_descriptors, matches.at(i), 2);
      }
    }

//    double max_dist = 0;
//    double min_dist = 100;
//
//    for (int i = 0; i < object_descriptors.rows; i++)
//    {
//      double dist = matches[i].distance;
//      if (dist < FLT_EPSILON)
//        continue;
//      if (dist < min_dist)
//        min_dist = dist;
//      if (dist > max_dist)
//        max_dist = dist;
//    }
//
//    for (int i = 0; i < object_descriptors.rows; i++)
//    {
//      if (matches[i].distance < 3 * min_dist)
//      {
//        good_matches.push_back(matches[i]);
//      }
//    }

    if (!(enable_gpu && sift))
    {
      for (size_t i=0; i < object_bw_img.size(); i++)
      {
        for (size_t k = 0; k < std::min(object_keypoints.at(i).size()-1, matches.at(i).size()); k++)
        {
                if ( (matches[i][k][0].distance < 0.6*(matches[i][k][1].distance)) &&
                                ((int)matches[i][k].size() <= 2 && (int)matches[i][k].size()>0) )
                {
                        // take the first result only if its distance is smaller than 0.6*second_best_dist
                        // that means this descriptor is ignored if the second distance is bigger or of similar
                        good_matches[i].push_back(matches[i][k][0]);
                }
        }
      }
    }



    std::cout << "Feature matching: " << (int)(double(clock() - clock_matching) / clocks_per_ms) << " ms" << std::endl;
//
//    std::cout << "Found " << object_keypoints.size() << " object_keypoints" << std::endl;
//    std::cout << "Found " << frame_keypoints.size() << " frame_keypoints" << std::endl;
//    std::cout << "Found " << matches.size() << " matches" << std::endl;
//    std::cout << "Found " << good_matches.size() << " good matches" << std::endl;
    std::cout << std::endl;

#ifdef video
    if (!(enable_gpu && sift))
        {
          if (good_matches.at(0).size() < 1)
          {
            std::cout << "No matches found for first object." << std::endl;
            continue;
          }
        }
#endif

    Mat img_matches = Mat::zeros(frame_img.size(), CV_8UC3);
    if (good_matches.at(0).size() > 0)
    {
    drawKeypoints(frame_img, frame_keypoints, output_frame, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    drawMatches(object_img.at(0), object_keypoints.at(0), frame_img, frame_keypoints, good_matches.at(0), img_matches);
    }



    clock_homography = clock();
    std::vector<Mat> H;

    for (size_t i=0; i < object_bw_img.size(); i++)
    {
      std::vector<Point2f> obj;
      std::vector<Point2f> env;
      if (enable_gpu && sift)
      {
        float homography[9];
        int numMatches;
        FindHomography(frame_sift_data, homography, &numMatches, 1000, 0.00f, 0.95f, 5.0);
  //      int numFit = ImproveHomography(frame_sift_data, homography, 5, 0.00f, 0.80f, 3.0);
        H = cv::Mat(3, 3, CV_32F, homography);
  //      std::cout << "Number of matching features: " << numFit << " " << numMatches << " " << 100.0f*numFit/std::min(siftData1.numPts, siftData2.numPts) << "% " << initBlur << " " << thresh << std::endl;
        std::cout << "Number of original features: " <<  object_sift_data.numPts << " " << frame_sift_data.numPts << std::endl;
        std::cout << "Number of matching features: " << numMatches << std::endl;
      }
      else
      {
      for (size_t j = 0; j < good_matches.size(); j++)
      {
        obj.push_back(object_keypoints[i][good_matches[i][j].queryIdx].pt);
        env.push_back(frame_keypoints[good_matches[i][j].trainIdx].pt);
      }

      H.push_back(findHomography(obj, env, CV_RANSAC));
      }

//      std::cout << "Homography " << H[i] << std::endl;

      std::vector<Point2f> obj_corners(4);
      obj_corners[0] = cvPoint(0, 0);
      obj_corners[1] = cvPoint(object_bw_img.at(i).cols, 0);
      obj_corners[2] = cvPoint(object_bw_img.at(i).cols, object_bw_img.at(i).rows);
      obj_corners[3] = cvPoint(0, object_bw_img.at(i).rows);
      std::vector<Point2f> scene_corners(4);

      try
      {
        perspectiveTransform(obj_corners, scene_corners, H.at(i));
      }
      catch (const cv::Exception& e)
      {
        std::cout << "Perspective transform failed because: " << e.what() << std::endl;
      }
      catch (...)
      {
        std::cout << "Perspective transform failed" << std::endl;
      }


//    std::cout << "Homography: " << (int)(double(clock() - clock_homography) / clocks_per_ms) << " ms" << std::endl;

      line(img_matches, scene_corners[0] + Point2f(object_bw_img.at(i).cols, 0), scene_corners[1] + Point2f(object_bw_img.at(i).cols, 0),
           Scalar(0, 255, 0), 4);
      line(img_matches, scene_corners[1] + Point2f(object_bw_img.at(i).cols, 0), scene_corners[2] + Point2f(object_bw_img.at(i).cols, 0),
           Scalar(0, 255, 0), 4);
      line(img_matches, scene_corners[2] + Point2f(object_bw_img.at(i).cols, 0), scene_corners[3] + Point2f(object_bw_img.at(i).cols, 0),
           Scalar(0, 255, 0), 4);
      line(img_matches, scene_corners[3] + Point2f(object_bw_img.at(i).cols, 0), scene_corners[0] + Point2f(object_bw_img.at(i).cols, 0),
           Scalar(0, 255, 0), 4);
    }

    clock_total = clock_now;
    clock_now = clock();
    int elapsed_time = (int)(double(clock_now - clock_total) / clocks_per_ms);
    char elapsed_string[20];
    sprintf(elapsed_string, "%d ms", elapsed_time);
    int baseline=0;
    Size textSize = getTextSize("999 ms", FONT_HERSHEY_PLAIN,
                                2, 2, &baseline);
    Point text_origin(img_matches.cols - textSize.width,
                  textSize.height + 10);
    putText(img_matches, elapsed_string, text_origin, FONT_HERSHEY_PLAIN, 2,
            Scalar::all(255), 2, 8);
    imshow("Matches", img_matches);

#ifdef video
    if (waitKey(30) >= 0)
      break;
  }
#else
  waitKey(0);
#endif
  if (enable_gpu && sift)
  {
  FreeSiftData(frame_sift_data);
  FreeSiftData(object_sift_data);
  }
  return 0;
}

