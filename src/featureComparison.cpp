//============================================================================
// Name        : Feature Comparison
// Author      :
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#define video
int feature_type = 1; // 0 SIFT, 1 SURF, 2 ORB
bool enable_gpu = 1;

#include <stdio.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "../CudaSift/cudaImage.h"
#include "../CudaSift/cudaSift.h"
//#include "../CudaSift/geomFuncs.cpp"

#include "opencv2/cudaobjdetect.hpp"
#include "opencv2/cudafeatures2d.hpp"
//#include "opencv2/gpu/gpu.hpp"

using namespace cv;

// Path to folder with images of the objects to be recognized.
String folder_path = "/home/teddy/objects";
String video_path = "/home/teddy/test.mkv";

int camera_device = 0;

int numOctaves = 5; /* Number of octaves in Gaussian pyramid */
float initBlur = 1.0f; /* Amount of initial Gaussian blurring in standard deviations */
float thresh = 3.5f; /* Threshold on difference of Gaussians for feature pruning */
float minScale = 0.0f; /* Minimum acceptable scale to remove fine-scale features */
bool upScale = true; /* Whether to upscale image before extraction */

int MAX_FOCUS = 250;
int focus = 0;
int match_thres = 70;

void settingsWindow()
{
  namedWindow("Settings", 0);

  createTrackbar("Focus", "Settings", &focus, MAX_FOCUS);
  createTrackbar("Match thres", "Settings", &match_thres, 100);
}

bool intersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2)
{
    Point2f x = o2 - o1;
    Point2f d1 = p1 - o1;
    Point2f d2 = p2 - o2;

    float cross = d1.x*d2.y - d1.y*d2.x;
    if (abs(cross) < /*EPS*/1e-8)
        return false;

    return true;
}

int main(int, char**)
{
  // Open the default camera
  VideoCapture cap(camera_device);
//  VideoCapture cap(video_path);
  // Check if camera works
  if (!cap.isOpened())
    return -1;

  settingsWindow();

  namedWindow("Matches", CV_GUI_NORMAL);
  std::cout << "feature_type = " << feature_type << std::endl;
  std::cout << "Image resolution: " << cap.get(CV_CAP_PROP_FRAME_WIDTH) << "x" << cap.get(CV_CAP_PROP_FRAME_HEIGHT) << std::endl;

  // Load images of objects to search for
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

  // Create array with a color for each object

  std::vector<Scalar> colours;
  for (size_t i=0; i < imported_objects; i++)
  {
    Mat color(5,5, CV_8UC3, Scalar((int)(i*180.0/imported_objects), 255, 255));
    cvtColor(color, color, COLOR_HSV2BGR);
    std::cout << "Angle " << (int)(i*180.0/(float)imported_objects) << " color " << color.at<Vec3b>(Point(0,0)) <<std::endl;
    colours.push_back(color.at<Vec3b>(Point(0,0)));
  }

  // Convert images to greyscale
  std::vector<Mat> object_bw_img;
  for (size_t i=0; i < imported_objects; i++)
    {
      object_bw_img.push_back(Mat(object_img.at(0).cols, object_img.at(0).rows, CV_8UC3));
      cvtColor(object_img.at(i), object_bw_img.back(), COLOR_BGR2GRAY);
    }

  cv::Mat object_sift_img;

  CudaImage object_cuda_sift_img;
  SiftData frame_sift_data, object_sift_data;
  cv::Mat frame_sift_img;
  CudaImage frame_cuda_sift_img;

  VideoWriter outputVideo;
  outputVideo.open("/home/teddy/output.avi",
                   static_cast<int>(cap.get(CV_CAP_PROP_FOURCC)),
                   cap.get(CV_CAP_PROP_FPS),
                   Size((int) cap.get(CV_CAP_PROP_FRAME_WIDTH) + object_bw_img.at(0).cols,
                   (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT)), true);

  if (!outputVideo.isOpened())
  {
      std::cout  << "Could not open the output video " << std::endl;
      return -1;
  }

//  int ex = static_cast<int>(cap.get(CAP_PROP_FOURCC));     // Get Codec Type- Int form
//  // Transform from int to char via Bitwise operators
//  char EXT[] = {(char)(ex & 0XFF) , (char)((ex & 0XFF00) >> 8),(char)((ex & 0XFF0000) >> 16),(char)((ex & 0XFF000000) >> 24), 0};
//  Size S = Size((int) cap.get(CAP_PROP_FRAME_WIDTH),    // Acquire input size
//                (int) cap.get(CAP_PROP_FRAME_HEIGHT));
//  VideoWriter outputVideo;                                        // Open the output
//  outputVideo.open("/home/teddy/output.avi", ex, cap.get(CAP_PROP_FPS), S, true);
//  if (!outputVideo.isOpened())
//  {
//      std::cout  << "Could not open the output video " << std::endl;
//      return -1;
//  }


  if (enable_gpu && feature_type == 0)
  {
  InitCuda(camera_device);

  object_bw_img.at(0).convertTo(object_sift_img, CV_32FC1);
  object_cuda_sift_img.Allocate(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT),
                         cap.get(CV_CAP_PROP_FRAME_WIDTH), false, NULL, (float*)object_sift_img.data);

  // Get a frame to initialize cudaSift
//  cv::Mat frame_sift_img(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT), CV_32FC1);

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


  frame_cuda_sift_img.Allocate(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT),
                         cap.get(CV_CAP_PROP_FRAME_WIDTH), false, NULL, (float*)frame_sift_img.data);

  // Reserve memory on GPU as allocating is slow
  InitSiftData(frame_sift_data, 2500000, true, true);
  InitSiftData(object_sift_data, 2500000, true, true);
  }


  cv::Ptr<Feature2D> feature_extractor;

  int min_hessian = 400;
  std::vector<KeyPoint> frame_keypoints;
  std::vector<std::vector<KeyPoint> > object_keypoints;
  Mat frame_descriptors;
  std::vector<Mat> object_descriptors;

  std::vector<cuda::GpuMat> object_gpu_img;
  for (size_t i=0; i < object_bw_img.size(); i++)
  {
    object_gpu_img.push_back(cuda::GpuMat(object_bw_img.at(i)));
    object_cuda_sift_img.Download();
  }

  cuda::GpuMat frame_gpu_keypoints, frame_gpu_descriptors;
  std::vector<cuda::GpuMat> object_gpu_keypoints, object_gpu_descriptors(CV_32F);

  cuda::SURF_CUDA* surf;

  cv::Ptr<cv::cuda::ORB> gpu_orb = cv::cuda::ORB::create();

  if (feature_type == 0)
  {
    if (enable_gpu)
    {
      object_bw_img.at(0).convertTo(object_sift_img, CV_32FC1);
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
           object_keypoints.push_back(std::vector<KeyPoint>());
           object_descriptors.push_back(Mat());
          feature_extractor->detect(object_bw_img.at(i), object_keypoints.at(i));
          feature_extractor->compute(object_bw_img.at(i), object_keypoints.at(i), object_descriptors.at(i));
        }

    }

  }
  else if (feature_type == 1)// SURF
  {
    if (enable_gpu)
    {
      surf = new cuda::SURF_CUDA(min_hessian);
      for (size_t i=0; i < object_bw_img.size(); i++)
      {
        object_gpu_keypoints.push_back(cuda::GpuMat());
        object_gpu_descriptors.push_back(cuda::GpuMat(CV_32F));
        object_keypoints.push_back(std::vector<KeyPoint>());
        try
        {
          surf->operator()( object_gpu_img.at(i), cuda::GpuMat(), object_gpu_keypoints.at(i), object_gpu_descriptors.at(i));
          surf->downloadKeypoints(object_gpu_keypoints.at(i), object_keypoints.at(i));
          std::cout << "Found " << object_keypoints.at(i).size() << " keypoints in object " << i << "." << std::endl;
        }
        catch (const cv::Exception& e)
        {
          std::cout << "Couldn't extract keypoints from object " << i << "." << std::endl;
          object_bw_img.erase(object_bw_img.begin() + i);
          object_gpu_keypoints.erase(object_gpu_keypoints.begin() + i);
          object_gpu_descriptors.erase(object_gpu_descriptors.begin() + i);
          object_keypoints.erase(object_keypoints.begin() + i);
          object_gpu_img.erase(object_gpu_img.begin() + i);
        }
      }
    }
    else
    {
    feature_extractor = xfeatures2d::SURF::create(min_hessian);
    for (size_t i=0; i < object_bw_img.size(); i++)
    {
      object_keypoints.push_back(std::vector<KeyPoint>());
      object_descriptors.push_back(Mat());
      feature_extractor->detect(object_bw_img.at(i), object_keypoints.at(i));
      feature_extractor->compute(object_bw_img.at(i), object_keypoints.at(i), object_descriptors.at(i));
    }
    }
  }
  else // ORB
  {
    if (enable_gpu)
    {
      for (size_t i=0; i < object_bw_img.size(); i++)
      {
        object_gpu_keypoints.push_back(cuda::GpuMat());
        object_gpu_descriptors.push_back(cuda::GpuMat(CV_32F));
        object_keypoints.push_back(std::vector<KeyPoint>());
        try
        {
          gpu_orb->detectAndCompute(object_gpu_img.at(i), noArray(), object_keypoints.at(i), object_gpu_descriptors.at(i));
          std::cout << "Found " << object_keypoints.at(i).size() << " keypoints in object " << i << "." << std::endl;
        }
        catch (const cv::Exception& e)
        {
          std::cout << "Couldn't extract keypoints from object " << i << "." << std::endl;
          object_bw_img.erase(object_bw_img.begin() + i);
          object_gpu_keypoints.erase(object_gpu_keypoints.begin() + i);
          object_gpu_descriptors.erase(object_gpu_descriptors.begin() + i);
          object_keypoints.erase(object_keypoints.begin() + i);
          object_gpu_img.erase(object_gpu_img.begin() + i);
        }
      }
    }
  }


  clock_t clock_now = clock();
  clock_t clock_total, clock_extraction, clock_matching, clock_homography;
  int clocks_per_ms = CLOCKS_PER_SEC / 1000;

#ifdef video
  while (cvWaitKey(1) != '\33')
  {
#endif

    Mat frame_img, frame_bw_img, output_frame;

#ifdef video
    cap.set(CV_CAP_PROP_FOCUS, focus);
    // Get a frame from camera
    cap >> frame_img;
    if (frame_img.empty())
      continue;
#else
    frame_img = imread("/home/teddy/env.jpg", 1);
#endif

    std::cout << "heyy" << std::endl;
    cvtColor(frame_img, frame_bw_img, COLOR_BGR2GRAY);

    clock_extraction = clock();

    if (feature_type == 0)
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
    else if (feature_type == 1)// SURF
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
    else // ORB
    {
      if (enable_gpu)
      {
        cuda::GpuMat frame_gpu_img(frame_bw_img);
        gpu_orb->detectAndCompute(frame_gpu_img, noArray(), frame_keypoints, frame_gpu_descriptors);
      }
    }

    std::cout << "Feature extraction: " << (int)(double(clock() - clock_extraction) / clocks_per_ms) << " ms" << std::endl;

    clock_matching = clock();

    std::vector<std::vector<DMatch> > good_matches;
//    std::vector<DMatch> matches;
    std::vector<std::vector<std::vector< DMatch> > > matches;
    if (enable_gpu)
    {
      if (feature_type == 0)
      {
        MatchSiftData(frame_sift_data, object_sift_data);
      }
      else
      {
        Ptr< cuda::DescriptorMatcher > matcher;
        if (feature_type == 1)
        {
          matcher = cuda::DescriptorMatcher::createBFMatcher();
        }
        else
        {
          matcher = cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
        }
//        std::vector< std::vector< DMatch> > matches;
        std::cout << "Matching..." << std::endl;
        for (size_t i=0; i < object_bw_img.size(); i++)
        {
          matches.push_back(std::vector<std::vector<DMatch> >());
          matcher->knnMatch(object_gpu_descriptors.at(i), frame_gpu_descriptors, matches.back(), 2);
          std::cout << "Found " << matches.at(i).size() << " matches for object " << i << std::endl;
        }
        if (feature_type == 1)
        {
          surf->downloadKeypoints(frame_gpu_keypoints, frame_keypoints);
          std::cout << "Found " << frame_keypoints.size() << " keypoints in frame."<< std::endl;
          std::cout << "feature_type = " << feature_type << std::endl;
        }
      }
    }
    else
    {
      FlannBasedMatcher matcher;

      for (size_t i=0; i < object_bw_img.size(); i++)
      {
        matches.push_back(std::vector<std::vector<DMatch> >());
        matcher.knnMatch(object_descriptors.at(i), frame_descriptors, matches.at(i), 2);
      }

    }



    if (!(enable_gpu && feature_type == 0))
    {
      for (size_t i=0; i < object_bw_img.size(); i++)
      {
        good_matches.push_back(std::vector<DMatch>());
//
        for(std::vector<std::vector<cv::DMatch> >::const_iterator it = matches[i].begin(); it != matches[i].end(); ++it) {
            if(it->size() > 1 && (*it)[0].distance/(*it)[1].distance < match_thres/100.0) {
              good_matches[i].push_back((*it)[0]);
            }
//            else
//            {
//              std::cout << "Distance 1: " << (*it)[0].distance << " Distance 2: "<< (*it)[1].distance << " Ratio: "<< (*it)[0].distance/(*it)[1].distance << std::endl;
//            }
        }

//        double max_dist = 0; double min_dist = 100;
//        for( int j = 0; j < object_bw_img.size(); j++)
//        {
//          double dist = matches[i][j][0].distance;
//            if( dist < min_dist ) min_dist = dist;
//            if( dist > max_dist ) max_dist = dist;
//        }
//
//        for( int j = 0; j < object_bw_img.size(); j++ )
//
//        { if( matches[i][j][0].distance < (max_dist/1.6) )
//            { good_matches[i].push_back( matches[i][j][0]); }
//        }


//        for (size_t k = 0; k < std::min(object_keypoints.at(i).size()-1, matches.at(i).size()); k++)
//        {
//                if ( (matches[i][k][0].distance < 0.6*(matches[i][k][1].distance)) &&
//                                ((int)matches[i][k].size() <= 2 && (int)matches[i][k].size()>0) )
//                {
//                        // take the first result only if its distance is smaller than 0.6*second_best_dist
//                        // that means this descriptor is ignored if the second distance is bigger or similar
//                        good_matches[i].push_back(matches[i][k][0]);
//                }
//                else
//                {
//                  std::cout << "First and second match distance: " << matches[i][k][0].distance << " "<< matches[i][k][1].distance << std::endl;
//                  std::cout << "Number of match candidates "<< matches[i][k].size() << std::endl;
//                }
//        }
        std::cout << "Found " << good_matches.at(i).size() << " good matches for object " << i << std::endl;

      }
    }



    std::cout << "Feature matching: " << (int)(double(clock() - clock_matching) / clocks_per_ms) << " ms" << std::endl;

    Mat img_matches = Mat::zeros(frame_img.size(), CV_8UC3);

    if (enable_gpu && feature_type == 0)
    {
          frame_img.copyTo(img_matches);
        }
    else
        {
          try {
      //    drawKeypoints(frame_img, frame_keypoints, output_frame, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
          drawMatches(object_img.at(0), object_keypoints.at(0), frame_img, frame_keypoints, good_matches.at(0), img_matches);
          }
          catch(...)
          {
            std::cout << "Error while drawing" << std::endl;
          }
        }

    clock_homography = clock();
    std::vector<Mat> H(object_bw_img.size(), Mat());

    for (size_t i=0; i < object_bw_img.size(); i++)
    {
      if (!(enable_gpu && feature_type == 0))
          {
            if (good_matches.at(i).size() < 1)
            {
              std::cout << "No matches found for object " << i << "." << std::endl;
              continue;
            }
          }
      std::vector<Point2f> obj;
      std::vector<Point2f> env;
      if (enable_gpu && feature_type == 0)
      {
        float homography[9];
        int numMatches;
        FindHomography(frame_sift_data, homography, &numMatches, 1000, 0.00f, 0.95f, 5.0);
        std::cout << "Did homography" << std::endl;
  //      int numFit = ImproveHomography(frame_sift_data, homography, 5, 0.00f, 0.80f, 3.0);
//        std::vector<float> tmp_vec (homography, homography + sizeof(homography) / sizeof(homography[0]) );
//        H.at(0) = cv::Mat(tmp_vec).reshape(3, 3);
        H.at(0) = cv::Mat(3, 3, CV_32FC1, homography);
        std::cout << "Number of original features: " <<  object_sift_data.numPts << " " << frame_sift_data.numPts << std::endl;
        std::cout << "Number of matching features: " << numMatches << std::endl;
//        for (int ij=0; ij < 9; ij++)
//        {
//        std::cout << "Homography "<< homography[ij] << std::endl;
//        }
      }
      else
      {
      for (size_t j = 0; j < good_matches.at(i).size(); j++)
      {
        obj.push_back(object_keypoints[i][good_matches[i][j].queryIdx].pt);
        env.push_back(frame_keypoints[good_matches[i][j].trainIdx].pt);
      }

      if (good_matches.at(i).size() > 0)
      {
        H.at(i) = findHomography(obj, env, CV_RANSAC);
      }
      }

      if (H.at(i).cols < 3)
      {
        continue;
      }


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



      if (isContourConvex(scene_corners))
      {
      if (enable_gpu && feature_type == 0)
          {
      line(img_matches, scene_corners[0], scene_corners[1],
           colours.at(i), 4);
      line(img_matches, scene_corners[1], scene_corners[2] + Point2f(object_bw_img.at(0).cols, 0),
           colours.at(i), 4);
      line(img_matches, scene_corners[2], scene_corners[3] + Point2f(object_bw_img.at(0).cols, 0),
           colours.at(i), 4);
      line(img_matches, scene_corners[3], scene_corners[0] + Point2f(object_bw_img.at(0).cols, 0),
           colours.at(i), 4);
          }
      else
      {
  line(img_matches, scene_corners[0] + Point2f(object_bw_img.at(0).cols, 0), scene_corners[1] + Point2f(object_bw_img.at(0).cols, 0),
       colours.at(i), 4);
  line(img_matches, scene_corners[1] + Point2f(object_bw_img.at(0).cols, 0), scene_corners[2] + Point2f(object_bw_img.at(0).cols, 0),
       colours.at(i), 4);
  line(img_matches, scene_corners[2] + Point2f(object_bw_img.at(0).cols, 0), scene_corners[3] + Point2f(object_bw_img.at(0).cols, 0),
       colours.at(i), 4);
  line(img_matches, scene_corners[3] + Point2f(object_bw_img.at(0).cols, 0), scene_corners[0] + Point2f(object_bw_img.at(0).cols, 0),
       colours.at(i), 4);
      }
      }
    }
    std::cout << "Homography: " << (int)(double(clock() - clock_homography) / clocks_per_ms) << " ms" << std::endl;

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
//    putText(img_matches, elapsed_string, text_origin, FONT_HERSHEY_PLAIN, 2,
//            Scalar::all(255), 2, 8);
    imshow("Matches", img_matches);
    outputVideo << img_matches;


#ifdef video
  }
#else
  waitKey(0);
#endif
  if (enable_gpu && feature_type == 0)
  {
  FreeSiftData(frame_sift_data);
  FreeSiftData(object_sift_data);
  }
  std::cout << "Terminating" << std::endl;

  return 0;
}
