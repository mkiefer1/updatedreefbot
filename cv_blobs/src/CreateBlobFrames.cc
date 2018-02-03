// Copyright 2010 ReefBot
// Author: Mark Desnoyer (mdesnoyer@gmail.com)
//
// A program that will generate jpegs for each blob of an image
//
// Usage: CreateBlobFrames [options]

#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/shared_ptr.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <gflags/gflags.h>
#include <iostream>

#include "base/BasicTypes.h"
#include "cv_blobs/BlobResult-Inline.h"
#include "cv_blobs/Blob.h"

using namespace boost;
namespace fs = boost::filesystem;
using namespace std;
using namespace cv;
using namespace cv_blobs;

DEFINE_string(input, "", "Blob file to process");
DEFINE_string(output_dir, ".", "Directory to output the jpegs to");

DEFINE_double(frame_width, 500,
             "Width of each resulting JPEG");
DEFINE_double(frame_height, 500,
             "Height of each resulting JPEG");
DEFINE_bool(use_static_size, true,
            "Size the frame to be a fixed size in pixels defined by "
            "frame_width and frame_height. If false, frame_width and "
            "frame_height are treated as fractions of the blob size, "
            "so you want a value > 1");

DEFINE_bool(draw_rect, true,
            "Should we draw a red rectangle around the blob?");

int main(int argc, char** argv) {
  // Parse the input
  google::ParseCommandLineFlags(&argc, &argv, true);

  fs::path outputPath(FLAGS_output_dir);
  fs::path inputPath(FLAGS_input);

  // First open up the stream to the blob definition
  fs::ifstream blobDescStream(inputPath, ios_base::in);
  
  string imgFilename; // Filename of the image with the blobs in it
  BlobResultSerializer<uint8> blobReader;
  shared_ptr<BlobResult<uint8> > blobs =
    blobReader.Deserialize(blobDescStream, &imgFilename, 
                           inputPath.parent_path().string());

  // Now open up the full image frame
  Mat fullImage = imread(imgFilename);

  // Finally, create the output frames one at a time
  for (int i = 0; i < blobs->nBlobs(); ++i) {
    const Blob& curBlob = blobs->GetBlob(i);
    const Rect& box = curBlob.GetBoundingBox();

    // Figure out the total frame to extract
    Point2f frameSize(FLAGS_frame_width / 2, FLAGS_frame_height / 2);
    if (!FLAGS_use_static_size) {
      frameSize = Point2f(box.width * FLAGS_frame_width / 2,
                          box.height * FLAGS_frame_height / 2);
    }
    Point2f minCorner = box.tl();
    minCorner -= frameSize;
    if (minCorner.x < 0) {
      minCorner.x = 0;
    }
    if (minCorner.y < 0) {
      minCorner.y = 0;
    }
    Point2f maxCorner = box.br();
    maxCorner += frameSize;
    if (maxCorner.x >= fullImage.cols) {
      maxCorner.x = fullImage.cols -1;
    }
    if (maxCorner.y >= fullImage.rows) {
      maxCorner.y = fullImage.rows -1;
    }

    Rect frameBox = Rect(minCorner, maxCorner);
    Mat curFrame = Mat(fullImage, frameBox).clone();

    // Draw a red rectagle around the fish
    if (FLAGS_draw_rect) {
      Point2f fishBottom = box.tl();
      fishBottom -= minCorner;
      fishBottom -= Point2f(1,1);
      if (fishBottom.x < 0) {
        fishBottom.x = 0;
      }
      if (fishBottom.y < 0) {
        fishBottom.y = 0;
      }
      Point2f fishTop = box.br();
      fishTop -= minCorner;
      
      IplImage iplFrame = IplImage(curFrame);
      cvRectangle(&iplFrame,
                  fishBottom,
                  fishTop,
                  CV_RGB(255,0,0),
                  3,
                  8,
                  0);
    }

    // Finally write out the jpeg
    const string outFilename = inputPath.filename() + '.' +
      lexical_cast<string>(i) + ".jpg";
    imwrite((outputPath / outFilename).file_string(), curFrame);
  }
}
