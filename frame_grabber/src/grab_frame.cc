/* frame_grabber 
 *  Grabs frames from the video stream and writes to the disk using 
 *  libVLC and SDL and modelled using the ROS framework.
 * Author: Srinivasan Vijayarangan (srinivasan@cmu.edu)
 * Date: 21 Feb 2011
 */

#include <iostream>
#include <sstream>
#include <string>
#include <vlc/vlc.h>
#include <SDL/SDL.h>
#include <SDL/SDL_mutex.h>
#include <ros/ros.h>
#include "std_msgs/String.h"
#include "frame_grabber/Rate.h"
#include "frame_grabber/Snapshot.h"
#include "frame_grabber/NumSnapshot.h"
#include "frame_grabber/Exit.h"

using namespace std;

int rate_hz = 7;
bool snapshot = true;
bool done = false;
int snapshot_limit = 20;

///ROS SERVICES

bool rate(frame_grabber::Rate::Request &req, frame_grabber::Rate::Response &res) {
  if(req.rate < 1) {
    stringstream ss;
    ss << "Cannot set the rate below 1Hz. Using the previous value " << rate_hz << "Hz";
    ROS_INFO("%s", ss.str().c_str());
    res.status = ss.str();
    return true;
  }
    rate_hz=req.rate;
    ROS_INFO("Setting the rate to %d", (int)req.rate);
    res.status = "Rate set successfully"; //Return success
    return true;
}

bool set_snapshot(frame_grabber::Snapshot::Request &req, frame_grabber::Snapshot::Response &res) {
  if(req.snapshot >= 1) {
    snapshot = true;
    ROS_INFO("Snapshot enabled.");
    res.status = "Snapshot enabled"; //Success
  } else {
    snapshot = false;
    ROS_INFO("Snapshot disabled");
    res.status = "Snapshot disabled"; //Success
  }
  return true;
}

bool do_exit(frame_grabber::Exit::Request &req, frame_grabber::Exit::Response &res) {
  done = true;
  ROS_INFO("Exit status code received.");
  res.status = "Exit status code set successfully."; //Success
  return true;
}

bool set_num_snapshot(frame_grabber::NumSnapshot::Request &req, frame_grabber::NumSnapshot::Response &res) {
  snapshot_limit = req.num;
  stringstream ss;
  ss << "Maximum number of snapshots set to " << req.num;
  ROS_INFO("%s", ss.str().c_str());
  res.status = ss.str();
  return true;
}

struct ctx {
    SDL_Surface *surf;
    SDL_mutex *mutex;
};

static void *lock(void *data, void **p_pixels) {
    struct ctx *ctx = (struct ctx*)data;
    SDL_LockMutex(ctx->mutex);
    SDL_LockSurface(ctx->surf);
    *p_pixels = ctx->surf->pixels;
    return NULL;
}

static void unlock(void *data, void *id, void *const *p_pixels) {
    struct ctx *ctx = (struct ctx*)data;
    SDL_UnlockSurface(ctx->surf);
    SDL_UnlockMutex(ctx->mutex);
    assert(id == NULL);
}

static void display(void *data, void *id) {
    (void)data;
    assert(id == NULL);
}

int main(int argc, char *argv[]) {

  //Initialize ROS
  ros::init(argc, argv, "frame_grabber");
  ros::NodeHandle nh;

  string media_path;
  if(!nh.getParam("frame_grabber/media_path", media_path)) {
    ROS_INFO("Set the frame_grabber/media_path parameter before starting the node. Exiting now...");
    ros::shutdown();
    return -1;
  }

  string snapshot_dir;
  if(nh.getParam("frame_grabber/snapshot_dir", snapshot_dir)) {
    ROS_INFO("Setting the snapshot directory to %s", snapshot_dir.c_str());
  } else {
    snapshot_dir = "snapshots/";
    ROS_INFO("Setting the snapshot directory(default) %s", snapshot_dir.c_str());
  }

  //Initialize services
  ros::Publisher filename_pub = nh.advertise<std_msgs::String>("frame_grabber/current_file_name", 100);
  ros::ServiceServer delay_service = nh.advertiseService("frame_grabber/set_rate", rate);
  ROS_INFO("Advertised set rate service.");
  ros::ServiceServer snapshot_service = nh.advertiseService("frame_grabber/enable_snapshot", set_snapshot);
  ROS_INFO("Advertised enable snapshot.");
  ros::ServiceServer num_snapshot_service = nh.advertiseService("frame_grabber/num_snapshot", set_num_snapshot);
  ROS_INFO("Advertised num_snapshot service.");
  ros::ServiceServer exit_service = nh.advertiseService("frame_grabber/exit", do_exit);
  ROS_INFO("Advertised exit service.");

  //Initialize SDL
  struct ctx ctx;
  if(SDL_Init(SDL_INIT_NOPARACHUTE | SDL_INIT_VIDEO) == -1) {
      //NOPARACHUTE ensures that no signal handlers are installed
      ROS_INFO("Could not initialize SDL. Exiting.");
      done = true;
  }

  ctx.surf = SDL_CreateRGBSurface(SDL_SWSURFACE, 1024, 768, 16, 0x001f, 0x07e0, 0xf800, 0);
  ctx.mutex = SDL_CreateMutex();

  //Initialize libVLC
  libvlc_instance_t *libvlc;
  libvlc_media_t *m;
  libvlc_media_player_t *mp;
  
  char const *vlc_argv[] = {
     "--no-audio", /* skip any audio track */
      "--no-xlib", /* tell VLC to not use Xlib */
     //"-vvv",
  };
  int vlc_argc = sizeof(vlc_argv)/sizeof(*vlc_argv);

  libvlc = libvlc_new(vlc_argc, vlc_argv);
  m = libvlc_media_new_path(libvlc, media_path.c_str());
  mp = libvlc_media_player_new_from_media(m);
  libvlc_media_release(m);
  
  libvlc_video_set_callbacks(mp, lock, unlock,  display,  &ctx);
  libvlc_video_set_format(mp, "RV16", 1024, 768, 512);
  
  if(mp == NULL) {
      ROS_INFO("Error creating media player object. Exiting.");
      done = true;
  }

  if(libvlc_media_player_play(mp) == -1) {
      ROS_INFO("Error playing media. Exiting.");
      done=true;
  } else {
      ROS_INFO("media player play successful");
  }
 
  sleep(2);

  int snapshot_num = 0;

  while(!done && ros::ok()) {
      SDL_LockMutex(ctx.mutex);
      SDL_UnlockMutex(ctx.mutex);
      
      if(snapshot) {
          stringstream ss;
          ss << snapshot_dir << "snapshot" << snapshot_num++ << ".png";
          std_msgs::String msg;
          msg.data = ss.str();
          filename_pub.publish(msg);
          ROS_INFO("Writing snapshot file :%s",ss.str().c_str()); 
          libvlc_video_take_snapshot(mp, 0, ss.str().c_str(), 1024, 768);
          if(snapshot_num >= snapshot_limit) { 
              snapshot_num = 0; //Keep rotating the names
          }
      }
      ros::spinOnce();
      ros::Rate loop_rate(rate_hz);
      loop_rate.sleep();
  }
  
  //Stop stream and clean up libVLC
  libvlc_media_player_stop(mp);
  libvlc_media_player_release(mp);
  libvlc_release(libvlc);
  
  //Clean SDL
  SDL_DestroyMutex(ctx.mutex);
  SDL_FreeSurface(ctx.surf);
  SDL_Quit();
  
  //Clean up ROS
  ros::shutdown();
  
  return 0;
}
