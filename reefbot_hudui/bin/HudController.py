#!/usr/bin/python
'''Controls the hud UI and resets the video stream as necessary.'''
import roslib; roslib.load_manifest('reefbot_hudui')
import rospy
import subprocess
import os.path
import sys
import re
import os
import signal
import multiprocessing

def WatchVideoThread(proc):
  errorWatcher = VideoErrorWatcher(proc)
  for line in proc.stdout:
    if not errorWatcher.CheckLogMessage(line):
      break

class VideoErrorWatcher:
  '''Class that replaces stderr so that we can count the video errors are reset if necessary.

  Return true if the output should still be processed'''
  def __init__(self, proc):
    self.maxErrorMsgs = rospy.get_param('~max_video_errors', 20)
    self.curCount = 0
    self.errorRe = re.compile(r'.*number of reference frames exceeds max.*')
    self.proc = proc

  def CheckLogMessage(self, msg):
    if self.errorRe.match(msg):
      self.curCount = self.curCount + 1
      if self.curCount > self.maxErrorMsgs:
        rospy.logerr("Killing the hud process. Please wait for it to restart")
        os.kill(proc.pid, signal.SIGTERM)
        self.curCount = 0
        return False
    rospy.loginfo(msg)
    return True

if __name__ == '__main__':
  rospy.init_node('HudController')

  scriptDir = os.path.dirname(sys.argv[0])


  while not rospy.is_shutdown():
    rospy.logwarn("Starting the HUD UI process")
    proc = subprocess.Popen([os.path.join(scriptDir, 'hudui.py')],
                            stderr=subprocess.STDOUT,
                            stdout=subprocess.PIPE)

    try:
      watcherThread = multiprocessing.Process(target=WatchVideoThread,
                                              args=tuple([proc]))
      watcherThread.start()
      proc.wait()
      watcherThread.terminate()
      

    finally:
      if proc.poll() is None:
        rospy.logwarn("Killing the hudui.py process")
        proc.kill()
        proc.wait()
