#! /usr/bin/python

import roslib; roslib.load_manifest('reefbot_hudui')
import rospy
import sys
from PyQt4 import Qt, QtGui, QtCore, QtSvg
import vlc
import threading
import thread
from Xlib import display
from reefbot_msgs.msg import RobotStatus
from reefbot_msgs.msg import VideoStream
import os.path
import re
import signal

class VideoWidget(Qt.QWidget):
  def __init__(self, parent=None):
    Qt.QWidget.__init__(self, parent)
    self.inputURL = rospy.get_param('video_url',
      "/home/mdesnoyer/data/reefbot/test_videos/test_h264stream.avi")
    self.rtpIp = rospy.get_param('video_bcast_ip', '239.255.15.42')
    self.rtpPort = rospy.get_param('video_bcast_port', '5004')

    self.lock = threading.Semaphore(1)
                                      
    self.vlcInst=None
    self.log = None
    self.player = None
    self.SetNewInputPath(self.inputURL)

  def RestartStream(self):
    self.lock.acquire()
    if self.player is not None:
      self.player.stop()
      self.player.release()
    if self.vlcInst is not None:
      self.vlcInst.release()

    self.vlcInst = vlc.Instance()
    self.log = self.vlcInst.log_open()

    self.player = self.vlcInst.media_player_new()
    self.player.set_xwindow(int(self.winId()))

    self.player.set_mrl(self.inputURL,
                        'rtsp-caching=400', # 200
                        'realrtsp-caching=400',
                        'file-caching=400',
                        'clock-synchro=0',
                        'clock-jitter=0',
                        'rtp-max-misorder=16', # packets in the past
                        #'rtp-max-dropout=16', #packets in the future
                        'sout-mux-caching=800',
                        'sout=#duplicate{dst=display, dst=rtp{dst=%s, port=%s, mux=ts}}' % (self.rtpIp, self.rtpPort))
    
    self.player.play()
    self.lock.release()

  def SetNewInputPath(self, path):
    '''Sets a new input path and restarts the stream.'''
    self.inputURL = path
    self.RestartStream()
    
class DepthGauge(QtGui.QWidget):
  def __init__(self, curDir, parent=None):
    QtGui.QWidget.__init__(self, parent)
    self.baseIm = QtGui.QImage(os.path.join(curDir,
                                             "../images/DepthGauge.png"))
    self.baseIm = self.baseIm.scaledToHeight(800,
                                             Qt.Qt.SmoothTransformation)
    self.setGeometry(0, 0, self.baseIm.width(), self.baseIm.height())
    
    self.setMask(QtGui.QBitmap.fromImage(self.baseIm.createAlphaMask()))
    self.setAttribute(Qt.Qt.WA_NoSystemBackground)
    self.minDepth = rospy.get_param("~min_depth", 0.0)
    self.maxDepth = rospy.get_param("~max_depth", 9.0)

    self.minDepthLoc = 40
    self.maxDepthLoc = self.baseIm.height() - 50

    self.SetDepth(0.0)
    self.setUpdatesEnabled(True)
    self.repaint()

  def paintEvent(self, event):
    '''Displays the current depth in meters'''
    elipseLoc = (self.minDepthLoc +
                 (self.depth - self.minDepth) / (self.maxDepth - self.minDepth) *
                 (self.maxDepthLoc - self.minDepthLoc))

    painter = QtGui.QPainter(self)
    painter.drawImage(0,0, self.baseIm)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 0)))
    painter.setPen(Qt.Qt.NoPen)
    painter.drawEllipse((self.baseIm.width()-60)/2,
                        elipseLoc,
                        60, 30)

    pen = QtGui.QPen(QtGui.QColor(180, 0, 0))
    pen.setWidthF(1.5)
    painter.setPen(pen)
    painter.drawText(30, elipseLoc + 20, '%3.1fm' % self.depth)
    painter.end()

  def SetDepth(self, depth):
    self.depth = depth

class CameraTarget(Qt.QLabel):
  def __init__(self, minX, minY, maxX, maxY, size, parent=None):
    Qt.QLabel.__init__(self, parent)
    width = maxX - minX
    height = maxY - minY
    self.setMinimumSize(width, height)
    self.pixmap = QtGui.QPixmap(width, height)
    painter = QtGui.QPainter(self.pixmap)
    painter.fillRect(QtCore.QRect(0, 0, width, height),
                     QtGui.QColor(255, 255, 255))
    self.DrawCorner(painter, 3, 3, size, size)
    self.DrawCorner(painter, 3, height-3, size, -size)
    self.DrawCorner(painter, width-3, 3, -size, size)
    self.DrawCorner(painter, width-3, height-3, -size, -size)
    painter.end()

    self.setPixmap(self.pixmap)
    self.setMask(self.pixmap.createMaskFromColor(QtGui.QColor(255, 255, 255)))

  def DrawCorner(self, painter, x, y, width, height):
    pen = QtGui.QPen(QtGui.QColor(180, 0, 0))
    pen.setWidthF(6.0)
    painter.setPen(pen)
    painter.drawLine(x, y, x+width, y)
    painter.drawLine(x, y, x, y+height)

class SpeedStatus(QtGui.QWidget):
  def __init__(self, curDir, parent=None):
    QtGui.QWidget.__init__(self, parent)
    self.height = 260
    self.width = 450
    self.setGeometry(0, 0, width, height)
    self.canvas = QtGui.QPixmap(width, height)
    self.bgColor = QtGui.QColor(17, 20, 50)
    self.horizSub = QtGui.QPixmap(os.path.join(curDir,
                                               '../images/ReefBotCLEO-1.png'))
    self.horizSub = self.horizSub.scaledToWidth(150,
                                                Qt.Qt.SmoothTransformation)
    self.vertSub = QtGui.QPixmap(os.path.join(curDir,
                                              '../images/ReefBotCLEO-A.png'))
    self.vertSub = self.vertSub.scaledToWidth(120,
                                              Qt.Qt.SmoothTransformation)

    self.arrow = QtGui.QPolygon([
      QtCore.QPoint(-10, 5),
      QtCore.QPoint(0, 5),
      QtCore.QPoint(0, 10),
      QtCore.QPoint(10, 0),
      QtCore.QPoint(0, -10),
      QtCore.QPoint(0, -5),
      QtCore.QPoint(-10, -5),
      ])
    
    # Paint the static elements
    painter = QtGui.QPainter(self.canvas)
    painter.fillRect(QtCore.QRect(0, 0, self.width, self.height),
                     self.bgColor)
    painter.save()
    painter.translate(150, self.height / 2)
    painter.drawPixmap(-self.vertSub.width()/2, -self.vertSub.height()/2,
                       self.vertSub)
    painter.restore()
    painter.end()

    self.SetSpeedStatus(0, 0, 0, 0)
    self.setUpdatesEnabled(True)
    

  def SetSpeedStatus(self, leftSpeed, rightSpeed, vertSpeed, pitch):
    self.leftSpeed = leftSpeed
    self.rightSpeed = rightSpeed
    self.vertSpeed = vertSpeed
    self.pitch = pitch

  def paintEvent(self, event):
    
    mainPainter = QtGui.QPainter(self)
    mainPainter.fillRect(QtCore.QRect(0, 0, self.width, self.height),
                     self.bgColor)
    
    mapCopy = QtGui.QPixmap(self.canvas)
    painter = QtGui.QPainter(mapCopy)

    self.DrawVerticalArrows(painter, self.vertSpeed)
    self.DrawPitchedSub(painter, self.pitch)
    self.DrawDirectionArrows(painter, self.leftSpeed, self.rightSpeed)

    painter.end()

    mainPainter.drawPixmap(0, 0, mapCopy)
    mainPainter.end()
    
    self.setMask(mapCopy.createMaskFromColor(self.bgColor))

  def DrawPitchedSub(self, painter, pitch):
    painter.save()

    painter.translate(350, self.height / 2)
    painter.rotate(pitch)

    painter.drawPixmap(-self.horizSub.width()/2, -self.horizSub.height()/2,
                       self.horizSub)

    painter.restore()

  def DrawArrow(self, painter, x, y, angle):
    painter.save()
    painter.translate(x, y)
    painter.rotate(angle)
    painter.scale(2, 2)
    
    painter.setBrush(QtGui.QBrush(QtGui.QColor(180, 0, 0)))
    painter.setPen(Qt.Qt.NoPen)

    painter.drawConvexPolygon(self.arrow)

    painter.restore()

  def DrawVerticalArrows(self, painter, vertSpeed):
    if vertSpeed > 0.02:
      # We're going up
      self.DrawArrow(painter, 350, 40, -90)
    elif vertSpeed < -0.02:
      # We're going down
      self.DrawArrow(painter, 350, self.height - 40, 90)

  def DrawDirectionArrows(self, painter, leftSpeed, rightSpeed):
    if (leftSpeed > 0.02 and rightSpeed < -0.02 and
        (abs((abs(leftSpeed) - abs(rightSpeed))) < 0.02)):
      # We're spinning right
      self.DrawArrow(painter, 250, self.height / 2, 0)

    if (leftSpeed < -0.02 and rightSpeed > 0.02 and
        (abs((abs(leftSpeed) - abs(rightSpeed))) < 0.02)):
      # We're spinning left
      self.DrawArrow(painter, 50, self.height / 2, 180)

    if leftSpeed > 0.02 and (abs(leftSpeed - rightSpeed) < 0.02):
      # We're going forward
      self.DrawArrow(painter, 150, 30, -90)

    if leftSpeed < -0.02 and (abs(leftSpeed - rightSpeed) < 0.02):
      # We're going backwards
      self.DrawArrow(painter, 150, self.height - 30, 90)

    if (leftSpeed > 0.02 and rightSpeed < -0.02 and
        (abs(leftSpeed) > (abs(rightSpeed) + 0.02))):
      # We're forward and to the right
      self.DrawArrow(painter, 230, 50, -45)

    if (leftSpeed < -0.02 and rightSpeed > 0.02 and
        (abs(rightSpeed) > (abs(leftSpeed) + 0.02))):
      # We're forward and to the left
      self.DrawArrow(painter, 70, 50, -135)

    if (leftSpeed > 0.02 and rightSpeed < -0.02 and
        (abs(rightSpeed) > (abs(leftSpeed) + 0.02))):
      # We're backwards and to the right
      self.DrawArrow(painter, 230, self.height - 50, 45)

    if (leftSpeed < -0.02 and rightSpeed > 0.02 and
        (abs(leftSpeed) > (abs(rightSpeed) + 0.02))):
      # We're forward and to the left
      self.DrawArrow(painter, 70, self.height - 50, 135)

class RedrawTimer(QtCore.QThread):
  '''A class that forces the UI to redraw and will then cause any signal handlers in the main thread to fire.'''
  def __init__(self, frame, parent = None):
    QtCore.QThread.__init__(self, parent)
    self.frame = frame

  def run(self):
    while True:
      self.RedrawFrame()
      self.sleep(1)

  def RedrawFrame(self):
    self.frame.update()

class ROSThread(QtCore.QThread):
  def __init__(self, parent = None):
    QtCore.QThread.__init__(self, parent)

  def run(self):
    rospy.spin()

def ShutdownROS():
  rospy.signal_shutdown("Qt Application is closing")

def Shutdown(app):
  app.exit(0)


def UpdateRobotStatus(msg, depthGauge, speedStatus):
  depthGauge.SetDepth(msg.depth)
  depthGauge.update()

  speedStatus.SetSpeedStatus(msg.left_speed, msg.right_speed,
                             msg.vertical_speed, msg.pitch)
  speedStatus.update()

def UpdateVideoFeed(msg, videoWidget):
  videoWidget.SetNewInputPath(msg.url.data)

if __name__ == "__main__":
  # We disable signals because we want the QT part of the application
  # to handle the various POSIX signals.
  rospy.init_node('HudUI', disable_signals=True)

  scriptDir = os.path.dirname(sys.argv[0])

  xDisplay = rospy.get_param("~x_display", ":0.0")
  sys.argv.extend(['-display', xDisplay])

  # Build up the UI
  qApp = Qt.QApplication(sys.argv)
  frame = Qt.QFrame()

  # These commands cause a segfault
  #screenInfo = display.Display(xDisplay).screen()
  #height = screenInfo.height_in_pixels
  #width = screenInfo.width_in_pixels
  height = 1080
  width = 1920

  videoWidget = VideoWidget(frame)
  videoWidget.setMinimumSize(width, height)

  depthGauge = DepthGauge(scriptDir, frame)
  depthGauge.move(20, height-800-20)

  cameraTarget = CameraTarget(width/2-330, height/2-218,
                              width/2+330, height/2+218,
                              50, frame)
  cameraTarget.move(width/2-330, height/2-218)

  speedStatus = SpeedStatus(scriptDir, frame)
  speedStatus.move(width - speedStatus.width,
                   height - speedStatus.height)
  frame.setWindowFlags(Qt.Qt.WindowStaysOnTopHint)
  frame.showFullScreen()

  # Set up the ROS callbacks
  statusCallback = lambda x: UpdateRobotStatus(x, depthGauge, speedStatus)
  rospy.Subscriber(rospy.get_param('robot_status_topic', 'robot_status'),
                   RobotStatus, statusCallback)

  videoCallback = lambda x: UpdateVideoFeed(x, videoWidget)
  rospy.Subscriber(rospy.get_param('video_stream_topic', 'video_stream'),
                   VideoStream, videoCallback)

  rosThread = ROSThread()
  rosThread.start()

  redrawTimerThread = RedrawTimer(frame)
  redrawTimerThread.start()

  # Setup the QT application to close the ROS connection cleanly
  qApp.aboutToQuit.connect(ShutdownROS)

  # Setup the signal handler that will close the Qt app on a SIGTERM.
  # For this to work, we need the redrawTimerThread because the signal
  # will be stuck in C land unless it is pulled back up into python by
  # a call to redraw the app.
  qtShutdown = lambda a,b: Shutdown(qApp)
  signal.signal(signal.SIGTERM, qtShutdown)

  # Start playing the default camera stream
  videoWidget.RestartStream()

  # Run the QT application
  qApp.exec_()
