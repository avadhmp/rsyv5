import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Point

def doMsg(msg):
    rospy.loginfo("point: %f, %f, %f, %f", msg.x, msg.y, msg.z, msg.c)

if __name__ == "__main__":
    rospy.init_node("SUB")
    pub = rospy.Publisher("/detect_result_out", Point, doMsg, queue_size = 10)
    rospy.spin()